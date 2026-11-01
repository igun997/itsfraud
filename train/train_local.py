"""Retrain the head locally, on CPU, with your own data. Your files never leave this machine.

Your data (git-ignored):
  data/real/**                 genuine receipts/invoices (images or PDFs); subfolders become sources (data/real/invoices -> user-real/invoices)
  data/fake/<generator>/**     AI-generated ones, one folder per generator (data/fake/gemini); files directly in data/fake -> "unspecified"
Public data (downloaded and cached under artifacts/public): SROIE scans and GPT-Image-2 fakes (Shuyiww/TextRich), CORD photos (cord-v2),
real invoice/receipt photos (Francisco-Cruz/InvoicesReceiptsPT, Apache-2.0).

Two experiments, then the final head:
  1. unseen generator: train WITHOUT your fakes; test on all of them (how the current kind of model does on a generator it never saw)
  2. leave-one-generator-out over your fake folders (needs two or more folders)
  3. final: 70/15/15 split per (label, source), thresholds from the validation split, one pass over the test split

  python train/train_local.py --out models/v0.2.0
  python train/train_local.py --selftest
Uses models/v0.1.0/backbone.onnx (frozen DINOv2-base) for features; preprocessing is train/common.py (same as the Rust service).
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMAGE_EXTS, augment, load_pages, normalize, page_tiles, to_tensor_array  # noqa: E402

VERSION = "0.2.0"
PREPROCESS = "long side 1024 (Lanczos), JPEG q75, up to 8 native 224px tiles per page, ImageNet mean/std"
EXTS = IMAGE_EXTS | {".pdf"}


def thr_at_fpr(s, y, fpr):
    neg = np.sort(s[y == 0])
    k = int(np.ceil((1 - fpr) * len(neg))) - 1
    return float(neg[min(max(k, 0), len(neg) - 1)])


def auc(s, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None


def rate(s, thr, mask):
    return None if not mask.any() else round(float((s[mask] > thr).mean()), 4)


def fit(Xtr, ytr, Xva, yva):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1):
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        a = roc_auc_score(yva, clf.decision_function(sc.transform(Xva))) if len(set(yva.tolist())) == 2 else 0.0
        if best is None or a > best[0]:
            best = (a, C, clf)
    return sc, best[2], best[1]


def collect_user(root: Path):
    items = []
    for label, kind in ((0, "real"), (1, "fake")):
        base = root / kind
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                rel = p.relative_to(base).parts[:-1]
                sub = "/".join(rel) if rel else ("unspecified" if kind == "fake" else "general")
                items.append(dict(path=str(p), label=label, source=f"user-{kind}/{sub}"))
    return items


def collect_invoices(cache: Path, n: int, seed=0):
    """n real invoice/receipt phone photos from Francisco-Cruz/InvoicesReceiptsPT (Apache-2.0)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    fs = sorted(f for f in list_repo_files("Francisco-Cruz/InvoicesReceiptsPT", repo_type="dataset") if f.startswith("1_Images/") and f.lower().endswith(".jpg"))
    pick = random.Random(seed).sample(fs, min(n, len(fs)))
    return [dict(path=hf_hub_download("Francisco-Cruz/InvoicesReceiptsPT", f, repo_type="dataset", local_dir=str(cache / "invoices_pt")), label=0, source="public-invoices-pt") for f in pick]


def collect_public(cache: Path, n: int, seed=0):
    """n files each of SROIE real, GPT-Image-2 fake and CORD real, downloaded and cached."""
    from huggingface_hub import hf_hub_download, list_repo_files

    rng = random.Random(seed)
    files = list_repo_files("Shuyiww/TextRich", repo_type="dataset")
    items = []
    for lab_dir, label, source in (("0_real", 0, "public-sroie"), ("1_fake", 1, "public-gpt-image-2")):
        fs = sorted(f for f in files if f.startswith(f"Data/Receipts/{lab_dir}/") and f.lower().endswith((".png", ".jpg", ".jpeg")))
        for f in rng.sample(fs, min(n, len(fs))):
            items.append(dict(path=hf_hub_download("Shuyiww/TextRich", f, repo_type="dataset", local_dir=str(cache / "textrich")), label=label, source=source))
    import pyarrow.parquet as pq

    out = cache / "cord"
    out.mkdir(parents=True, exist_ok=True)
    cord = []
    for f in ("data/test-00000-of-00001-9c204eb3f4e11791.parquet", "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"):
        pf = pq.ParquetFile(hf_hub_download("naver-clova-ix/cord-v2", f, repo_type="dataset", local_dir=str(cache / "cord_hf")))
        for rg in range(pf.num_row_groups):
            for rec in pf.read_row_group(rg, columns=["image"]).column("image").to_pylist():
                p = out / f"c{len(cord):05d}.img"
                p.write_bytes(rec["bytes"])
                cord.append(str(p))
    for p in rng.sample(cord, min(n, len(cord))):
        items.append(dict(path=p, label=0, source="public-cord"))
    return items


class Embedder:
    def __init__(self, onnx_path, threads=8):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def __call__(self, arr):  # arr: (n,3,224,224) float32 -> (n,768)
        out = [self.sess.run(None, {"pixel_values": arr[i : i + 16]})[0] for i in range(0, len(arr), 16)]
        return np.concatenate(out)


def file_feature(embed, path, variant, seed):
    pages = load_pages(Path(path))
    rng = random.Random(seed)
    tiles = []
    for p in pages:
        q = normalize(p)
        if variant != "norm":
            q = augment(q, rng)
        tiles += page_tiles(q)
    return embed(to_tensor_array(tiles)).mean(0)


def collect_generated(folder: Path):
    """Features of fakes made by bench/gen_fakes.py (decoded by scripts/decode_gen_fakes.py); one source per generator and style."""
    if not (folder / "features.npz").exists():
        return []
    feats = np.load(folder / "features.npz")["feats"]
    meta = json.loads((folder / "meta.json").read_text())
    return [dict(path=None, label=1, source=f"gen-{m['generator']}/{m['style']}", pre=dict(key=f"gen|{m['generator']}|{i}", feats=feats[i])) for i, m in enumerate(meta)]


def featurize(items, embed, cache_file, aug_copies):
    cache = dict(np.load(cache_file, allow_pickle=True)["cache"].item()) if cache_file.exists() else {}
    todo = 0
    for it in items:
        if "pre" in it:  # features already computed elsewhere (generated fakes)
            it["key"] = it["pre"]["key"]
            for v, f in zip(("norm", "aug1", "aug2"), it["pre"]["feats"]):
                cache[f"{it['key']}|{v}"] = f.astype(np.float32)
            continue
        st = Path(it["path"]).stat()
        base = hashlib.sha1(f"{it['path']}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
        it["key"] = base
        for v in ["norm"] + [f"aug{k}" for k in range(1, aug_copies + 1)]:
            if f"{base}|{v}" not in cache:
                todo += 1
    print(f"features to compute: {todo} (cached: {len(cache)})", flush=True)
    done = 0
    for i, it in enumerate(items):
        if "pre" in it:
            continue
        for v in ["norm"] + [f"aug{k}" for k in range(1, aug_copies + 1)]:
            k = f"{it['key']}|{v}"
            if k not in cache:
                try:
                    cache[k] = file_feature(embed, it["path"], v, seed=i * 10 + (0 if v == "norm" else int(v[3:])))
                except Exception as e:  # unreadable file: skip, do not poison the set
                    print("skip", it["path"], repr(e)[:100])
                    cache[k] = None
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{todo}", flush=True)
                    np.savez_compressed(cache_file, cache=np.array(cache, dtype=object))
    np.savez_compressed(cache_file, cache=np.array(cache, dtype=object))
    return cache


def split(items, seed=0):
    rng = np.random.default_rng(seed)
    groups, out = {}, {}
    for i, it in enumerate(items):
        groups.setdefault((it["label"], it["source"]), []).append(i)
    for ids in groups.values():
        ids = np.array(ids)
        rng.shuffle(ids)
        a, b = int(0.7 * len(ids)), int(0.85 * len(ids))
        for j, i in enumerate(ids):
            out[int(i)] = "train" if j < a else "val" if j < b else "test"
    return out


def run(items, cache, aug_copies):
    items = [it for it in items if cache.get(f"{it['key']}|norm") is not None]
    y = np.array([it["label"] for it in items])
    src = np.array([it["source"] for it in items])
    X = np.stack([cache[f"{it['key']}|norm"] for it in items])
    sp_map = split(items)
    sp = np.array([sp_map[i] for i in range(len(items))])

    def aug_rows(mask):
        rows = [np.stack([cache[f"{it['key']}|aug{k}"] for it, m in zip(items, mask) if m]) for k in range(1, aug_copies + 1) if all(cache.get(f"{it['key']}|aug{k}") is not None for it, m in zip(items, mask) if m)]
        return np.concatenate(rows) if rows else np.zeros((0, X.shape[1]))

    res = {"counts": {f"{s}": int((src == s).sum()) for s in sorted(set(src))}}
    user_fake = np.array([s.startswith("user-fake") for s in src])
    gen_any = user_fake | np.array([s.startswith("gen-") for s in src])
    user_real = np.array([s.startswith("user-real") for s in src])

    # 1. unseen generator: no user fakes in training
    if user_fake.any() and user_real.any():
        tr = (sp == "train") & ~user_fake
        va = (sp == "val") & ~user_fake
        sc, clf, C = fit(X[tr], y[tr], X[va], y[va])
        f = lambda Z: clf.decision_function(sc.transform(Z))
        s = f(X)
        t1, t5 = thr_at_fpr(s[va], y[va], 0.01), thr_at_fpr(s[va], y[va], 0.05)
        e = {"C": C, "thr_fpr1": t1, "thr_fpr5": t5, "auc_user_fake_vs_user_real_test": auc(np.r_[s[user_fake], s[user_real & (sp == "test")]], np.r_[np.ones(user_fake.sum()), np.zeros((user_real & (sp == "test")).sum())]), "by_source": {}}
        for name in sorted(set(src)):
            m = (src == name) & (user_fake | (sp == "test"))
            e["by_source"][name] = {"n": int(m.sum()), "flagged@fpr1": rate(s, t1, m), "flagged@fpr5": rate(s, t5, m), "median_logit": round(float(np.median(s[m])), 2)}
        res["1_unseen_generator"] = e

    # 2. leave-one-generator-out: hold out every fake source of one generator at a time (public, generated, or yours)
    def gen_of(name):
        return name.split("/")[0] if name.startswith(("gen-", "user-fake")) else name

    gens = sorted({gen_of(s) for s, l in zip(src, y) if l == 1})
    if len(gens) >= 2:
        res["2_leave_one_generator_out"] = {}
        for g in gens:
            held = np.array([gen_of(s) == g and l == 1 for s, l in zip(src, y)])
            tr = (sp == "train") & ~held
            va = (sp == "val") & ~held
            sc, clf, C = fit(X[tr], y[tr], X[va], y[va])
            s = clf.decision_function(sc.transform(X))
            t1, t5 = thr_at_fpr(s[va], y[va], 0.01), thr_at_fpr(s[va], y[va], 0.05)
            res["2_leave_one_generator_out"][g] = {"flagged@fpr1": rate(s, t1, held), "flagged@fpr5": rate(s, t5, held), "n": int(held.sum())}

    # 3. final
    tr, va, te = (sp == k for k in ("train", "val", "test"))
    Xa = np.concatenate([X[tr], aug_rows(tr)])
    ya = np.concatenate([y[tr]] * (1 + (len(Xa) // max(1, tr.sum()) - 1)))
    sc, clf, C = fit(Xa, ya[: len(Xa)], X[va], y[va])
    s = clf.decision_function(sc.transform(X))
    t1, t5 = thr_at_fpr(s[va], y[va], 0.01), thr_at_fpr(s[va], y[va], 0.05)
    fin = {"C": C, "thr_fpr1": t1, "thr_fpr5": t5, "test_auc": auc(s[te], y[te]), "by_source_test": {}}
    for name in sorted(set(src)):
        m = te & (src == name)
        if m.any():
            fin["by_source_test"][name] = {"n": int(m.sum()), "flagged@fpr1": rate(s, t1, m), "flagged@fpr5": rate(s, t5, m)}
    res["3_final"] = fin
    head = dict(model_version=VERSION, backbone="facebook/dinov2-base", preprocess=PREPROCESS, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]), threshold=t1, threshold_fpr1=t1, threshold_fpr5=t5)
    return res, head


def selftest():
    rng = np.random.default_rng(0)
    items, cache = [], {}
    for label, source, n in ((0, "public-sroie", 60), (0, "public-cord", 60), (1, "public-gpt-image-2", 60), (0, "user-real/general", 40), (1, "user-fake/gemini", 40), (1, "user-fake/other", 30)):
        for _ in range(n):
            k = f"k{len(items)}"
            items.append(dict(path=k, label=label, source=source, key=k))
            base = rng.normal(0, 1, 16) + (1.5 * label if source != "user-fake/gemini" else 0.4)
            cache[f"{k}|norm"] = base
            cache[f"{k}|aug1"] = base + rng.normal(0, 0.1, 16)
    res, head = run(items, cache, 1)
    assert "1_unseen_generator" in res and "2_leave_one_generator_out" in res and len(head["coef"]) == 16
    print(json.dumps({k: res[k] for k in ("1_unseen_generator", "2_leave_one_generator_out")}, indent=1)[:900])
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--public-cache", type=Path, default=Path("artifacts/public"))
    ap.add_argument("--n-public", type=int, default=250)
    ap.add_argument("--gen-fakes", type=Path, default=Path("artifacts/gen_fakes"), help="decoded output of bench/gen_fakes.py")
    ap.add_argument("--n-invoices", type=int, default=200, help="real invoice photos from InvoicesReceiptsPT")
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--backbone", type=Path, default=Path("models/v0.1.0/backbone.onnx"))
    ap.add_argument("--out", type=Path, default=Path("models/v0.2.0"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    a.public_cache.mkdir(parents=True, exist_ok=True)
    user = collect_user(a.data)
    if not any(it["label"] == 0 for it in user):
        sys.exit("need real files in data/real")
    items = collect_public(a.public_cache, a.n_public) + collect_invoices(a.public_cache, a.n_invoices) + collect_generated(a.gen_fakes) + user
    print(f"{len(items)} files:", {s: sum(1 for it in items if it['source'] == s) for s in sorted({it['source'] for it in items})}, flush=True)
    cache = featurize(items, Embedder(a.backbone), a.public_cache / "features.npz", a.aug_copies)
    res, head = run(items, cache, a.aug_copies)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "head.json").write_text(json.dumps(head))
    (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"wrote {a.out}/head.json (backbone: reuse {a.backbone})")


if __name__ == "__main__":
    main()
