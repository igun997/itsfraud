"""Extract DINOv2-base features (production 'norm' preprocessing) for public and gated receipt/invoice sources,
fit a head on the pod, and print the features, head and metrics. Images stay on the pod.

Sources (label, source name):
  public:  SROIE scans (0, public-sroie), GPT-Image-2 receipts (1, public-gpt-image-2)   Shuyiww/TextRich
           CORD photos (0, public-cord)                                                  naver-clova-ix/cord-v2
           real invoice/receipt photos (0, public-invoices-pt)                           Francisco-Cruz/InvoicesReceiptsPT
  gated (needs HF_TOKEN from an account that accepted the terms; CC-BY-NC-SA-4.0, non-commercial):
           GPT-4o receipts (1, gated-gpt4o-receipt)                                      Scam-AI/gpt4o-receipt
           authentic documents (0, gated-aiforge-authentic) and AI-inpainted edits of them (1, gated-aiforge-edited)   Scam-AI/AIForge-Doc-v1
Output blocks (base64 with sha256): SRC_FEATURES (meta + float16 features), SRC_RESULTS (metrics + head, JSON).

  python bench/extract_sources.py --work /workspace/src
  python bench/extract_sources.py --selftest
"""
import argparse
import base64
import hashlib
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from common import normalize, page_tiles  # train/common.py; the single-file pod build inlines these
except ImportError:
    pass
from gen_fakes import generate
from run_benchmark import Backbone, extract_all, log, thr_at_fpr

PREPROCESS = "long side 1024 (Lanczos), JPEG q75, up to 8 native 224px tiles per page, ImageNet mean/std"


def bucket(name):
    """Deterministic 70/15/15 split from the file name."""
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def fetch_all(work: Path, a):
    from huggingface_hub import hf_hub_download, list_repo_files

    tok = os.environ.get("HF_TOKEN") or None
    rng = random.Random(0)
    items = []

    def get(repo, files, label, source, split=None, token=None):
        def one(f):
            return hf_hub_download(repo, f, repo_type="dataset", local_dir=str(work / repo.replace("/", "__")), token=token)

        with ThreadPoolExecutor(16) as ex:
            paths = list(ex.map(one, files))
        for p, f in zip(paths, files):
            items.append(dict(path=p, label=label, source=source, split=split or bucket(f), name=f))
        log(f"{source}: {len(files)} files")

    tr = [f for f in list_repo_files("Shuyiww/TextRich", repo_type="dataset") if f.startswith("Data/Receipts/") and f.lower().endswith((".png", ".jpg", ".jpeg"))]
    get("Shuyiww/TextRich", rng.sample(sorted(f for f in tr if "/0_real/" in f), a.n_sroie), 0, "public-sroie")
    get("Shuyiww/TextRich", rng.sample(sorted(f for f in tr if "/1_fake/" in f), a.n_gpt), 1, "public-gpt-image-2")
    inv = sorted(f for f in list_repo_files("Francisco-Cruz/InvoicesReceiptsPT", repo_type="dataset") if f.startswith("1_Images/") and f.endswith(".jpg"))
    get("Francisco-Cruz/InvoicesReceiptsPT", rng.sample(inv, min(a.n_invoices, len(inv))), 0, "public-invoices-pt")

    import pyarrow.parquet as pq

    cord, out = [], work / "cord"
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(x for x in list_repo_files("naver-clova-ix/cord-v2", repo_type="dataset") if x.startswith("data/") and x.endswith(".parquet")):
        pf = pq.ParquetFile(hf_hub_download("naver-clova-ix/cord-v2", f, repo_type="dataset", local_dir=str(work / "cord_hf")))
        for rg in range(pf.num_row_groups):
            for rec in pf.read_row_group(rg, columns=["image"]).column("image").to_pylist():
                p = out / f"c{len(cord):05d}.img"
                p.write_bytes(rec["bytes"])
                cord.append(str(p))
    for p in rng.sample(cord, min(a.n_cord, len(cord))):
        items.append(dict(path=p, label=0, source="public-cord", split=bucket(p), name=p))
    log(f"public-cord: {min(a.n_cord, len(cord))} files")

    if not tok:
        log("HF_TOKEN not set: skipping gated sources")
        return items
    try:
        files = [f for f in list_repo_files("Scam-AI/gpt4o-receipt", repo_type="dataset", token=tok) if f.endswith(".png")]
        get("Scam-AI/gpt4o-receipt", sorted(files), 1, "gated-gpt4o-receipt", token=tok)
        files = list_repo_files("Scam-AI/AIForge-Doc-v1", repo_type="dataset", token=tok)
        for sub, label, source in (("authentic", 0, "gated-aiforge-authentic"), ("images", 1, "gated-aiforge-edited")):
            for part, n, split in (("TrainingSet", a.n_aiforge, None), ("TestingSet", a.n_aiforge // 2, "test")):
                cand = sorted(f for f in files if f.startswith(f"{part}/{sub}/") and f.endswith(".png"))
                get("Scam-AI/AIForge-Doc-v1", rng.sample(cand, min(n, len(cand))), label, source, split=split, token=tok)
    except Exception as e:  # access problem: keep going with what we have
        log(f"gated download failed: {e!r}"[:300])
    # AIForge training files: keep the dataset's own test split, carve val out of train only
    for it in items:
        if it["source"].startswith("gated-aiforge") and it["split"] != "test":
            it["split"] = "val" if int(hashlib.sha1(it["name"].encode()).hexdigest(), 16) % 100 < 18 else "train"
    return items


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


def rate(s, thr, m):
    return None if not m.any() else round(float((s[m] > thr).mean()), 4)


def report(items, F, exclude=()):
    from sklearn.metrics import roc_auc_score

    y = np.array([it["label"] for it in items])
    src = np.array([it["source"] for it in items])
    sp = np.array([it["split"] for it in items])
    out = {"counts": {s: {k: int(((src == s) & (sp == k)).sum()) for k in ("train", "val", "test")} for s in sorted(set(src))}, "excluded_from_training": sorted(exclude)}
    skip = np.array([s in exclude for s in src])

    def gen_of(s):
        return s.split("/")[0]  # gen-sdxl/photo -> gen-sdxl: hold out a whole generator, not one style

    def run(excl):
        tr, va = (sp == "train") & ~excl & ~skip, (sp == "val") & ~excl & ~skip
        sc, clf, C = fit(F[tr], y[tr], F[va], y[va])
        s = clf.decision_function(sc.transform(F))
        return sc, clf, C, s, thr_at_fpr(s[va], y[va], 0.01), thr_at_fpr(s[va], y[va], 0.05)

    out["leave_one_fake_source_out"] = {}
    for g in sorted({gen_of(s) for s, l in zip(src, y) if l == 1 and s not in exclude}):
        held = np.array([gen_of(s) == g for s in src]) & (y == 1)
        _, _, C, s, t1, t5 = run(held)
        out["leave_one_fake_source_out"][g] = {"n": int(held.sum()), "flagged@fpr1": rate(s, t1, held), "flagged@fpr5": rate(s, t5, held),
                                              "real_test_fpr@1": rate(s, t1, (y == 0) & (sp == "test")), "real_test_fpr@5": rate(s, t5, (y == 0) & (sp == "test"))}
    sc, clf, C, s, t1, t5 = run(np.zeros(len(y), bool))
    te = sp == "test"
    out["final"] = {"C": C, "thr_fpr1": t1, "thr_fpr5": t5, "test_auc": round(float(roc_auc_score(y[te], s[te])), 4),
                    "by_source_test": {n: {"n": int((te & (src == n)).sum()), "flagged@fpr1": rate(s, t1, te & (src == n)), "flagged@fpr5": rate(s, t5, te & (src == n))} for n in sorted(set(src)) if (te & (src == n)).any()}}
    head = dict(model_version="0.3.0-podfit", backbone="facebook/dinov2-base", preprocess=PREPROCESS, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]), threshold=t1, threshold_fpr1=t1, threshold_fpr5=t5)
    return out, head


def emit(tag, payload: bytes):
    b64 = base64.b64encode(payload).decode()
    print(f"==%s_BEGIN== sha256={hashlib.sha256(payload).hexdigest()} bytes={len(payload)}" % tag)
    for i in range(0, len(b64), 12000):
        print(b64[i : i + 12000])
    print("==%s_END==" % tag, flush=True)


def selftest():
    rng = np.random.default_rng(0)
    items, feats = [], []
    for label, source, n, shift in ((0, "public-sroie", 150, 0), (0, "public-cord", 150, 0.2), (0, "gated-aiforge-authentic", 120, 0.1), (1, "public-gpt-image-2", 150, 2.0), (1, "gated-gpt4o-receipt", 150, 1.5), (1, "gated-aiforge-edited", 120, 0.5), (1, "gen-sdxl/photo", 100, 1.2), (1, "gen-z-image-turbo/photo", 100, 1.4)):
        for i in range(n):
            items.append(dict(label=label, source=source, split=bucket(f"{source}{i}"), name=f"{source}{i}"))
            feats.append(rng.normal(0, 1, 32) + shift)
    out, head = report(items, np.array(feats), exclude=("gated-aiforge-edited",))
    assert out["final"]["test_auc"] > 0.7 and len(head["coef"]) == 32 and "gen-sdxl" in out["leave_one_fake_source_out"] and "gated-aiforge-edited" not in out["leave_one_fake_source_out"]
    print(json.dumps(out["leave_one_fake_source_out"])[:300])
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/workspace/src"))
    ap.add_argument("--n-sroie", type=int, default=300)
    ap.add_argument("--n-gpt", type=int, default=300)
    ap.add_argument("--n-cord", type=int, default=500)
    ap.add_argument("--n-invoices", type=int, default=400)
    ap.add_argument("--n-aiforge", type=int, default=400, help="per class from TrainingSet; half as many from TestingSet")
    ap.add_argument("--n-zimage", type=int, default=160)
    ap.add_argument("--n-sdxl", type=int, default=120)
    ap.add_argument("--gen-minutes", type=float, default=7, help="time budget per generator")
    ap.add_argument("--exclude-train", default="gated-aiforge-edited", help="comma list of sources used for testing only")
    ap.add_argument("--emit-features", action="store_true", help="also print the feature blob (several MB; the log tool truncates it)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import torch

    torch.set_num_threads(8)
    a.work.mkdir(parents=True, exist_ok=True)
    gen_imgs, gen_items = [], []
    for name, n, seed in (("z-image-turbo", a.n_zimage, 1), ("sdxl", a.n_sdxl, 3)):
        if n <= 0:
            continue
        try:
            got = generate(name, n, a.work, time.time() + a.gen_minutes * 60, False, seed)
        except Exception as e:  # keep what the other generator made
            log(f"{name} FAILED: {e!r}"[:300])
            continue
        for k, (img, pr) in enumerate(got):
            gen_imgs.append(img)
            gen_items.append(dict(label=1, source=f"gen-{name}/{pr['style']}", split=bucket(f"{name}|{k}"), name=f"{name}|{k}"))
        log(f"{name}: {len(got)} images")
    items = fetch_all(a.work, a)
    for i, it in enumerate(items):
        it["id"] = i
    log(f"{len(items)} files; extracting features")
    bb = {"dino_b": Backbone("dino", "facebook/dinov2-base", "cuda")}
    F = extract_all(items, "norm", bb, "cuda", a.workers, "src")["dino_b"]
    ok = np.isfinite(F).all(1) & (np.abs(F).sum(1) > 0)
    items = [it for it, k in zip(items, ok) if k]
    F = F[ok]
    if gen_imgs:
        import torch
        Fg = []
        for img in gen_imgs:
            arr = np.stack([np.asarray(t) for t in page_tiles(normalize(img))])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to("cuda")
            Fg.append(bb["dino_b"].embed(x).float().cpu().numpy().mean(0))
        items += gen_items
        F = np.vstack([F, np.stack(Fg)])
    if a.emit_features:
        meta = [{k: it[k] for k in ("source", "label", "split", "name")} for it in items]
        buf = io.BytesIO()
        np.savez_compressed(buf, feats=F.astype(np.float16))
        emit("SRC_FEATURES", json.dumps(meta).encode() + b"\n--NPZ--\n" + buf.getvalue())
    res, head = report(items, F, exclude=tuple(x for x in a.exclude_train.split(",") if x))
    emit("SRC_RESULTS", json.dumps(dict(results=res, head=head)).encode())
    log(json.dumps(res["final"]))


if __name__ == "__main__":
    main()
