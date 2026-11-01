"""Benchmark frozen-backbone detectors for AI-generated receipts on the public TextRich dataset
(Shuyiww/TextRich: real SROIE receipts vs GPT-Image-2 fakes, plus other document domains).

Self-contained for a rented GPU: downloads data, extracts tile features for several backbones,
trains linear heads, evaluates, prints a JSON blob and markdown tables to stdout.

  python bench/run_benchmark.py --work /workspace/bench
  python bench/run_benchmark.py --selftest          # CPU, synthetic features, no downloads

Conditions (why they exist): real and fake files differ in size/format, so a detector can win by
reading the file pipeline instead of AI artifacts. `raw` keeps files as shipped; `norm` resizes every
image to long side 1024 and JPEG-75 re-encodes it so both classes share one pipeline; `pp` adds
downscale + JPEG-50 (WhatsApp-like) on top of `norm` to test robustness.
"""
import argparse
import io
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import augment, page_tiles  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
REPO = "Shuyiww/TextRich"
RECEIPTS = "Receipts"
CROSS_DOMAINS = ["Tables", "UIScreenshots", "InfographicCharts"]
BACKBONES = {  # name -> (kind, hf id)
    "dino_s": ("dino", "facebook/dinov2-small"),
    "dino_b": ("dino", "facebook/dinov2-base"),
    "clip_b16": ("clip", "openai/clip-vit-base-patch16"),
}
SEED = 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- image variants
def jpeg_roundtrip(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def v_raw(img, _):
    return img


def v_norm(img, _):
    w, h = img.size
    s = 1024 / max(w, h)
    return jpeg_roundtrip(img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS), 75)


def v_pp(img, _):
    img = v_norm(img, None)
    return jpeg_roundtrip(img.resize((max(1, int(img.width * 0.6)), max(1, int(img.height * 0.6))), Image.BILINEAR), 50)


def v_aug(k):
    return lambda img, seed: augment(v_norm(img, None), random.Random(seed * 10 + k))


VARIANTS = {"raw": v_raw, "norm": v_norm, "pp": v_pp, "aug1": v_aug(1), "aug2": v_aug(2)}


# ---------------------------------------------------------------- data
def build_items(work: Path, per_label_cross: int, limit: int | None):
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    rng = random.Random(SEED)
    chosen = []  # (path_in_repo, label, domain)
    for dom in [RECEIPTS] + CROSS_DOMAINS:
        for lab_dir, lab in (("0_real", 0), ("1_fake", 1)):
            fs = sorted(f for f in files if f.startswith(f"Data/{dom}/{lab_dir}/"))
            if dom != RECEIPTS:
                rng.shuffle(fs)
                fs = fs[:per_label_cross]
            elif limit:
                rng.shuffle(fs)
                fs = fs[: limit // 2]
            chosen += [(f, lab, dom) for f in fs]
    log(f"downloading {len(chosen)} files from {REPO}")

    def get(c):
        try:
            return hf_hub_download(REPO, c[0], repo_type="dataset", local_dir=str(work / "data"))
        except Exception as e:  # skip, keep going
            log("download failed", c[0], repr(e)[:120])
            return None

    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(get, chosen))
    items = [dict(path=p, label=c[1], domain=c[2]) for p, c in zip(paths, chosen) if p]
    for i, it in enumerate(items):
        it["id"] = i
    log(f"have {len(items)} files")
    return items


def split_receipts(items):
    rng = np.random.default_rng(SEED)
    split = {}
    for lab in (0, 1):
        ids = np.array([it["id"] for it in items if it["domain"] == RECEIPTS and it["label"] == lab])
        rng.shuffle(ids)
        n = len(ids)
        a, b = int(0.7 * n), int(0.85 * n)
        for j, i in enumerate(ids):
            split[int(i)] = "train" if j < a else "val" if j < b else "test"
    return split


def file_stats(items):
    def one(it):
        with Image.open(it["path"]) as im:
            w, h = im.size
            fmt, mode = im.format, im.mode
        return dict(w=w, h=h, size=os.path.getsize(it["path"]), fmt=fmt, mode=mode)

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, items))


def meta_features(stats):
    X = []
    for s in stats:
        px = s["w"] * s["h"]
        X.append([s["w"], s["h"], px, s["w"] / s["h"], s["size"], s["size"] / px, s["fmt"] == "PNG", s["fmt"] == "JPEG", s["mode"] == "RGB", s["mode"] == "RGBA"])
    return np.array(X, dtype=np.float64)


def describe(items, stats, split):
    out = {}
    for lab, name in ((0, "real"), (1, "fake")):
        sel = [s for it, s in zip(items, stats) if it["domain"] == RECEIPTS and it["label"] == lab]
        if not sel:
            continue
        fm = {}
        for s in sel:
            fm[s["fmt"]] = fm.get(s["fmt"], 0) + 1
        out[name] = dict(
            n=len(sel),
            median_w=int(np.median([s["w"] for s in sel])),
            median_h=int(np.median([s["h"] for s in sel])),
            median_kb=round(float(np.median([s["size"] for s in sel])) / 1024, 1),
            formats=fm,
        )
    out["splits"] = {k: sum(1 for v in split.values() if v == k) for k in ("train", "val", "test")}
    return out


# ---------------------------------------------------------------- feature extraction (GPU)
def extract_all(items, variant, backbones, device, workers, tag):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            it = items[i]
            img = VARIANTS[variant](Image.open(it["path"]).convert("RGB"), it["id"])
            arr = np.stack([np.asarray(t) for t in page_tiles(img)])
            return i, torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

    out = {n: np.zeros((len(items), b.dim), np.float32) for n, b in backbones.items()}
    buf, t0 = [], time.time()

    def flush():
        x = torch.cat([t for _, t in buf]).to(device)
        counts = [len(t) for _, t in buf]
        idx = [i for i, _ in buf]
        for n, b in backbones.items():
            f = b.embed(x).float().cpu().numpy()
            o = 0
            for i, c in zip(idx, counts):
                out[n][i] = f[o : o + c].mean(0)
                o += c
        buf.clear()

    dl = DataLoader(DS(), batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4 if workers else None)
    for k, (i, t) in enumerate(dl):
        buf.append((i, t))
        if sum(len(x) for _, x in buf) >= 96:
            flush()
        if k % 200 == 0:
            log(f"[{tag}:{variant}] {k}/{len(items)} ({time.time() - t0:.0f}s)")
    if buf:
        flush()
    log(f"[{tag}:{variant}] done in {time.time() - t0:.0f}s")
    return out


class Backbone:
    def __init__(self, kind, model_id, device):
        import torch

        self.kind, self.device = kind, device
        if kind == "dino":
            from transformers import AutoModel

            self.m = AutoModel.from_pretrained(model_id)
            self.dim = self.m.config.hidden_size
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            from transformers import CLIPModel

            self.m = CLIPModel.from_pretrained(model_id)
            self.dim = self.m.config.projection_dim
            mean, std = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)
        self.m = self.m.to(device).eval()
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def embed(self, x_u8):
        import torch

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            x = (x_u8.float() / 255.0 - self.mean) / self.std
            if self.kind == "dino":
                return self.m(pixel_values=x).last_hidden_state[:, 0]
            return self.m.visual_projection(self.m.vision_model(pixel_values=x).pooler_output)


# ---------------------------------------------------------------- evaluation
def fit_head(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        auc = roc_auc_score(yva, clf.decision_function(sc.transform(Xva)))
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return (lambda X: best[2].decision_function(sc.transform(X))), best[1]


def thr_at_fpr(s, y, fpr):
    neg = np.sort(s[y == 0])
    k = int(np.ceil((1 - fpr) * len(neg))) - 1
    return float(neg[min(max(k, 0), len(neg) - 1)])


def metrics(s, y, thr1, thr5, rng):
    if len(set(y.tolist())) < 2:
        return None
    aucs = []
    for _ in range(500):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b].tolist())) == 2:
            aucs.append(roc_auc_score(y[b], s[b]))
    r = dict(
        n=int(len(y)),
        auc=round(float(roc_auc_score(y, s)), 4),
        auc_ci95=[round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)],
    )
    for name, thr in (("fpr1", thr1), ("fpr5", thr5)):
        r[f"tpr@{name}thr"] = round(float((s[y == 1] > thr).mean()), 4)
        r[f"fpr@{name}thr"] = round(float((s[y == 0] > thr).mean()), 4)
    return r


def evaluate(score_fn, Xval, yval, tests, rng):
    sval = score_fn(Xval)
    thr1, thr5 = thr_at_fpr(sval, yval, 0.01), thr_at_fpr(sval, yval, 0.05)
    return {name: metrics(score_fn(X), y, thr1, thr5, rng) for name, (X, y) in tests.items()}


def run_experiments(feat, y, split, domains, backbone_names, meta=None):
    """feat[variant][backbone] -> (n_items, dim) aligned with `y`/`split`/`domains` over all items;
    variants not computed for an item are zeros and never selected."""
    rng = np.random.default_rng(1)
    ids = np.arange(len(y))
    is_rec = domains == RECEIPTS
    sp = np.array([split.get(int(i), "") for i in ids])
    tr, va, te = (is_rec & (sp == s) for s in ("train", "val", "test"))
    results = {}

    def cross(var, b):
        return {f"x:{d}": (feat[var][b][domains == d], y[domains == d]) for d in CROSS_DOMAINS if (domains == d).any()}

    for b in backbone_names:
        F = {v: feat[v][b] for v in feat}
        # A: train raw -> test raw, and raw-trained model on normalized files (shortcut probe)
        fn, C = fit_head(F["raw"][tr], y[tr], F["raw"][va], y[va])
        results[f"{b}|raw->raw"] = dict(C=C, **evaluate(fn, F["raw"][va], y[va], {"test_raw": (F["raw"][te], y[te]), "test_norm": (F["norm"][te], y[te])}, rng))
        # B: normalized pipeline
        fn, C = fit_head(F["norm"][tr], y[tr], F["norm"][va], y[va])
        tests = {"test_norm": (F["norm"][te], y[te]), "test_pp": (F["pp"][te], y[te]), **cross("norm", b)}
        results[f"{b}|norm->norm"] = dict(C=C, **evaluate(fn, F["norm"][va], y[va], tests, rng))
        # C: normalized + augmented training copies (aug features exist for train items only)
        Xa = np.concatenate([F["norm"][tr], F["aug1"][tr], F["aug2"][tr]])
        ya = np.concatenate([y[tr]] * 3)
        fn, C = fit_head(Xa, ya, F["norm"][va], y[va])
        results[f"{b}|norm+aug->norm"] = dict(C=C, **evaluate(fn, F["norm"][va], y[va], tests, rng))
    if meta is not None:  # shortcut baseline: file metadata only (raw files)
        for name, mk in (("hgb", lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0)),):
            clf = mk().fit(meta[tr], y[tr])
            fn = lambda X, c=clf: c.predict_proba(X)[:, 1]
            results[f"metadata_only({name})|raw->raw"] = evaluate(fn, meta[va], y[va], {"test_raw": (meta[te], y[te])}, rng)
    return results


def markdown(results):
    cols = ["test_raw", "test_norm", "test_pp", "x:Tables", "x:UIScreenshots", "x:InfographicCharts"]
    lines = ["| model | trained on | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k, v in results.items():
        name, cond = k.split("|")
        cells = []
        for c in cols:
            m = v.get(c)
            cells.append("" if not m else f"AUC {m['auc']:.3f} (TPR {m['tpr@fpr5thr']:.2f} / FPR {m['fpr@fpr5thr']:.2f})")
        lines.append(f"| {name} | {cond} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def selftest():
    rng = np.random.default_rng(0)
    n = 600
    y = np.array([0, 1] * (n // 2))
    domains = np.array([RECEIPTS] * 400 + CROSS_DOMAINS[0:1] * 100 + CROSS_DOMAINS[1:2] * 100)
    items = [dict(id=i, domain=d, label=int(l)) for i, (d, l) in enumerate(zip(domains, y))]
    split = split_receipts(items)
    feat = {v: {b: rng.normal(0, 1, (n, 32)) + y[:, None] * 0.8 for b in ("dino_s",)} for v in VARIANTS}
    meta = rng.normal(0, 1, (n, 10)) + y[:, None] * 0.5
    res = run_experiments(feat, y, split, domains, ["dino_s"], meta)
    print(markdown(res))
    assert res["dino_s|norm->norm"]["test_norm"]["auc"] > 0.8
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/workspace/bench"))
    ap.add_argument("--cross-per-label", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="cap receipts (smoke test)")
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = dict(
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        torch=torch.__version__,
        python=platform.python_version(),
        cpus=os.cpu_count(),
        repo=REPO,
        seed=SEED,
    )
    log("env", info)
    args.work.mkdir(parents=True, exist_ok=True)
    items = build_items(args.work, args.cross_per_label, args.limit)
    y = np.array([it["label"] for it in items])
    domains = np.array([it["domain"] for it in items])
    split = split_receipts(items)
    stats = file_stats(items)
    data_desc = describe(items, stats, split)
    log("data", json.dumps(data_desc))

    backbones = {n: Backbone(k, mid, device) for n, (k, mid) in BACKBONES.items()}
    feat = {v: {n: np.zeros((len(items), b.dim), np.float32) for n, b in backbones.items()} for v in VARIANTS}
    rec_idx = [i for i, it in enumerate(items) if it["domain"] == RECEIPTS]
    train_idx = [i for i in rec_idx if split[i] == "train"]
    cross_idx = [i for i, it in enumerate(items) if it["domain"] != RECEIPTS]
    jobs = [("raw", rec_idx), ("norm", rec_idx + cross_idx), ("pp", rec_idx), ("aug1", train_idx), ("aug2", train_idx)]
    for variant, idx in jobs:
        sub = [items[i] for i in idx]
        got = extract_all(sub, variant, backbones, device, args.workers, "feat")
        for n in backbones:
            feat[variant][n][idx] = got[n]

    meta = meta_features(stats)
    results = run_experiments(feat, y, split, domains, list(BACKBONES), meta)
    md = markdown(results)
    log("RESULTS TABLE\n" + md)
    payload = dict(info=info, data=data_desc, results=results)
    (args.work / "results.json").write_text(json.dumps(payload))
    print("==RESULTS_JSON_BEGIN==")
    print(json.dumps(payload, separators=(",", ":")))
    print("==RESULTS_JSON_END==", flush=True)


if __name__ == "__main__":
    main()
