"""Train the production head: DINOv2-base tile features on normalized images + logistic regression.

Real receipts: SROIE scans (TextRich/Receipts/0_real) and CORD photos (naver-clova-ix/cord-v2).
Fake receipts: GPT-Image-2 (TextRich/Receipts/1_fake). Production preprocessing is the `norm` variant of
bench/run_benchmark.py: long side 1024, JPEG quality 75, then 224px native tiles.

Stage A (diagnostic): train WITHOUT CORD, then measure false alarms on CORD, a real-receipt source the model never saw.
Stage B (final): train with CORD, thresholds from the validation split, one pass over the test split.

  python bench/train_final.py --work /workspace/final            # on a GPU box
  python bench/train_final.py --selftest                         # CPU, synthetic features

Prints the head as ==HEAD_JSON_BEGIN== ... ==HEAD_JSON_END== (with a sha256) and metrics as ==METRICS_JSON_BEGIN== ... END.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_benchmark import RECEIPTS, Backbone, build_items, extract_all, log, metrics, thr_at_fpr

CORD_REPO = "naver-clova-ix/cord-v2"
MODEL_ID = "facebook/dinov2-base"
VERSION = "0.1.0"
PREPROCESS = "long side 1024 (Lanczos), JPEG q75, up to 8 native 224px tiles per page, ImageNet mean/std"


def load_cord(work: Path):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(CORD_REPO, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))
    out_dir = work / "cord"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        local = hf_hub_download(CORD_REPO, f, repo_type="dataset", local_dir=str(work / "cord_hf"))
        pf = pq.ParquetFile(local)
        for rg in range(pf.num_row_groups):
            for rec in pf.read_row_group(rg, columns=["image"]).column("image").to_pylist():
                p = out_dir / f"c{len(paths):05d}.img"
                p.write_bytes(rec["bytes"])
                paths.append(str(p))
        log(f"cord: {len(paths)} images after {f}")
    return paths


def split_items(items):
    """70/15/15 per (label, source) so every group appears in every split."""
    rng = np.random.default_rng(0)
    groups, split = {}, {}
    for it in items:
        groups.setdefault((it["label"], it["source"]), []).append(it["id"])
    for ids in groups.values():
        ids = np.array(ids)
        rng.shuffle(ids)
        a, b = int(0.7 * len(ids)), int(0.85 * len(ids))
        for j, i in enumerate(ids):
            split[int(i)] = "train" if j < a else "val" if j < b else "test"
    return split


def fit_lr(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1):
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        auc = roc_auc_score(yva, clf.decision_function(sc.transform(Xva)))
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    return sc, best[2], best[1]


def rates(s, y, thr):
    return dict(
        tpr=float((s[y == 1] > thr).mean()) if (y == 1).any() else None,
        fpr=float((s[y == 0] > thr).mean()) if (y == 0).any() else None,
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
    )


def run_stages(items, split, F):
    """F: dict variant -> (n_items, dim) features (norm for all; pp for val/test; aug1/aug2 for train)."""
    rng = np.random.default_rng(1)
    y = np.array([it["label"] for it in items])
    src = np.array([it["source"] for it in items])
    sp = np.array([split[it["id"]] for it in items])
    out = {}

    # Stage A: CORD never seen
    nc = src != "cord"
    tr, va, te = (nc & (sp == s) for s in ("train", "val", "test"))
    sc, clf, C = fit_lr(F["norm"][tr], y[tr], F["norm"][va], y[va])
    f = lambda X: clf.decision_function(sc.transform(X))
    sv = f(F["norm"][va])
    t1, t5 = thr_at_fpr(sv, y[va], 0.01), thr_at_fpr(sv, y[va], 0.05)
    cord = src == "cord"
    s_cord, s_fake = f(F["norm"][cord]), f(F["norm"][te & (y == 1)])
    out["A_without_cord"] = dict(
        C=C, thr_fpr1=t1, thr_fpr5=t5,
        test_norm=metrics(f(F["norm"][te]), y[te], t1, t5, rng),
        test_pp=metrics(f(F["pp"][te]), y[te], t1, t5, rng),
        cord_unseen=dict(
            n=int(cord.sum()),
            fpr_at_fpr1_thr=float((s_cord > t1).mean()),
            fpr_at_fpr5_thr=float((s_cord > t5).mean()),
            auc_vs_fake_test=float(roc_auc_score(np.r_[np.zeros(len(s_cord)), np.ones(len(s_fake))], np.r_[s_cord, s_fake])),
        ),
    )

    # Stage B: final model
    tr, va, te = (sp == s for s in ("train", "val", "test"))
    Xa = np.concatenate([F["norm"][tr], F["aug1"][tr], F["aug2"][tr]])
    ya = np.concatenate([y[tr]] * 3)
    sc, clf, C = fit_lr(Xa, ya, F["norm"][va], y[va])
    f = lambda X: clf.decision_function(sc.transform(X))
    sv = f(F["norm"][va])
    t1, t5 = thr_at_fpr(sv, y[va], 0.01), thr_at_fpr(sv, y[va], 0.05)
    s_te = f(F["norm"][te])
    per_source = {}
    for name in sorted(set(src)):
        m = te & (src == name)
        per_source[name] = {"thr_fpr1": rates(f(F["norm"][m]), y[m], t1), "thr_fpr5": rates(f(F["norm"][m]), y[m], t5)}
    out["B_final"] = dict(
        C=C, thr_fpr1=t1, thr_fpr5=t5,
        test_norm=metrics(s_te, y[te], t1, t5, rng),
        test_pp=metrics(f(F["pp"][te]), y[te], t1, t5, rng),
        per_source=per_source,
    )
    head = dict(
        model_version=VERSION,
        backbone=MODEL_ID,
        preprocess=PREPROCESS,
        mean=sc.mean_.tolist(),
        scale=sc.scale_.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
        threshold=t1,
        threshold_fpr1=t1,
        threshold_fpr5=t5,
    )
    return out, head


def emit(tag, text):
    print(f"==%s_BEGIN==" % tag)
    print(text)
    print(f"==%s_END==" % tag, flush=True)


def selftest():
    rng = np.random.default_rng(0)
    items = []
    for label, source, n in ((0, "sroie", 200), (0, "cord", 100), (1, "gpt-image-2", 250)):
        for _ in range(n):
            items.append(dict(id=len(items), label=label, source=source))
    split = split_items(items)
    y = np.array([it["label"] for it in items])
    F = {v: rng.normal(0, 1, (len(items), 16)) + y[:, None] * 1.5 for v in ("norm", "pp", "aug1", "aug2")}
    res, head = run_stages(items, split, F)
    assert res["B_final"]["test_norm"]["auc"] > 0.9
    assert len(head["coef"]) == 16 and head["threshold"] == head["threshold_fpr1"]
    print(json.dumps({k: v.get("cord_unseen") for k, v in res.items() if "cord_unseen" in v}))
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/workspace/final"))
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work.mkdir(parents=True, exist_ok=True)
    items = build_items(args.work, 0, None)  # TextRich receipts only
    for it in items:
        it["source"] = "sroie" if it["label"] == 0 else "gpt-image-2"
    for p in load_cord(args.work):
        items.append(dict(path=p, label=0, domain=RECEIPTS, source="cord"))
    for i, it in enumerate(items):
        it["id"] = i
    split = split_items(items)
    counts = {}
    for it in items:
        k = f"{it['source']}/{split[it['id']]}"
        counts[k] = counts.get(k, 0) + 1
    log("counts", json.dumps(counts, sort_keys=True))

    bb = {"dino_b": Backbone("dino", MODEL_ID, device)}
    n = len(items)
    F = {v: np.zeros((n, bb["dino_b"].dim), np.float32) for v in ("norm", "pp", "aug1", "aug2")}
    idx = lambda *s: [i for i in range(n) if split[i] in s]
    for variant, ids in (("norm", idx("train", "val", "test")), ("pp", idx("val", "test")), ("aug1", idx("train")), ("aug2", idx("train"))):
        got = extract_all([items[i] for i in ids], variant, bb, device, args.workers, "final")
        F[variant][ids] = got["dino_b"]

    results, head = run_stages(items, split, F)
    results["counts"] = counts
    results["model_version"] = VERSION
    emit("METRICS_JSON", json.dumps(results, separators=(",", ":")))
    head_text = json.dumps(head)
    log("head sha256", hashlib.sha256(head_text.encode()).hexdigest())
    emit("HEAD_JSON", head_text)


if __name__ == "__main__":
    main()
