"""Score folders of images with a head (models/<version>) and report detection rates per group. CPU, local.

  python scripts/score_sources.py --group gpt4o:1:'artifacts/gated/gpt4o_receipt/*.png' \
      --group aiforge-edited:1:'artifacts/gated/aiforge_v1/TestingSet/images/*.png' \
      --group aiforge-authentic:0:'artifacts/gated/aiforge_v1/TestingSet/authentic/*.png'

--group NAME:LABEL:GLOB where LABEL is 1 (expected AI) or 0 (expected real). Features are cached, so reruns are fast.
Reports flagged-as-AI rate at the head's 1% and 5% thresholds, the median logit, and AUC of each AI group against each real group.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import train_local as t  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--model", type=Path, default=Path("models/v0.1.0"))
    ap.add_argument("--cache", type=Path, default=Path("artifacts/public/features.npz"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    items = []
    for g in a.group:
        name, label, pattern = g.split(":", 2)
        files = sorted(glob.glob(pattern))
        items += [dict(path=f, label=int(label), source=name) for f in files]
    head = json.loads((a.model / "head.json").read_text())
    embed = t.Embedder(a.model / "backbone.onnx")
    cache = t.featurize(items, embed, a.cache, aug_copies=0)
    mean, scale, coef, b = (np.array(head[k]) for k in ("mean", "scale", "coef")) + (head["intercept"],)
    res = {}
    for name in dict.fromkeys(it["source"] for it in items):
        rows = [cache.get(f"{it['key']}|norm") for it in items if it["source"] == name]
        rows = [r for r in rows if r is not None]
        res[name] = np.array([((r - mean) / scale) @ coef + b for r in rows])
    t1, t5 = head["threshold_fpr1"], head["threshold_fpr5"]
    label_of = {it["source"]: it["label"] for it in items}
    print(f"model {head['model_version']}  thresholds: 1%={t1:.2f} 5%={t5:.2f}")
    for name, s in res.items():
        print(f"  {name} (expected {'AI' if label_of[name] else 'real'}): n={len(s)} flagged@1%={np.mean(s > t1):.0%} flagged@5%={np.mean(s > t5):.0%} median logit={np.median(s):+.2f}")
    for fake in [n for n in res if label_of[n] == 1]:
        for real in [n for n in res if label_of[n] == 0]:
            print(f"  AUC {fake} vs {real}: {t.auc(np.r_[res[fake], res[real]], np.r_[np.ones(len(res[fake])), np.zeros(len(res[real]))]):.3f}")
    if a.out:
        a.out.write_text(json.dumps({k: v.tolist() for k, v in res.items()}))


if __name__ == "__main__":
    main()
