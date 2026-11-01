"""Compare public, off-the-shelf AI-image detectors on the same receipt split used to train our head.

Rebuilds the exact items and split of bench/train_final.py (TextRich receipts + CORD, seed 0), so the test split is the same
313+150-style held-out set our head is evaluated on. Each public detector is run zero-shot (its own preprocessing, whole image).
Thresholds for the "val-calibrated" columns are set on the validation split at 1% / 5% false positives on real receipts,
which is the same protocol as our head and is generous to the baselines: they get to see our validation data's score range.

  python bench/compare_detectors.py --work /workspace/compare
  python bench/compare_detectors.py --selftest
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from run_benchmark import RECEIPTS, VARIANTS, build_items, log, metrics, thr_at_fpr
from train_final import emit, load_cord, split_items

DETECTORS = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
    "Ateeqq/ai-vs-human-image-detector",
    "haywoodsloan/ai-image-detector-deploy",
    "dima806/ai_vs_human_generated_image_detection",
    "Smogy/SMOGY-Ai-images-detector",
]
CONDITIONS = ("norm", "raw", "pp")
AI_TOKENS = {"ai", "artificial", "fake", "synthetic", "generated", "sdxl", "aigc"}
HUMAN_TOKENS = {"human", "hum", "real", "natural", "authentic"}


def resolve_ai_labels(id2label):
    """Indices of labels that mean 'AI generated'; None if the label names do not say."""
    ai = []
    for i, label in id2label.items():
        tokens = set(re.split(r"[^a-z0-9]+", str(label).lower()))
        if tokens & AI_TOKENS and not tokens & HUMAN_TOKENS:
            ai.append(int(i))
    return ai if 0 < len(ai) < len(id2label) else None


def score_detector(name, images_by_cond, device, bs=32):
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    labels = dict(model.config.id2label)
    log(f"loaded {name} ({time.time() - t0:.0f}s) labels={labels}")
    ai = resolve_ai_labels(labels)
    if ai is None:
        return None, labels
    out = {}
    with torch.no_grad():
        for cond, imgs in images_by_cond.items():
            chunks = []
            for i in range(0, len(imgs), bs):
                x = proc(images=imgs[i : i + bs], return_tensors="pt").to(device)
                chunks.append(model(**x).logits.softmax(-1)[:, ai].sum(-1).float().cpu().numpy())
                if (i // bs) % 10 == 0:
                    log(f"  {name}|{cond} {i}/{len(imgs)} ({time.time() - t0:.0f}s)")
            out[cond] = np.concatenate(chunks)
    del model
    return out, labels


def evaluate_scores(s, y, src, sp, rng):
    va, te = sp == "val", sp == "test"
    thr1, thr5 = thr_at_fpr(s[va], y[va], 0.01), thr_at_fpr(s[va], y[va], 0.05)
    r = metrics(s[te], y[te], thr1, thr5, rng)
    r["tpr@0.5"] = round(float((s[te & (y == 1)] > 0.5).mean()), 4)
    r["fpr@0.5"] = round(float((s[te & (y == 0)] > 0.5).mean()), 4)
    r["fpr@0.5_by_source"] = {k: round(float((s[te & (src == k)] > 0.5).mean()), 4) for k in ("sroie", "cord") if (te & (src == k)).any()}
    r["median_score_real"] = round(float(np.median(s[te & (y == 0)])), 4)
    r["median_score_fake"] = round(float(np.median(s[te & (y == 1)])), 4)
    return r


def markdown(results):
    lines = ["| detector | condition | AUC (95% CI) | TPR / FPR @0.5 | TPR / FPR @5% val thr | FPR@0.5 sroie / cord |", "|---|---|---|---|---|---|"]
    for k, r in results.items():
        name, cond = k.split("|")
        if "auc" not in r:
            lines.append(f"| {name} | {cond} | {r.get('error') or 'labels unresolved'} | | | |")
            continue
        fs = r["fpr@0.5_by_source"]
        lines.append(
            f"| {name} | {cond} | {r['auc']:.3f} ({r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f}) | {r['tpr@0.5']:.2f} / {r['fpr@0.5']:.2f} "
            f"| {r['tpr@fpr5thr']:.2f} / {r['fpr@fpr5thr']:.2f} | {fs.get('sroie', float('nan')):.2f} / {fs.get('cord', float('nan')):.2f} |"
        )
    return "\n".join(lines)


def selftest():
    assert resolve_ai_labels({0: "artificial", 1: "human"}) == [0]
    assert resolve_ai_labels({0: "ai", 1: "hum"}) == [0]
    assert resolve_ai_labels({0: "human", 1: "AI-generated"}) == [1]
    assert resolve_ai_labels({0: "artificial", 1: "real"}) == [0]
    assert resolve_ai_labels({0: "LABEL_0", 1: "LABEL_1"}) is None
    rng = np.random.default_rng(0)
    n = 400
    y = np.array([0, 1] * (n // 2))
    src = np.where(y == 1, "gpt-image-2", np.where(np.arange(n) % 4 == 0, "cord", "sroie"))
    sp = np.array(["train", "train", "val", "val", "test", "test", "test", "test"] * (n // 8))
    s = np.clip(rng.normal(0.3, 0.2, n) + y * 0.4, 0, 1)
    r = evaluate_scores(s, y, src, sp, rng)
    assert 0.8 < r["auc"] <= 1.0 and set(r["fpr@0.5_by_source"]) == {"sroie", "cord"}
    print(markdown({"demo|raw": r}))
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/workspace/compare"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    import torch

    torch.set_num_threads(8)  # the container sees every host CPU but is limited to a few; avoid oversubscription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work.mkdir(parents=True, exist_ok=True)
    items = build_items(args.work, 0, None)
    for it in items:
        it["source"] = "sroie" if it["label"] == 0 else "gpt-image-2"
    for p in load_cord(args.work):
        items.append(dict(path=p, label=0, domain=RECEIPTS, source="cord"))
    for i, it in enumerate(items):
        it["id"] = i
    split = split_items(items)
    ev = [it for it in items if split[it["id"]] in ("val", "test")]
    y = np.array([it["label"] for it in ev])
    src = np.array([it["source"] for it in ev])
    sp = np.array([split[it["id"]] for it in ev])
    log(f"eval items: {len(ev)} (val {int((sp == 'val').sum())}, test {int((sp == 'test').sum())})")

    def load(it, cond):
        return VARIANTS[cond](Image.open(it["path"]).convert("RGB"), it["id"])

    images = {}
    with ThreadPoolExecutor(16) as ex:
        for cond in CONDITIONS:
            images[cond] = list(ex.map(lambda it: load(it, cond), ev))
            log(f"built {cond} images")

    rng = np.random.default_rng(1)
    results = {}
    for det in DETECTORS:
        try:
            scores, labels = score_detector(det, images, device)
        except Exception as e:
            log("failed", det, repr(e)[:200])
            for cond in CONDITIONS:
                results[f"{det}|{cond}"] = dict(error=repr(e)[:200])
            continue
        for cond in CONDITIONS:
            if scores is None:
                results[f"{det}|{cond}"] = dict(error=f"labels unresolved: {labels}")
            else:
                results[f"{det}|{cond}"] = dict(labels={str(k): v for k, v in labels.items()}, **evaluate_scores(scores[cond], y, src, sp, rng))
        log("done", det, labels)

    log("RESULTS\n" + markdown(results))
    emit("COMPARE_JSON", json.dumps(dict(n_eval=len(ev), results=results), separators=(",", ":")))


if __name__ == "__main__":
    main()
