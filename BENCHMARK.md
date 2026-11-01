# Benchmark: frozen-feature detectors for AI-generated receipts

Run on 2026-11-01 on one rented NVIDIA A100-SXM4-80GB (Runpod secure cloud, US-MD-1), torch 2.8.0+cu128, about 17 minutes of wall time.
Code: [bench/run_benchmark.py](bench/run_benchmark.py); the pod ran the single-file build from [bench/flatten.py](bench/flatten.py)
(sha256 prefix `c3bdea156a8a0ebb`, verified on the pod before it ran). Single run, single split (seed 0): no seed-to-seed variance is measured.

## Data

[Shuyiww/TextRich](https://huggingface.co/datasets/Shuyiww/TextRich) (public, ungated; licence not stated on the dataset page, so check it before reuse).
Receipts: 986 real (SROIE scans) and 1,095 fake (all GPT-Image-2). Split by file: train 1,456 / val 312 / test 313. Cross-domain test sets
(300 images each, 150 real + 150 fake): Tables, UIScreenshots, InfographicCharts.

| | real receipts | fake receipts |
|---|---|---|
| count | 986 | 1,095 |
| file format | JPEG (all) | PNG (all) |
| median size | 818 x 1697 px, 346 KB | 1023 x 1537 px, 1,937 KB |

Real and fake differ completely in file format and size, so a file-metadata-only model (width, height, bytes, format) scores AUC 1.000 on the raw files.
Every `raw` number below is therefore meaningless as evidence of AI detection. The `norm` condition resizes every image to long side 1024 and
re-encodes it as JPEG quality 75, so both classes share one pipeline. `pp` adds a 0.6x downscale and JPEG quality 50.

## Setup

Per page: up to 8 native-resolution 224 px tiles (`train/common.py`), frozen backbone embeddings mean-pooled per file, logistic regression head
(C chosen on validation AUC), file score = head logit. Thresholds are set on the validation split for 1% and 5% false positives on real receipts,
then applied unchanged to every test set. Backbones: DINOv2-small, DINOv2-base, CLIP ViT-B/16. "norm+aug" adds two augmented training copies
(downscale and JPEG recompression) of every training receipt.

## Results (test AUC; TPR / FPR at the 5%-FPR validation threshold)

| model | trained on | test_raw | test_norm | test_pp | x:Tables | x:UIScreenshots | x:InfographicCharts |
|---|---|---|---|---|---|---|---|
| dino_s | raw | 1.000 (1.00 / 0.07) | 0.998 (0.97 / 0.04) | | | | |
| dino_s | norm | | 1.000 (1.00 / 0.03) | 0.999 (1.00 / 0.02) | 0.816 (0.95 / 0.43) | 0.465 (0.27 / 0.31) | 0.428 (0.37 / 0.55) |
| dino_s | norm+aug | | 1.000 (1.00 / 0.05) | 0.999 (1.00 / 0.03) | 0.858 (0.99 / 0.59) | 0.494 (0.25 / 0.22) | 0.425 (0.35 / 0.49) |
| dino_b | raw | 1.000 (1.00 / 0.08) | 1.000 (1.00 / 0.02) | | | | |
| dino_b | norm | | 1.000 (1.00 / 0.05) | 0.999 (0.99 / 0.02) | 0.915 (1.00 / 0.79) | 0.677 (0.77 / 0.60) | 0.763 (0.99 / 0.78) |
| dino_b | norm+aug | | 1.000 (1.00 / 0.05) | 1.000 (1.00 / 0.03) | 0.951 (1.00 / 0.94) | 0.673 (0.63 / 0.47) | 0.763 (0.88 / 0.51) |
| clip_b16 | raw | 1.000 (1.00 / 0.05) | 1.000 (1.00 / 0.03) | | | | |
| clip_b16 | norm | | 1.000 (1.00 / 0.07) | 1.000 (1.00 / 0.02) | 0.876 (1.00 / 1.00) | 0.424 (1.00 / 1.00) | 0.311 (1.00 / 1.00) |
| clip_b16 | norm+aug | | 1.000 (1.00 / 0.08) | 1.000 (1.00 / 0.03) | 0.882 (1.00 / 0.99) | 0.643 (1.00 / 1.00) | 0.429 (1.00 / 1.00) |
| metadata only (HGB) | raw | 1.000 (1.00 / 0.00) | | | | | |

95% bootstrap confidence intervals for every cell are in the run's JSON output (`results.json` on the pod, lost with the pod); for test_norm the intervals are
[1.0, 1.0] for dino_s and dino_b trained on norm, and the lower bound is above 0.995 everywhere on the receipt test sets.

## What this does and does not show

- In-distribution, all three frozen backbones separate real SROIE scans from GPT-Image-2 receipts almost perfectly (AUC about 1.000), including after
  pipeline normalization and after extra compression (`pp`, AUC 0.999 to 1.000). Training with augmentation changes little in-distribution.
- That is not evidence that the models detect AI generation. Even after normalization, real receipts are phone or scanner captures of paper, and the fakes are clean renders in
  one generator's style, so "scan vs clean render" is a plausible shortcut. The cross-domain results point the same way: on tables, UI screenshots and charts
  AUC ranges from 0.31 to 0.95, below 0.5 for several (CLIP on charts and UI), and thresholds fixed on receipts give false-positive rates of 0.2 to 1.0.
  Those cross-domain sets also change the source of the real images, so they mix "AI vs real" with "source shift"; they show lack of transfer, not a precise measure.
- DINOv2-base transferred best (Tables 0.92 to 0.95, charts 0.76, UI 0.67); CLIP scores shifted so far that its receipt-derived threshold flags almost everything.
- With a 5% target the test false-positive rate landed between 2% and 8%; the validation set has only about 148 real receipts, so a 1% threshold is coarse.
- Only one generator is present, so unseen-generator generalization is untested. Not run: LFM2-VL LoRA, PDF inputs, the Rust runtime.

## Next steps that would make the benchmark meaningful

1. Add fakes from at least two more generators and hold one out (for example Scam-AI/gpt4o-receipt, gated: it needs you to accept its terms).
2. Add real receipts from a second source, preferably phone photos, and match real and fake on layout and merchant style.
3. Repeat over several seeds and report the spread.

## Run cost

Two A100 pods were used at $1.79/hr: a first pod that I terminated after about a minute and a half because the pasted code bundle was corrupted, and the pod that produced these
results (about 17 minutes). The estimate is from pod create and terminate times, not a billing read. Both pods are terminated; the account has no pods.
