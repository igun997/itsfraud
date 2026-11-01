# Model card: itsfraud receipt detector v0.1.0

Detects whether a receipt or invoice image (or PDF) is AI generated. Output: a score (logit and probability) and a yes/no verdict.
This is a research-grade baseline, not a proven production detector. Read "Limits" before relying on it.

## What it is

- **Backbone**: DINOv2-base (`facebook/dinov2-base`), frozen, exported to ONNX (`backbone.onnx`).
- **Head**: logistic regression on the mean-pooled CLS embedding of up to 8 native-resolution 224 px tiles per page (first 5 pages for PDFs).
  The file score is the mean tile logit, so the head is linear in the pooled feature.
- **Preprocessing (must match training)**: resize so the long side is 1024 px (Lanczos), re-encode as JPEG quality 75, then tile.
  The head declares this in `head.json` (`preprocess`), and the Rust service applies it automatically.
- **Provenance**: a file that carries the IPTC `trainedAlgorithmicMedia` tag is flagged as AI generated regardless of the model score. Generator names alone are informational.
- **Decision threshold**: logit -0.882 (about 1% false positives on the validation split); a looser threshold at -1.931 (about 5%) is also stored.

## Training data

| source | label | files | train / val / test |
|---|---|---|---|
| SROIE scans (from [Shuyiww/TextRich](https://huggingface.co/datasets/Shuyiww/TextRich) `Receipts/0_real`) | real | 986 | 690 / 148 / 148 |
| CORD, Indonesian receipt photos ([naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2), CC-BY-4.0) | real | 1,000 | 700 / 150 / 150 |
| GPT-Image-2 receipts (TextRich `Receipts/1_fake`) | fake | 1,095 | 766 / 164 / 165 |

Split 70/15/15 per source, seed 0. Training also uses two augmented copies of each training image (downscale and JPEG recompression).
The TextRich dataset page states no licence; check it before redistributing anything derived from it.

## Results (test split, 463 files: 148 SROIE, 150 CORD, 165 GPT-Image-2)

| condition | AUC (95% CI) | at 1% threshold: TPR / FPR | at 5% threshold: TPR / FPR |
|---|---|---|---|
| normalized (production pipeline) | 1.000 (1.000-1.000) | 1.00 / 0.007 | 1.00 / 0.057 |
| extra compression (0.6x downscale, JPEG 50) | 0.9999 (0.9997-1.000) | 0.994 / 0.010 | 1.00 / 0.020 |

False positives by real source at the 1% threshold: CORD 0.0%, SROIE 1.4%. Fake detection rate on GPT-Image-2: 100% (165 of 165).

### Comparison with public detectors (same 463 test files, zero-shot, normalized condition)

| detector | AUC (95% CI) | TPR / FPR at its default 0.5 | TPR / FPR at 5% validation threshold |
|---|---|---|---|
| **itsfraud v0.1.0** | **1.000** | 1.00 / 0.007 (at 1% threshold) | 1.00 / 0.057 |
| Ateeqq/ai-vs-human-image-detector | 0.924 (0.898-0.948) | 0.99 / 0.52 | 0.57 / 0.06 |
| umm-maybe/AI-image-detector | 0.605 (0.548-0.652) | 0.00 / 0.00 | 0.00 / 0.04 |
| Smogy/SMOGY-Ai-images-detector | 0.537 (0.481-0.588) | 0.00 / 0.00 | 0.01 / 0.04 |
| dima806/ai_vs_human_generated_image_detection | 0.501 (0.446-0.549) | 0.00 / 0.01 | 0.00 / 0.03 |
| Organika/sdxl-detector | 0.290 (0.240-0.334) | 0.01 / 0.26 | 0.00 / 0.05 |
| haywoodsloan/ai-image-detector-deploy | 0.220 (0.177-0.266) | 0.00 / 0.00 | 0.00 / 0.02 |

Full table, other conditions and caveats: [BENCHMARK.md](BENCHMARK.md).

## Limits (read these)

1. **One generator.** Every fake is from GPT-Image-2. The model has never seen output from Gemini, Flux, Stable Diffusion, Midjourney or any editing tool, and detectors of this kind usually degrade on unseen generators. Untested.
2. **Fake-vs-real source shortcut.** Real receipts are scans and phone photos; fakes are clean renders in one style. The model may partly detect "scan vs render". Real and fake files were put through one pipeline (normalization), which removes the file-format shortcut but not the style gap.
3. **New real sources raise false alarms.** Trained without CORD, the same model flagged **20.2%** of CORD receipts at its 1% threshold and 60.6% at its 5% threshold (ranking stayed good: AUC 0.999 against the fakes). Thresholds do not transfer to a new kind of real receipt. Before deployment, calibrate the threshold on a few hundred of your own genuine receipts, or retrain with them.
4. **The comparison is not like-for-like.** Our head was trained on this distribution; the public detectors were not. It shows that off-the-shelf detectors do not work on text-heavy receipts, not that ours would beat a detector trained on the same data.
5. **AI edits are not tested.** Inpainted numbers or dates on a real receipt (the more common fraud) are a different task; earlier published results show generic detectors fail on them.
6. **Single run, one split**, no seed-to-seed variance. A test AUC of 1.000 with this sample size means "no errors observed", not a guarantee.
7. **Metadata is a separate, fragile signal.** All 5 GPT-Image-2 files I checked carry OpenAI's C2PA/IPTC `trainedAlgorithmicMedia` marker and none of the 10 real receipts do, so the provenance check alone flags unmodified OpenAI output in the raw files.
   Any re-encode, screenshot or messaging-app upload strips it. The model metrics above are measured after normalization (JPEG re-encode), which removes the metadata, so they reflect the model, not the tag.
   In production the tag catches unmodified files from generators that sign their output (5 of 5 sampled here), and the model is the fallback when it is missing. Only 5 files were checked; the share across all 1,095 fakes is unknown.
8. Rust and Python preprocessing differ slightly (resize and JPEG encoder implementations). On 15 sample receipts the logit differed by 0.12 on average and 0.34 at most, with identical verdicts; the closest real sample sat about 1 logit from the threshold.

## Recommended use

Score as a risk signal and route the middle band to human review rather than auto-rejecting. Log every score with the reviewer's decision so the model can be retrained on your documents. Re-evaluate whenever a new image generator becomes common.

## Artifacts

`models/v0.1.0/`: `head.json` (sha256 prefix `1e7026a134d171f4`), `metrics.json`, and `backbone.onnx` (330 MB, sha256 prefix `ad0439cee9cc55d7`, not stored in git).
Rebuild the ONNX file with `python train/export_onnx.py --model-id facebook/dinov2-base --out models/v0.1.0/backbone.onnx` (needs torch, transformers, onnx);
the maximum difference from the PyTorch model was 1.6e-5. Training and evaluation code: `bench/train_final.py`, `bench/compare_detectors.py`.

```bash
cargo build --release
ITSFRAUD_ARTIFACTS=models/v0.1.0 ./target/release/itsfraud scan receipt.jpg
docker build -t itsfraud . && docker run --rm -p 8080:8080 -v "$PWD/models/v0.1.0:/models:ro" itsfraud
```
