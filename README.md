# itsfraud

Send an image or PDF (receipt/invoice) to a fine-tuned model and get back whether it is AI generated or not.
Background and design notes: [RESEARCH.md](RESEARCH.md).

How it works: provenance scan (a file that declares `trainedAlgorithmicMedia` is flagged outright) plus up to 8 native-resolution
224px tiles per page (first 5 pages for PDFs) embedded by DINOv2-small and scored by a linear head. File score = mean tile logit;
the threshold is set for 1% false positives on real files using out-of-fold scores.

## Quick start

1. Get the model from [Hugging Face](https://huggingface.co/igun997/itsfraud-receipt-detector) (the licence is CC-BY-NC-SA-4.0, non-commercial; see MODEL_CARD.md):

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("igun997/itsfraud-receipt-detector", local_dir="artifacts")
EOF
cp artifacts/v0.3.0-podfit/head.json artifacts/head.json    # pick the head version to serve; backbone.onnx is shared
```

2. Build and score a file (`--features pdf` and `libpdfium.so` on `LD_LIBRARY_PATH` are only needed for PDFs):

```bash
cargo build --release --features pdf
./target/release/itsfraud scan receipt.jpg
```

```json
{
  "ai_generated": true,
  "probability": 0.99,
  "logit": 4.62,
  "threshold": -0.64,
  "threshold_fpr5": -1.66,
  "model_version": "0.3.0-podfit",
  "pages": 1,
  "tiles": 8,
  "signals": [],
  "decided_by": "model"
}
```

3. Or run it as a service and call it over HTTP:

```bash
./target/release/itsfraud serve 127.0.0.1:8080
curl localhost:8080/health                       # ok
curl -F file=@receipt.pdf localhost:8080/detect  # same JSON as above
```

```python
import requests

with open("receipt.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8080/detect", files={"file": f}, timeout=60)
r.raise_for_status()                       # unsupported or corrupt files return HTTP 422 with {"error": ...}
res = r.json()
print("AI generated" if res["ai_generated"] else "looks real", f"(p={res['probability']:.2f}, by {res['decided_by']})")
```

Reading the result: `ai_generated` is true when the logit is above `threshold` (set for 1% false positives on real validation files); `threshold_fpr5` is a more sensitive cut-off (5%) that you can apply to `logit` yourself, at the cost of more false alarms.
`decided_by` is `provenance` when the file declares an AI generator in its metadata (`signals` lists it) and `model` otherwise. A real file scoring just below the threshold is not proof of authenticity.
This detector is not proven on generators outside its training data (for example photo-style Gemini output was missed) or on AI-edited receipts: treat a "real" verdict as "no AI signal found", not as verification. Details in MODEL_CARD.md, "Limits".

## Train (Python, in `train/`)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r train/requirements.txt
cd train
python extract_features.py      # reads ../data (see data/README.md), writes ../artifacts/features.npz
python train_head.py            # leave-one-generator-out report, writes ../artifacts/head.json
python export_onnx.py           # writes ../artifacts/backbone.onnx
python predict.py some.pdf      # reference score to compare with the Rust output
```

Feature extraction is the heavy step; use a GPU (for example a rented Runpod pod) for large datasets. The head trains on CPU in seconds.

## Run (Rust)

```bash
cargo build --release
./target/release/itsfraud scan path/to/file.png
./target/release/itsfraud serve 127.0.0.1:8080
curl -F file=@receipt.pdf localhost:8080/detect
```

`ITSFRAUD_ARTIFACTS` points at the folder with `backbone.onnx` and `head.json` (default `./artifacts`).
PDF input needs the pdfium shared library at runtime and the `pdf` feature: `cargo build --release --features pdf`, with `libpdfium.so` on `LD_LIBRARY_PATH`.
The crate is pinned to pdfium API level 7881, which works with pdfium builds 7881 and newer; the library bundled in the `pypdfium2` Python package (build 8076) works.
Without the feature or library, PDFs return HTTP 422. The Docker image does not include pdfium, so it serves images only.
`python scripts/check_pdfs.py --pdfium-dir <dir with libpdfium.so>` scores everything under `tests/pdfs` and `data/` and summarizes by folder (`fake/` expected AI, `real/` expected genuine).

## Model, benchmark and comparison

[MODEL_CARD.md](MODEL_CARD.md) describes the v0.1.0 production head, its limits and how to run it. [BENCHMARK.md](BENCHMARK.md) has the GPU benchmark on public receipt data and the comparison with public detectors (`bench/*.py`, shipped to a pod via `bench/flatten.py`).
Short version: near-perfect on the test split and far better than public detectors on receipts, but only one generator is covered and thresholds do not transfer to new real-receipt sources.

## Status

- Verified: Rust unit tests; real DINOv2-base head trained and evaluated on GPU (see BENCHMARK.md and MODEL_CARD.md); ONNX export matches PyTorch to 1.6e-5; Rust and Python scores agree on 15 sample receipts (verdicts identical, logit difference 0.12 on average); HTTP endpoints, provenance override, bad-input 422.
- PDF rendering works (13 held-out GPT-Image-2 fake PDFs all caught, 7 public real PDFs and your real files mostly passed: one A4-layout public real PDF was a false alarm). Not verified: unseen generators beyond a single missed spot check, AI-edited receipts, any non-receipt document (the Docker image was run and answers image requests; it has no PDF support).
- Preprocessing is duplicated in `train/common.py` (PIL) and `crates/detector/src/tiles.rs` (Rust); keep them equivalent. The Rust resize and JPEG encoder are not bit-identical to PIL/libjpeg.
- Do not treat v0.1.0 as proven: it was trained on one generator and its threshold does not transfer to new real-receipt sources (MODEL_CARD.md, "Limits").
