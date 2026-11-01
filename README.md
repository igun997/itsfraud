# itsfraud

Send an image or PDF (receipt/invoice) to a fine-tuned model and get back whether it is AI generated or not.
Background and design notes: [RESEARCH.md](RESEARCH.md).

How it works: provenance scan (a file that declares `trainedAlgorithmicMedia` is flagged outright) plus up to 8 native-resolution
224px tiles per page (first 5 pages for PDFs) embedded by DINOv2-small and scored by a linear head. File score = mean tile logit;
the threshold is set for 1% false positives on real files using out-of-fold scores.

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
PDF input is behind the `detector` crate's `pdf` feature and needs the pdfium shared library on the library path; without it PDFs
return HTTP 422. The server crate does not forward that feature yet, so enable it by adding `features = ["pdf"]` to the `detector`
dependency in `crates/server/Cargo.toml`. It compiles but has not been run.

## Model, benchmark and comparison

[MODEL_CARD.md](MODEL_CARD.md) describes the v0.1.0 production head, its limits and how to run it. [BENCHMARK.md](BENCHMARK.md) has the GPU benchmark on public receipt data and the comparison with public detectors (`bench/*.py`, shipped to a pod via `bench/flatten.py`).
Short version: near-perfect on the test split and far better than public detectors on receipts, but only one generator is covered and thresholds do not transfer to new real-receipt sources.

## Status

- Verified: Rust unit tests; real DINOv2-base head trained and evaluated on GPU (see BENCHMARK.md and MODEL_CARD.md); ONNX export matches PyTorch to 1.6e-5; Rust and Python scores agree on 15 sample receipts (verdicts identical, logit difference 0.12 on average); HTTP endpoints, provenance override, bad-input 422.
- Not verified: PDF rendering (compiles, never run), unseen generators, AI-edited receipts, the Docker image at runtime, any non-receipt document.
- Preprocessing is duplicated in `train/common.py` (PIL) and `crates/detector/src/tiles.rs` (Rust); keep them equivalent. The Rust resize and JPEG encoder are not bit-identical to PIL/libjpeg.
- Do not treat v0.1.0 as proven: it was trained on one generator and its threshold does not transfer to new real-receipt sources (MODEL_CARD.md, "Limits").
