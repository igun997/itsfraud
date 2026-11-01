"""Upload the release to a Hugging Face model repo (needs a WRITE token: HF_WRITE_TOKEN in env, or run `hf auth login`).

  HF_WRITE_TOKEN=... python scripts/push_hf.py --repo igun997/itsfraud-receipt-detector [--private]

Uploads MODEL_CARD.md as README.md (licence cc-by-nc-sa-4.0 because v0.2/v0.3 heads use CC-BY-NC-SA data),
the head.json/metrics.json of each version, and v0.1.0/backbone.onnx (DINOv2-base, Apache-2.0).
"""
import argparse
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FRONT = "---\nlicense: cc-by-nc-sa-4.0\nlibrary_name: onnx\ntags: [ai-generated-image-detection, receipts, invoices, dinov2]\n---\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_WRITE_TOKEN") or None)
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        card = Path(d) / "README.md"
        card.write_text(FRONT + (ROOT / "MODEL_CARD.md").read_text())
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=a.repo)
    for v in sorted(p for p in (ROOT / "models").iterdir() if p.is_dir()):
        for f in ("head.json", "metrics.json"):
            api.upload_file(path_or_fileobj=str(v / f), path_in_repo=f"{v.name}/{f}", repo_id=a.repo)
    api.upload_file(path_or_fileobj=str((ROOT / "models/v0.1.0/backbone.onnx").resolve()), path_in_repo="backbone.onnx", repo_id=a.repo)
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
