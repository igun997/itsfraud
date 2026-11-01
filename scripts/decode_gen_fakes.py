"""Decode the GEN_FEATURES / GEN_SAMPLES blocks printed by bench/gen_fakes.py from a saved pod log.

  python scripts/decode_gen_fakes.py <log file> [--out artifacts/gen_fakes]
The log is either the raw text or the JSON saved by the Runpod log tool ({"items": [{"line": ...}]}); several saved files can be passed
(chunks are joined in order). Verifies the sha256 printed by the pod, then writes features.npz, meta.json and sample JPEGs.
"""
import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np


def lines_of(path: Path):
    text = path.read_text()
    try:
        return [i["line"] for i in json.loads(text)["items"]]
    except Exception:
        return text.splitlines()


def block(lines, tag):
    begin = next(i for i, l in enumerate(lines) if l.startswith(f"=={tag}_BEGIN=="))
    end = next(i for i, l in enumerate(lines) if i > begin and l.startswith(f"=={tag}_END=="))
    sha = re.search(r"sha256=([0-9a-f]{64})", lines[begin]).group(1)
    payload = base64.b64decode("".join(lines[begin + 1 : end]))
    if hashlib.sha256(payload).hexdigest() != sha:
        raise SystemExit(f"{tag}: checksum mismatch (log incomplete or altered)")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("artifacts/gen_fakes"))
    a = ap.parse_args()
    lines = [l for p in a.logs for l in lines_of(p)]
    feats_blob = block(lines, "GEN_FEATURES")
    meta_raw, npz = feats_blob.split(b"\n--NPZ--\n", 1)
    meta = json.loads(meta_raw)
    feats = np.load(io.BytesIO(npz))["feats"]
    assert len(meta) == len(feats), (len(meta), len(feats))
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "features.npz", feats=feats)
    (a.out / "meta.json").write_text(json.dumps(meta))
    samples = json.loads(block(lines, "GEN_SAMPLES"))
    for i, s in enumerate(samples):
        m = s["meta"]
        (a.out / f"sample_{m['generator']}_{m['style']}_{i:02d}.jpg").write_bytes(base64.b64decode(s["jpeg_b64"]))
    counts = {}
    for m in meta:
        k = f"{m['generator']}/{m['style']}"
        counts[k] = counts.get(k, 0) + 1
    print(f"verified; {len(meta)} images, features {feats.shape}, {len(samples)} samples -> {a.out}")
    print(counts)


if __name__ == "__main__":
    main()
