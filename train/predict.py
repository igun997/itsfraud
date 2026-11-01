"""Reference scorer (onnxruntime + head.json). Use it to check the Rust detector gives the same score."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import load_pages, normalize, page_tiles, to_tensor_array


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--artifacts", type=Path, default=Path("../artifacts"))
    args = ap.parse_args()

    head = json.loads((args.artifacts / "head.json").read_text())
    sess = ort.InferenceSession(str(args.artifacts / "backbone.onnx"))
    norm = head.get("preprocess", "").startswith("long side 1024")
    tiles = [t for p in load_pages(args.file) for t in page_tiles(normalize(p) if norm else p)]
    emb = sess.run(None, {"pixel_values": to_tensor_array(tiles)})[0]
    z = (emb - np.array(head["mean"])) / np.array(head["scale"])
    logit = float((z @ np.array(head["coef"]) + head["intercept"]).mean())
    print(json.dumps({"tiles": len(tiles), "logit": logit, "probability": 1 / (1 + np.exp(-logit)),
                      "ai_generated": logit > head["threshold"]}))


if __name__ == "__main__":
    main()
