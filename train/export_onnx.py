"""Export the DINOv2-S backbone (CLS embedding, dynamic batch) to ONNX for the Rust server."""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from extract_features import MODEL_ID


class Cls(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).last_hidden_state[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("../artifacts/backbone.onnx"))
    args = ap.parse_args()

    model = Cls(AutoModel.from_pretrained(MODEL_ID)).eval()
    dummy = torch.randn(2, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(args.out), input_names=["pixel_values"], output_names=["cls"],
        dynamic_axes={"pixel_values": {0: "n"}, "cls": {0: "n"}}, opset_version=17, dynamo=False,
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out))
    x = np.random.randn(3, 3, 224, 224).astype(np.float32)
    got = sess.run(None, {"pixel_values": x})[0]
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    print(f"wrote {args.out}; max abs diff vs torch = {np.abs(got - want).max():.2e}")


if __name__ == "__main__":
    main()
