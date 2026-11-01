"""Extract frozen DINOv2-S features for every tile of every file under data/.

Layout: data/real/**  (label 0)   data/fake/<generator>/**  (label 1; <generator> is used
for leave-one-generator-out evaluation). Images and PDFs are both accepted.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from common import IMAGE_EXTS, augment, load_pages, page_tiles, to_tensor_array

MODEL_ID = "facebook/dinov2-small"


def collect(data: Path):
    items = []  # (path, label, generator)
    for p in sorted((data / "real").rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS | {".pdf"}:
            items.append((p, 0, "real"))
    for gen_dir in sorted(d for d in (data / "fake").iterdir() if d.is_dir()):
        for p in sorted(gen_dir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS | {".pdf"}:
                items.append((p, 1, gen_dir.name))
    return items


@torch.no_grad()
def embed(model, tiles_np: np.ndarray, device, bs=32) -> np.ndarray:
    out = []
    for i in range(0, len(tiles_np), bs):
        x = torch.from_numpy(tiles_np[i : i + bs]).to(device)
        out.append(model(pixel_values=x).last_hidden_state[:, 0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../artifacts/features.npz"))
    ap.add_argument("--aug-copies", type=int, default=2, help="augmented copies per image (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    rng = random.Random(args.seed)

    feats, file_ids, labels, gens, variants, paths = [], [], [], [], [], []
    items = collect(args.data)
    if not items:
        raise SystemExit(f"no files found under {args.data}/real and {args.data}/fake/<generator>/")
    for fid, (path, label, gen) in enumerate(tqdm(items)):
        try:
            pages = load_pages(path)
        except Exception as e:  # corrupt file: skip, do not poison the dataset
            print(f"skip {path}: {e}")
            continue
        is_pdf = path.suffix.lower() == ".pdf"
        # variant 0 = untouched; augmented copies only for raster images
        for v in range(1 + (0 if is_pdf else args.aug_copies)):
            pgs = pages if v == 0 else [augment(p, rng) for p in pages]
            tiles = [t for p in pgs for t in page_tiles(p)]
            f = embed(model, to_tensor_array(tiles), device)
            feats.append(f)
            file_ids += [fid] * len(f)
            labels += [label] * len(f)
            gens += [gen] * len(f)
            variants += [v] * len(f)
        paths.append(str(path))

    np.savez_compressed(
        args.out,
        feats=np.concatenate(feats),
        file_id=np.array(file_ids),
        label=np.array(labels),
        gen=np.array(gens),
        variant=np.array(variants),
        paths=np.array(paths),
    )
    print(f"saved {args.out}: {sum(len(f) for f in feats)} tiles from {len(paths)} files")


if __name__ == "__main__":
    main()
