"""Shared preprocessing. The Rust crate (crates/detector/src/tiles.rs) must stay
byte-for-byte equivalent: native-resolution 224px tiles, integer-only layout math."""
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 224
MAX_TILES = 8
MAX_PDF_PAGES = 5
PDF_DPI = 150
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_pages(path: Path) -> list[Image.Image]:
    """Image file -> [image]; PDF -> rendered pages (first MAX_PDF_PAGES)."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(min(len(pdf), MAX_PDF_PAGES)):
            pages.append(pdf[i].render(scale=PDF_DPI / 72).to_pil().convert("RGB"))
        return pages
    return [Image.open(path).convert("RGB")]


def axis_positions(size: int) -> list[int]:
    if size <= TILE:
        return [0]
    n = -(-size // TILE)  # ceil
    return [(i * (size - TILE) + (n - 1) // 2) // (n - 1) for i in range(n)]


def page_tiles(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    if w < TILE or h < TILE:
        canvas = Image.new("RGB", (max(w, TILE), max(h, TILE)), (255, 255, 255))
        canvas.paste(img, (0, 0))
        img = canvas
        w, h = img.size
    boxes = [(x, y) for y in axis_positions(h) for x in axis_positions(w)]
    if len(boxes) > MAX_TILES:
        boxes = [boxes[(k * len(boxes)) // MAX_TILES] for k in range(MAX_TILES)]
    return [img.crop((x, y, x + TILE, y + TILE)) for x, y in boxes]


def to_tensor_array(tiles: list[Image.Image]) -> np.ndarray:
    arr = np.stack([np.asarray(t, dtype=np.float32) / 255.0 for t in tiles])
    arr = (arr - MEAN) / STD
    return arr.transpose(0, 3, 1, 2).astype(np.float32)  # N,3,H,W


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mimic real-world handling: downscale and JPEG recompression (WhatsApp/screenshot-like)."""
    if rng.random() < 0.6:
        s = rng.uniform(0.5, 1.0)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    if rng.random() < 0.8:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(45, 92))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img
