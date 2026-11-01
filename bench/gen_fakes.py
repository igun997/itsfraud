"""Generate fake receipts/invoices with open image models and return DINOv2-base features for training.

Generators (both Apache-2.0, ungated): Tongyi-MAI/Z-Image-Turbo and Qwen/Qwen-Image. Images stay on the pod; only
features (norm + two augmented copies, same preprocessing as the Rust service) and a few small sample JPEGs are printed,
between ==GEN_FEATURES_BEGIN== / ==GEN_SAMPLES_BEGIN== markers, with a sha256 of the payload.
Prompts use fictional shop names only.

  python bench/gen_fakes.py --work /workspace/gen --n-zimage 160 --n-qwen 60
  python bench/gen_fakes.py --stub        # CPU, random images and random features, tests the plumbing
"""
import argparse
import base64
import gc
import hashlib
import io
import json
import random
import time

import numpy as np
from PIL import Image

SHOPS = ["Toko Sumber Rejeki", "Kopi Senja", "Warung Bu Tini", "Mini Mart Harapan", "Apotek Sehat Sentosa", "Bakery Roti Manis", "Cafe Lumbung", "Mercado Boa Vista",
         "Padaria Dois Irmaos", "Green Leaf Grocer", "Urban Cuts Barbershop", "Bengkel Maju Jaya", "Fresh Basket Market", "Restoran Selera Nusantara", "Pet Corner Supplies", "Toko Buku Cahaya"]
COMPANIES = ["PT Maju Bersama Abadi", "CV Karya Mandiri", "Northwind Supplies Ltd", "Lumen Office Solutions", "Studio Biru Design", "Alpha Logistics Sdn Bhd", "Vista Consulting GmbH", "Sinar Teknik Utama"]
CITIES = ["Jakarta", "Bandung", "Surabaya", "Lisbon", "Kuala Lumpur", "Singapore", "Medan", "Porto"]
SURFACES = ["a dark wooden table", "a brushed steel counter", "a white desk", "a car dashboard", "a marble countertop", "a grey fabric sofa"]
LANGS = ["Indonesian", "English", "Portuguese", "Malay"]
CURS = [("Rp", lambda r: f"{r.randint(15, 950) * 1000:,}".replace(",", ".")), ("RM", lambda r: f"{r.randint(5, 400)}.{r.randint(0, 99):02d}"), ("EUR", lambda r: f"{r.randint(5, 300)},{r.randint(0, 99):02d}"), ("USD", lambda r: f"{r.randint(5, 600)}.{r.randint(0, 99):02d}")]
STYLES = [("photo", 0.5), ("scan", 0.2), ("invoice", 0.3)]


def make_prompt(rng: random.Random):
    style = rng.choices([s for s, _ in STYLES], weights=[w for _, w in STYLES])[0]
    cur, fmt = rng.choice(CURS)
    total, lang, city = f"{cur} {fmt(rng)}", rng.choice(LANGS), rng.choice(CITIES)
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.choice([2023, 2024, 2025])}"
    if style == "photo":
        p = (f"A realistic phone photograph of a crumpled thermal paper shop receipt from '{rng.choice(SHOPS)}', {city}, lying on {rng.choice(SURFACES)}, slightly tilted, "
             f"uneven lighting and soft shadows, printed items with prices, subtotal and a total of {total}, date {date}, legible printed text in {lang}")
        doc = "receipt"
    elif style == "scan":
        p = (f"A flat scan of a {lang} store receipt from '{rng.choice(SHOPS)}', {city}, with a header and address, three to six purchased items with prices, subtotal, tax and "
             f"total {total}, date {date}, black text on white thermal paper")
        doc = "receipt"
    else:
        p = (f"A scanned A4 invoice from '{rng.choice(COMPANIES)}' in {lang}, company logo at the top, invoice number INV-{rng.randint(1000, 9999)}, a table with item descriptions, "
             f"quantities, unit prices and line totals, subtotal, VAT, total due {total}, due date {date}, clean office document")
        doc = "invoice"
    return dict(prompt=p, style=style, doc=doc)


def generate(name, n, work, deadline, stub, seed0):
    import torch

    rng = random.Random(seed0)
    prompts = [make_prompt(rng) for _ in range(n)]
    out = []
    if stub:
        for i, pr in enumerate(prompts):
            out.append((Image.fromarray(np.random.default_rng(i).integers(0, 255, (1024, 768, 3), dtype=np.uint8)), pr))
        return out
    if name == "z-image-turbo":
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16, low_cpu_mem_usage=False).to("cuda")
        kw = dict(height=1024, width=768, num_inference_steps=9, guidance_scale=0.0)
    elif name == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
        kw = dict(height=1024, width=768, num_inference_steps=30, guidance_scale=7.0, negative_prompt="blurry, low quality, cartoon")
    else:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
        kw = dict(height=1024, width=768, num_inference_steps=25, true_cfg_scale=4.0, negative_prompt=" ")
    t0 = time.time()
    for i, pr in enumerate(prompts):
        if time.time() > deadline:
            print(f"{name}: time budget reached after {i} images", flush=True)
            break
        img = pipe(prompt=pr["prompt"], generator=torch.Generator("cuda").manual_seed(seed0 * 1000 + i), **kw).images[0]
        out.append((img.convert("RGB"), pr))
        if i % 10 == 0:
            print(f"{name}: {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return out


def features(images, stub):
    """images: list of PIL. Returns (N, 3, 768) float16: norm, aug1, aug2 (same preprocessing as the service)."""
    from common import augment, normalize, page_tiles, to_tensor_array

    if stub:
        embed = lambda arr: np.random.default_rng(int(arr.sum()) % 2**31).normal(0, 1, (len(arr), 768)).astype(np.float32)
    else:
        import torch
        from transformers import AutoModel

        m = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()

        @torch.no_grad()
        def embed(arr):
            outs = []
            for i in range(0, len(arr), 64):
                x = torch.from_numpy(arr[i : i + 64]).to("cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    outs.append(m(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
            return np.concatenate(outs)

    feats = np.zeros((len(images), 3, 768), np.float16)
    for i, img in enumerate(images):
        base = normalize(img)
        for k, v in enumerate((base, augment(base, random.Random(i * 10 + 1)), augment(base, random.Random(i * 10 + 2)))):
            feats[i, k] = embed(to_tensor_array(page_tiles(v))).mean(0)
        if i % 50 == 0:
            print(f"features {i + 1}/{len(images)}", flush=True)
    return feats


def emit(tag, payload: bytes):
    b64 = base64.b64encode(payload).decode()
    print(f"==%s_BEGIN== sha256={hashlib.sha256(payload).hexdigest()} bytes={len(payload)}" % tag)
    for i in range(0, len(b64), 12000):
        print(b64[i : i + 12000])
    print(f"==%s_END==" % tag, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/gen")
    ap.add_argument("--n-zimage", type=int, default=160)
    ap.add_argument("--n-qwen", type=int, default=60)
    ap.add_argument("--qwen-minutes", type=float, default=15)
    ap.add_argument("--stub", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    imgs, meta = [], []
    for name, n, seed, budget in (("z-image-turbo", a.n_zimage, 1, 12 * 60), ("qwen-image", a.n_qwen, 2, a.qwen_minutes * 60)):
        try:
            got = generate(name, n, a.work, time.time() + budget, a.stub, seed)
        except Exception as e:  # keep whatever the other generator produced
            print(f"{name} FAILED: {e!r}"[:400], flush=True)
            continue
        for k, (img, pr) in enumerate(got):
            imgs.append(img)
            meta.append(dict(generator=name, style=pr["style"], doc=pr["doc"], prompt=pr["prompt"], idx=k))
        print(f"{name}: {len(got)} images", flush=True)
    if not imgs:
        raise SystemExit("no images generated")
    feats = features(imgs, a.stub)
    buf = io.BytesIO()
    np.savez_compressed(buf, feats=feats)
    emit("GEN_FEATURES", json.dumps(meta).encode() + b"\n--NPZ--\n" + buf.getvalue())
    samples, seen = [], {}
    for img, m in zip(imgs, meta):
        key = (m["generator"], m["style"])
        if seen.get(key, 0) >= 3:
            continue
        seen[key] = seen.get(key, 0) + 1
        small = img.copy()
        small.thumbnail((512, 512))
        b = io.BytesIO()
        small.save(b, "JPEG", quality=80)
        samples.append(dict(meta=m, jpeg_b64=base64.b64encode(b.getvalue()).decode()))
    emit("GEN_SAMPLES", json.dumps(samples).encode())
    print(f"total {len(imgs)} images in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
