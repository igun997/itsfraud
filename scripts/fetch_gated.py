"""Download subsets of gated Scam-AI datasets into artifacts/gated (git-ignored; licence is CC-BY-NC-SA-4.0, non-commercial).

Needs HF_TOKEN in .env from an account that accepted each dataset's terms. The token is read locally and sent only to huggingface.co.

  python scripts/fetch_gated.py --n-gpt4o 250 --n-aiforge 200
"""
import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def token():
    m = re.search(r'^\s*(?:export\s+)?HF_TOKEN\s*=\s*["\']?([^"\'\s#]+)', Path(".env").read_text(), re.M)
    if not m:
        raise SystemExit("HF_TOKEN not found in .env")
    return m.group(1)


def fetch(repo, files, out, tok):
    def one(f):
        return hf_hub_download(repo, f, repo_type="dataset", local_dir=str(out), token=tok)

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gpt4o", type=int, default=250)
    ap.add_argument("--n-aiforge", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("artifacts/gated"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok, rng = token(), random.Random(a.seed)

    files = sorted(f for f in list_repo_files("Scam-AI/gpt4o-receipt", repo_type="dataset", token=tok) if f.endswith(".png"))
    pick = rng.sample(files, min(a.n_gpt4o, len(files)))
    fetch("Scam-AI/gpt4o-receipt", pick, a.out / "gpt4o_receipt", tok)
    print(f"gpt4o-receipt: {len(pick)} of {len(files)} PNGs (all AI generated)", flush=True)

    files = list_repo_files("Scam-AI/AIForge-Doc-v1", repo_type="dataset", token=tok)
    for sub in ("authentic", "images"):
        cand = sorted(f for f in files if f.startswith(f"TestingSet/{sub}/") and f.endswith(".png"))
        pick = rng.sample(cand, min(a.n_aiforge, len(cand)))
        fetch("Scam-AI/AIForge-Doc-v1", pick, a.out / "aiforge_v1", tok)
        print(f"AIForge-Doc-v1 TestingSet/{sub}: {len(pick)} of {len(cand)}", flush=True)


if __name__ == "__main__":
    main()
