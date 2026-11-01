"""Score every PDF/image under tests/pdfs and data/real with the Rust service and summarize by folder.

Layout: a path containing a `fake` directory is expected to be AI generated, one containing `real` is expected genuine.
Results are grouped by folder (for example data/fake/gemini, data/real/invoices). Defaults: tests/pdfs and data.

Setup (PDF input needs the pdfium library):
  cargo build --release --features pdf
  python scripts/check_pdfs.py --pdfium-dir <folder containing libpdfium.so>
  # for example the one bundled with the pypdfium2 Python package: .../site-packages/pypdfium2_raw
Optional: --threshold-key threshold_fpr5 to also report the looser operating point.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


def expected(path: Path):
    parts = path.parts
    if "fake" in parts:
        return 1
    if "real" in parts:
        return 0
    return None


def group(path: Path, root: Path):
    return str(path.parent)


def scan(binary, artifacts, env, path):
    r = subprocess.run([binary, "scan", str(path)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return dict(error=(r.stderr.strip().splitlines() or ["failed"])[-1])
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="target/release/itsfraud")
    ap.add_argument("--artifacts", default="models/v0.1.0")
    ap.add_argument("--pdfium-dir", default=None)
    ap.add_argument("--roots", nargs="*", default=["tests/pdfs", "data"])
    ap.add_argument("--threshold-key", default=None, help="also report using this head threshold, e.g. threshold_fpr5")
    args = ap.parse_args()

    env = dict(os.environ, ITSFRAUD_ARTIFACTS=args.artifacts)
    if args.pdfium_dir:
        env["LD_LIBRARY_PATH"] = args.pdfium_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    head = json.loads((Path(args.artifacts) / "head.json").read_text())
    alt = head.get(args.threshold_key) if args.threshold_key else None

    rows = []
    for root in map(Path, args.roots):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                exp = expected(p)
                if exp is None:
                    continue
                rows.append((p, root, exp))
    if not rows:
        sys.exit("no files found")

    stats = defaultdict(lambda: dict(n=0, flagged=0, flagged_alt=0, errors=0))
    detail = []
    for p, root, exp in rows:
        res = scan(args.binary, args.artifacts, env, p)
        g = group(p, root)
        st = stats[g]
        st["n"] += 1
        st["exp"] = exp
        if "error" in res:
            st["errors"] += 1
            detail.append((g, p.name, "ERROR", res["error"]))
            continue
        st["flagged"] += int(res["ai_generated"])
        if alt is not None:
            st["flagged_alt"] += int(res["logit"] > alt)
        detail.append((g, p.name, "AI" if res["ai_generated"] else "real", f"logit {res['logit']:+.2f} by {res['decided_by']} pages {res['pages']}"))

    print("per file:")
    for g, name, verdict, info in detail:
        print(f"  [{g}] {name}: {verdict} ({info})")
    print("\nper group (expected -> flagged):")
    tp = fn = fp = tn = 0
    for g, st in sorted(stats.items()):
        ok = st["n"] - st["errors"]
        line = f"  {g}: {st['flagged']}/{ok} flagged as AI (expected {'AI' if st['exp'] else 'real'})"
        if alt is not None:
            line += f"; at {args.threshold_key}: {st['flagged_alt']}/{ok}"
        if st["errors"]:
            line += f"; {st['errors']} errors"
        print(line)
        if st["exp"]:
            tp += st["flagged"]
            fn += ok - st["flagged"]
        else:
            fp += st["flagged"]
            tn += ok - st["flagged"]
    if tp + fn:
        print(f"\nfake detection rate: {tp}/{tp + fn} = {tp / (tp + fn):.0%}")
    if fp + tn:
        print(f"false alarms on real: {fp}/{fp + tn} = {fp / (fp + tn):.0%}")
    print("(small samples: treat as a smoke test, not a measurement)")


if __name__ == "__main__":
    main()
