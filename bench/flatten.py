"""Inline train/common.py (and run_benchmark.py for train_final) into one dependency-free file.

Strips comments/docstrings so it fits in a single environment variable on a pod. Prints its
length and sha256 prefix so the pod can verify an intact copy.

  python bench/flatten.py out.py [run_benchmark|train_final]
"""
import ast
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
entry = sys.argv[2] if len(sys.argv) > 2 else "run_benchmark"

common = (ROOT / "train/common.py").read_text()
bench = (ROOT / "bench/run_benchmark.py").read_text()
bench = bench.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))\nfrom common import augment, page_tiles  # noqa: E402\n', "")
common = re.sub(r'^"""[\s\S]*?"""\n', "", common, count=1)
common = common.replace("import io\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nfrom PIL import Image\n", "")
anchor = re.search(r"^Image\.MAX_IMAGE_PIXELS", bench, re.M)
bench = bench[: anchor.start()] + common + "\n" + bench[anchor.start() :]
tree = ast.parse(bench)

CHAIN = {"run_benchmark": [], "train_final": ["train_final"], "compare_detectors": ["train_final", "compare_detectors"]}[entry]
LOCAL = {"run_benchmark", "train_final", "compare_detectors"}


def drop_main_guard(body):
    return [n for n in body if not (isinstance(n, ast.If) and "__main__" in ast.unparse(n.test))]


if CHAIN:
    for i, mod in enumerate(CHAIN):
        extra = ast.parse((ROOT / f"bench/{mod}.py").read_text())
        extra.body = [n for n in extra.body if not (isinstance(n, ast.ImportFrom) and n.module in LOCAL)]
        tree.body = drop_main_guard(tree.body) + extra.body
    tree.body = [ast.parse("import hashlib").body[0]] + tree.body


def strip_docstrings(t):
    for node in ast.walk(t):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]


strip_docstrings(tree)
out = ast.unparse(tree) + "\n"
compile(out, "flat.py", "exec")
Path(sys.argv[1]).write_text(out)
print(len(out), "bytes, sha256", hashlib.sha256(out.encode()).hexdigest()[:16])
