"""Inline train/common.py into bench/run_benchmark.py and strip comments/docstrings.

Produces one dependency-free file that can be shipped to a pod as a single environment
variable. Prints its length and sha256 prefix so the pod can verify an intact copy.

  python bench/flatten.py out.py
"""
import ast
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
common = (ROOT / "train/common.py").read_text()
bench = (ROOT / "bench/run_benchmark.py").read_text()

bench = bench.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))\nfrom common import augment, page_tiles  # noqa: E402\n', "")
common = re.sub(r'^"""[\s\S]*?"""\n', "", common, count=1)
common = common.replace("import io\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nfrom PIL import Image\n", "")
anchor = re.search(r"^Image\.MAX_IMAGE_PIXELS", bench, re.M)
bench = bench[: anchor.start()] + common + "\n" + bench[anchor.start() :]

tree = ast.parse(bench)
for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
out = ast.unparse(tree) + "\n"
compile(out, "flat.py", "exec")
Path(sys.argv[1]).write_text(out)
print(len(out), "bytes, sha256", hashlib.sha256(out.encode()).hexdigest()[:16])
