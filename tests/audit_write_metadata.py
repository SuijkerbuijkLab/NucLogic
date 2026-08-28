"""Static audit: what metadata does each pipeline TIFF write actually record?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import pathlib
import re

ROOT = pathlib.Path(PROJECT_ROOT)

FILES = [
    "main_functions/add_advanced_statistics.py",
    "main_functions/segment_organoid.py",
    "main_functions/crop_sample.py",
    "main_functions/split_phenotype_mask.py",
    "utils/crop.py",
    "utils/crop_fixed.py",
    "utils/max_project.py",
]

AXES_RE = re.compile(r'"axes":\s*"([A-Z]+)"')
NAME_RE = re.compile(r'f?"([^"]*\.tif)"')

rows = []
for rel in FILES:
    src = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"tifffile\.imwrite\(", src):
        if src[:m.start()].rstrip().endswith("#"):
            continue
        depth, i = 0, m.end() - 1
        while i < len(src):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        call = src[m.start():i + 1]
        line = src[:m.start()].count("\n") + 1

        name_m = NAME_RE.search(call)
        axes_m = AXES_RE.search(call)
        rows.append({
            "site": f"{rel}:{line}",
            "target": name_m.group(1) if name_m else "?",
            "axes": axes_m.group(1) if axes_m else "-",
            "xy": "resolution=" in call,
            "z": "spacing" in call,
            "t": "finterval" in call,
        })

print(f"{'site':<48} {'target':<34} {'axes':<6} {'xy':<4} {'z':<4} {'t':<4}")
print("-" * 106)
for r in rows:
    print(f"{r['site']:<48} {r['target']:<34} {r['axes']:<6} "
          f"{'YES' if r['xy'] else '--':<4} {'YES' if r['z'] else '--':<4} "
          f"{'YES' if r['t'] else '--':<4}")

complete = [r for r in rows if r["xy"] and r["z"] and r["t"]]
print(f"\n{len(complete)}/{len(rows)} writes record full spatial+temporal metadata")
