"""Run a real Nikon ND2 file through utils/load_image.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import sys
import time

import numpy as np

ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)

PATH = require("NUCLOGIC_TEST_ND2")

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}")


print("== what the nd2 package reports ==")
import nd2

with nd2.ND2File(PATH) as f:
    sizes = dict(f.sizes)
    voxel = f.voxel_size()
    print("  sizes        :", sizes)
    print("  voxel_size() :", voxel)
    print("  dtype        :", f.dtype)
    print("  is_rgb       :", f.is_rgb)
    print("  experiment   :", f.experiment)
    try:
        print("  channels     :", [c.channel.name for c in f.metadata.channels])
    except Exception as e:
        print("  channels     : ?", e)

print("\n== load_image (lazy) ==")
from utils.load_image import as_numpy, load_image

t0 = time.time()
r = load_image(PATH, lazy=True)
t_open = time.time() - t0
print(f"  open took       : {t_open:.2f}s")
print(f"  shape (T,C,Z,Y,X): {r.data.shape}")
print(f"  dtype           : {r.data.dtype}")
print(f"  voxel_size      : {r.voxel_size}")
print(f"  time_interval   : {r.time_interval} h")
print(f"  metadata_missing: {r.metadata_missing}")

import dask.array as da

check("5D TCZYX", r.data.ndim == 5, f"got {r.data.shape}")
# nd2 hands back ResourceBackedDaskArray, a dask.array.Array subclass that keeps
# the file handle alive; isinstance is the meaningful check, not the module name.
check("lazy is a dask array", isinstance(r.data, da.Array), f"{type(r.data)}")
check("open is fast (lazy)", t_open < 10, f"{t_open:.2f}s")
check("voxel is 3-tuple of floats", len(r.voxel_size) == 3
      and all(isinstance(v, float) for v in r.voxel_size))

# Axis sanity: Y/X should be the two largest dims and match the raw file.
expected_y, expected_x = sizes.get("Y"), sizes.get("X")
check("Y matches file", r.data.shape[3] == expected_y, f"{r.data.shape[3]} vs {expected_y}")
check("X matches file", r.data.shape[4] == expected_x, f"{r.data.shape[4]} vs {expected_x}")
check("C matches file", r.data.shape[1] == sizes.get("C", 1),
      f"{r.data.shape[1]} vs {sizes.get('C', 1)}")
check("Z matches file", r.data.shape[2] == sizes.get("Z", 1),
      f"{r.data.shape[2]} vs {sizes.get('Z', 1)}")
check("T matches file", r.data.shape[0] == sizes.get("T", 1),
      f"{r.data.shape[0]} vs {sizes.get('T', 1)}")

print("\n== partial read (the point of laziness) ==")
t0 = time.time()
plane = as_numpy(r.data[0, 0, r.data.shape[2] // 2])
t_plane = time.time() - t0
print(f"  one Z plane     : {plane.shape} in {t_plane:.2f}s  "
      f"min/max={plane.min()}/{plane.max()}")
check("plane is 2D YX", plane.shape == (expected_y, expected_x), f"got {plane.shape}")
check("plane read is fast", t_plane < 15, f"{t_plane:.2f}s")
check("plane has real signal", plane.max() > plane.min())

print("\n== pixel fidelity vs the nd2 package directly ==")
raw = nd2.imread(PATH)
print(f"  nd2.imread shape: {raw.shape}  (axes {''.join(sizes.keys())})")
axes = "".join(sizes.keys()).upper()

# Independently reproduce what load_image should have done: take position 0,
# reorder to TCZYX, insert the missing Z axis.
if "P" in axes:
    p = axes.index("P")
    raw = raw[(slice(None),) * p + (0,)]
    axes = axes.replace("P", "")
order = [axes.index(a) for a in "TCZYX" if a in axes]
reordered = np.transpose(raw, order)
for i, a in enumerate("TCZYX"):
    if a not in axes:
        reordered = np.expand_dims(reordered, axis=i)
check("shape matches independent transpose", r.data.shape == reordered.shape,
      f"{r.data.shape} vs {reordered.shape}")
check("pixels identical to nd2.imread", np.array_equal(as_numpy(r.data), reordered))

print("\n== eager path ==")
r2 = load_image(PATH, lazy=False)
check("eager is numpy", isinstance(r2.data, np.ndarray), f"{type(r2.data)}")
check("eager equals lazy", np.array_equal(r2.data, as_numpy(r.data)))
check("eager metadata equals lazy",
      (r2.voxel_size, r2.time_interval, r2.metadata_missing)
      == (r.voxel_size, r.time_interval, r.metadata_missing))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
