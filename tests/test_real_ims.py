"""Run real Imaris .ims files through utils/load_image.py.

The originals are 17 GB / 5.4 GB in RAM, so the eager path is deliberately not
exercised here -- only lazy access and partial reads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import sys
import time

import numpy as np


import dask.array as da

from imaris_ims_file_reader.ims import ims
from utils.load_image import as_numpy, load_image

D = None  # resolved below
BIG = require("NUCLOGIC_TEST_IMS")
CROPPED = require("NUCLOGIC_TEST_IMS_CROPPED")

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}")


for label, path, expected_shape in (
    ("32 GB original", BIG, (35, 3, 77, 1024, 1024)),
    ("cropped", CROPPED, (35, 3, 76, 597, 570)),
):
    print(f"\n=== {label} ===")
    t0 = time.time()
    r = load_image(path, lazy=True)
    t_open = time.time() - t0

    gb = np.prod(r.data.shape) * 2 / 1e9
    print(f"  open            : {t_open:.2f}s   ({gb:.1f} GB if materialised)")
    print(f"  shape           : {r.data.shape}")
    print(f"  chunks          : {r.data.chunksize}")
    print(f"  voxel_size      : {r.voxel_size}")
    print(f"  time_interval   : {r.time_interval} h")
    print(f"  metadata_missing: {r.metadata_missing}")

    check(f"{label}: 5D", r.data.ndim == 5, f"got {r.data.shape}")
    check(f"{label}: shape", r.data.shape == expected_shape, f"got {r.data.shape}")
    check(f"{label}: dask array", isinstance(r.data, da.Array), f"{type(r.data)}")
    check(f"{label}: open is lazy/fast", t_open < 5, f"{t_open:.2f}s")
    check(f"{label}: dtype uint16", r.data.dtype == np.uint16, f"{r.data.dtype}")
    check(f"{label}: voxel from file",
          np.allclose(r.voxel_size, (2.479, 0.642, 0.642)), f"{r.voxel_size}")
    check(f"{label}: metadata present", r.metadata_missing is False)
    check(f"{label}: chunk is one (T,C) stack",
          r.data.chunksize == (1, 1) + expected_shape[2:], f"{r.data.chunksize}")

    print("  -- partial reads vs the reader itself --")
    raw = ims(path)
    for t, c in ((0, 0), (17, 1), (34, 2)):
        t0 = time.time()
        mine = as_numpy(r.data[t, c])
        dt = time.time() - t0
        theirs = raw[t, c]
        check(f"{label}: [{t},{c}] pixels identical",
              np.array_equal(mine, theirs), "mismatch")
        print(f"      [{t},{c}] {mine.shape} in {dt:.2f}s  "
              f"mean={mine.mean():.1f}")

    print("  -- single Z plane --")
    t0 = time.time()
    plane = as_numpy(r.data[0, 0, expected_shape[2] // 2])
    print(f"      plane {plane.shape} in {time.time() - t0:.2f}s")
    check(f"{label}: plane matches reader",
          np.array_equal(plane, raw[0, 0, expected_shape[2] // 2]))

    print("  -- channel independence (nuclei channel selection) --")
    c0 = as_numpy(r.data[0, 0, 40])
    c1 = as_numpy(r.data[0, 1, 40])
    check(f"{label}: channels differ", not np.array_equal(c0, c1),
          "channels identical - possible axis mix-up")

print("\n=== old pipeline path vs load_image (same file, same pixels) ===")
# What segment_organoid does today for .ims
old = ims(BIG)
while old.ndim < 5:
    old = np.expand_dims(old, axis=0)
new = load_image(BIG, lazy=True).data
check("old and new agree on shape", old.shape == new.shape, f"{old.shape} vs {new.shape}")
check("old and new agree on pixels [3,1]",
      np.array_equal(np.asarray(old[3, 1]), as_numpy(new[3, 1])))
check("old and new agree on voxel size",
      tuple(ims(BIG).resolution) == load_image(BIG).voxel_size)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
