"""Multi-chunk reads on real IMS -- the case the single-chunk tests missed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import sys

import numpy as np


from imaris_ims_file_reader.ims import ims
from utils.load_image import as_numpy, load_image

D = None  # resolved below
BIG = require("NUCLOGIC_TEST_IMS")

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}",
          flush=True)


r = load_image(BIG, lazy=True)
raw = ims(BIG)
print(f"  shape {r.data.shape} chunks {r.data.chunksize}", flush=True)

print("\n== multi-chunk: 2 timepoints, all channels (12 chunks) ==", flush=True)
block = as_numpy(r.data[0:2])
check("shape keeps 5D", block.shape == (2, 3, 77, 1024, 1024), f"{block.shape}")
check("[0,1] matches reader", np.array_equal(block[0, 1], raw[0, 1]))
check("[1,2] matches reader", np.array_equal(block[1, 2], raw[1, 2]))

print("\n== multi-chunk across channels only ==", flush=True)
block2 = as_numpy(r.data[5, 0:3])
check("shape keeps 4D", block2.shape == (3, 77, 1024, 1024), f"{block2.shape}")
check("channel 0 matches", np.array_equal(block2[0], raw[5, 0]))
check("channel 2 matches", np.array_equal(block2[2], raw[5, 2]))

print("\n== slice with length-1 range must not collapse ==", flush=True)
block3 = as_numpy(r.data[0:1, 0:1])
check("length-1 slices keep dims", block3.shape == (1, 1, 77, 1024, 1024),
      f"{block3.shape}")
check("content correct", np.array_equal(block3[0, 0], raw[0, 0]))

print("\n== a dask reduction over several chunks ==", flush=True)
sub = r.data[0:3, 0, 38]
mx = sub.max().compute()
expected = max(raw[t, 0, 38].max() for t in range(3))
check("max over 3 chunks", mx == expected, f"{mx} vs {expected}")

print("\n== single chunk still fine (regression) ==", flush=True)
check("single (t,c)", np.array_equal(as_numpy(r.data[17, 1]), raw[17, 1]))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"), flush=True)
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
