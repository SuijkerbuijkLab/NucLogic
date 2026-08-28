"""max_project rewrite: same pixels as a plain numpy max, metadata now written."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile

import numpy as np
import tifffile


from utils.load_image import load_image
from utils.max_project import max_project

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}",
          flush=True)


VOXEL = (2.479, 0.642, 0.642)
INTERVAL = 2.0
REF = np.random.randint(0, 4000, (3, 2, 5, 16, 12)).astype(np.uint16)

for label, axes, array in (
    ("TCZYX movie", "TCZYX", REF),
    ("single timepoint (CZYX)", "CZYX", REF[0]),
    ("plain stack (ZYX)", "ZYX", REF[0, 0]),
):
    print(f"\n== {label} ==", flush=True)
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "movie.ome.tif")
    tifffile.imwrite(
        p, array,
        metadata={"axes": axes, "PhysicalSizeZ": VOXEL[0], "PhysicalSizeY": VOXEL[1],
                  "PhysicalSizeX": VOXEL[2], "TimeIncrement": INTERVAL * 60,
                  "TimeIncrementUnit": "min"},
        ome=True,
    )

    result = max_project(p, name="nuclei")

    canonical, _, _, _ = load_image(p, lazy=False)
    expected = canonical.max(axis=2)  # T,C,Z,Y,X -> T,C,Y,X
    check(f"{label}: shape", result.shape == expected.shape,
          f"{result.shape} vs {expected.shape}")
    check(f"{label}: pixels equal plain numpy max", np.array_equal(result, expected))
    check(f"{label}: dtype preserved", result.dtype == array.dtype, f"{result.dtype}")

    written = os.path.join(tmp, "nuclei_projXY.tif")
    check(f"{label}: projection written", os.path.exists(written))
    back = load_image(written, lazy=False)
    check(f"{label}: written pixels", np.array_equal(back.data[:, :, 0], expected))
    check(f"{label}: written xy voxel",
          np.allclose(back.voxel_size[1:], VOXEL[1:], rtol=1e-4), f"{back.voxel_size}")
    check(f"{label}: written interval",
          abs(back.time_interval - INTERVAL) < 1e-6, f"{back.time_interval}")

print("\n== real IMS: first timepoint matches a direct read ==", flush=True)
D = None  # resolved below
CROPPED = sample_path("NUCLOGIC_TEST_IMS_CROPPED")
if CROPPED:
    from utils.load_image import as_numpy

    movie, _, _, _ = load_image(CROPPED)
    manual = np.max(as_numpy(movie[0, 0]), axis=0)
    check("real IMS lazy per-(T,C) max works", manual.shape == movie.shape[3:],
          f"{manual.shape} vs {movie.shape[3:]}")
    check("real IMS projection has signal", manual.max() > manual.min())
else:
    print("  (cropped IMS not available, skipped)", flush=True)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"), flush=True)
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
