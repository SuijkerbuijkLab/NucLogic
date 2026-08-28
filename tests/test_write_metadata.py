"""Do the files the pipeline writes read back correctly through load_image?

The TIFF writers go through utils/save_as_tiff.save_as_tiff, the same call the
pipeline makes, so this exercises the real thing rather than a copy of it.
tifffile warnings are promoted to errors for those writes: that is what catches
a return of the "nonconformant BigTIFF ImageJ" combination.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import tifffile


import warnings

from utils.load_image import ims_acquisition_start, load_image
from utils.save_as_ims import save_as_ims
from utils.save_as_tiff import save_as_tiff as _save_as_tiff

failures = []


def save_as_tiff(*args, **kwargs):
    """The pipeline's writer, with any tifffile warning turned into a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        return _save_as_tiff(*args, **kwargs)


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}",
          flush=True)


tmp = tempfile.mkdtemp()
VOXEL = (2.479, 0.642, 0.642)     # (z, y, x) um -- from the real Elise sample
INTERVAL = 2.0                     # hours
T, C, Z, Y, X = 3, 2, 4, 16, 12    # Y != X to catch transposes

MOVIE_TCZYX = np.random.randint(0, 4000, (T, C, Z, Y, X)).astype(np.uint16)
MASK_TZYX = np.random.randint(0, 50, (T, Z, Y, X)).astype(np.uint16)


print("== crop_sample: {name}_cropped.tif (TZCYX) ==", flush=True)
p = os.path.join(tmp, "s_cropped.tif")
save_as_tiff(
    p,
    np.transpose(MOVIE_TCZYX, (0, 2, 1, 3, 4)),  # TCZYX -> TZCYX as the pipeline does
    "TZCYX",
    VOXEL,
    INTERVAL,
)
r = load_image(p, lazy=False)
print(f"    shape={r.data.shape} voxel={r.voxel_size} interval={r.time_interval} "
      f"missing={r.metadata_missing}", flush=True)
check("cropped.tif shape back to TCZYX", r.data.shape == MOVIE_TCZYX.shape,
      f"{r.data.shape}")
check("cropped.tif pixels identical", np.array_equal(r.data, MOVIE_TCZYX))
check("cropped.tif voxel", np.allclose(r.voxel_size, VOXEL, rtol=1e-4),
      f"{r.voxel_size} vs {VOXEL}")
check("cropped.tif interval", abs(r.time_interval - INTERVAL) < 1e-6,
      f"{r.time_interval}")
check("cropped.tif metadata_missing False", r.metadata_missing is False)

print("\n== segment_organoid: {name}_segmented.tif (TZYX mask) ==", flush=True)
p = os.path.join(tmp, "s_segmented.tif")
save_as_tiff(p, MASK_TZYX, "TZYX", VOXEL, INTERVAL)
r = load_image(p, lazy=False)
print(f"    shape={r.data.shape} voxel={r.voxel_size} interval={r.time_interval}",
      flush=True)
check("segmented.tif is 5D with C=1", r.data.shape == (T, 1, Z, Y, X), f"{r.data.shape}")
check("segmented.tif pixels", np.array_equal(r.data[:, 0], MASK_TZYX))
check("segmented.tif voxel", np.allclose(r.voxel_size, VOXEL, rtol=1e-4),
      f"{r.voxel_size}")
check("segmented.tif interval", abs(r.time_interval - INTERVAL) < 1e-6,
      f"{r.time_interval}")

print("\n== segment_organoid: Frame-N.tif (single timepoint TZCYX) ==", flush=True)
p = os.path.join(tmp, "Frame-0.tif")
frame_tzcyx = np.transpose(MOVIE_TCZYX[0][np.newaxis], (0, 2, 1, 3, 4))
save_as_tiff(p, frame_tzcyx, "TZCYX", VOXEL, INTERVAL)
r = load_image(p, lazy=False)
print(f"    shape={r.data.shape} voxel={r.voxel_size} missing={r.metadata_missing}",
      flush=True)
check("Frame-N.tif shape", r.data.shape == (1, C, Z, Y, X), f"{r.data.shape}")
check("Frame-N.tif pixels", np.array_equal(r.data[0], MOVIE_TCZYX[0]))
check("Frame-N.tif voxel", np.allclose(r.voxel_size, VOXEL, rtol=1e-4),
      f"{r.voxel_size}")

print("\n== split_phenotype_mask: {phenotype}_mask.tif ==", flush=True)
p = os.path.join(tmp, "s_pheno_mask.tif")
save_as_tiff(p, MASK_TZYX, "TZYX", VOXEL, INTERVAL)
r = load_image(p, lazy=False)
print(f"    shape={r.data.shape} voxel={r.voxel_size} interval={r.time_interval}",
      flush=True)
check("pheno mask pixels survive", np.array_equal(r.data[:, 0], MASK_TZYX))
check("pheno mask voxel preserved", np.allclose(r.voxel_size, VOXEL, rtol=1e-4),
      f"{r.voxel_size}")
check("pheno mask interval preserved", abs(r.time_interval - INTERVAL) < 1e-6,
      f"{r.time_interval}")

print("\n== max_project / crop: _projXY.tif (TCYX, Z projected away) ==", flush=True)
p = os.path.join(tmp, "s_projXY.tif")
proj = MOVIE_TCZYX.max(axis=2)  # T,C,Y,X
save_as_tiff(p, proj, "TCYX", (1.0, VOXEL[1], VOXEL[2]), INTERVAL)
r = load_image(p, lazy=False)
print(f"    shape={r.data.shape} voxel={r.voxel_size} interval={r.time_interval}",
      flush=True)
check("projXY becomes 5D with Z=1", r.data.shape == (T, C, 1, Y, X), f"{r.data.shape}")
check("projXY pixels", np.array_equal(r.data[:, :, 0], proj))
check("projXY xy voxel preserved",
      np.allclose(r.voxel_size[1:], VOXEL[1:], rtol=1e-4), f"{r.voxel_size}")
check("projXY interval preserved", abs(r.time_interval - INTERVAL) < 1e-6,
      f"{r.time_interval}")
check("projXY z voxel is 1.0 (Z was projected away)", r.voxel_size[0] == 1.0,
      f"{r.voxel_size[0]}")

print("\n== crop_sample: {name}_cropped.ims ==", flush=True)
p = os.path.join(tmp, "s_cropped.ims")
origin = datetime(2025, 11, 21, 14, 47, 58)
save_as_ims(input_movie=MOVIE_TCZYX, output_filename=p, voxel_size=VOXEL,
            time_interval=INTERVAL, start_time=origin)
r = load_image(p, lazy=False)
print(f"    shape={r.data.shape} voxel={r.voxel_size} interval={r.time_interval} "
      f"start={ims_acquisition_start(p)}", flush=True)
check("cropped.ims shape", r.data.shape == MOVIE_TCZYX.shape, f"{r.data.shape}")
check("cropped.ims pixels", np.array_equal(r.data, MOVIE_TCZYX))
check("cropped.ims voxel", np.allclose(r.voxel_size, VOXEL, rtol=1e-4),
      f"{r.voxel_size}")
check("cropped.ims interval", abs(r.time_interval - INTERVAL) < 1e-6,
      f"{r.time_interval}")
check("cropped.ims start time", ims_acquisition_start(p) == origin)

print("\n== .ims vs .tif crop outputs agree on metadata ==", flush=True)
tif = load_image(os.path.join(tmp, "s_cropped.tif"), lazy=True)
ims_ = load_image(os.path.join(tmp, "s_cropped.ims"), lazy=True)
check("same shape from both crop formats", tif.data.shape == ims_.data.shape,
      f"{tif.data.shape} vs {ims_.data.shape}")
check("same voxel from both crop formats",
      np.allclose(tif.voxel_size, ims_.voxel_size, rtol=1e-4),
      f"{tif.voxel_size} vs {ims_.voxel_size}")
check("same interval from both crop formats",
      abs(tif.time_interval - ims_.time_interval) < 1e-6,
      f"{tif.time_interval} vs {ims_.time_interval}")

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"), flush=True)
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
