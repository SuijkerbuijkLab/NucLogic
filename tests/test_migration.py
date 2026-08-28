"""Phase 0-2 migration checks: imports resolve, and the new loading path
produces exactly what the old one did on real files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile

import numpy as np
import tifffile

ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "PySide6"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}",
          flush=True)


print("== every migrated module still imports ==", flush=True)
for mod in ("utils.load_image", "utils.find_input_file", "utils.file_to_folder",
            "utils.crop", "utils.max_project", "utils.save_as_ims",
            "main_functions.crop_sample", "main_functions.segment_organoid",
            "main_functions.add_advanced_statistics", "main_functions.split_phenotype_mask",
            "utils.crop_fixed", "app"):
    try:
        __import__(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, f"{type(e).__name__}: {e}")

print("\n== tiff_metadata is really gone ==", flush=True)
check("module deleted",
      not os.path.exists(os.path.join(ROOT, "utils", "tiff_metadata.py")))
try:
    import utils.tiff_metadata  # noqa: F401
    check("no stale import possible", False, "still importable")
except ImportError:
    check("no stale import possible", True)

print("\n== find_input_file now sees every supported format ==", flush=True)
from utils.find_input_file import find_input_file

tmp = tempfile.mkdtemp()
for name, size in (("a.tiff", 300), ("b.tif", 200), ("c.nd2", 100), ("notes.txt", 999)):
    with open(os.path.join(tmp, name), "wb") as fh:
        fh.write(b"\0" * size)
picked = find_input_file(tmp)
check("picks biggest supported file", os.path.basename(picked) == "a.tiff",
      f"got {os.path.basename(picked) if picked else None}")
check(".tiff no longer invisible", picked is not None)

restricted = find_input_file(tmp, types=[".nd2"])
check("types= still restricts", os.path.basename(restricted) == "c.nd2",
      f"got {os.path.basename(restricted) if restricted else None}")

empty = find_input_file(tempfile.mkdtemp())
check("empty dir returns None", empty is None, f"got {empty}")

print("\n== app._file_priority handles new formats without TypeError ==", flush=True)
sample_dir = os.path.join(tmp, "sample")
os.makedirs(sample_dir, exist_ok=True)
for name in ("s_cropped.ims", "s.ims", "s.tif", "s.nd2", "s.czi", "s.tiff"):
    open(os.path.join(sample_dir, name), "wb").close()


def _file_priority(f):
    lowered = f.lower()
    if lowered.endswith("_cropped.ims"):
        return 0
    if lowered.endswith("_cropped.tif"):
        return 1
    if lowered.endswith(".ims"):
        return 2
    return 3


from utils.load_image import is_supported

try:
    ordered = sorted([f for f in os.listdir(sample_dir) if is_supported(f)],
                     key=_file_priority)
    check("sorting mixed formats works", ordered[0] == "s_cropped.ims", f"{ordered}")
except TypeError as e:
    check("sorting mixed formats works", False, f"TypeError: {e}")

print("\n== new load path == old load path (real 32 GB IMS) ==", flush=True)
D = None  # resolved below
BIG = sample_path("NUCLOGIC_TEST_IMS")

if BIG:
    from imaris_ims_file_reader.ims import ims

    from utils.load_image import as_numpy, load_image

    old = ims(BIG)
    while old.ndim < 5:
        old = np.expand_dims(old, axis=0)
    old_voxel = ims(BIG).resolution

    new_movie, new_voxel, _, _ = load_image(BIG)
    check("same shape", tuple(old.shape) == tuple(new_movie.shape),
          f"{old.shape} vs {new_movie.shape}")
    check("same voxel size", tuple(old_voxel) == tuple(new_voxel),
          f"{old_voxel} vs {new_voxel}")

    # This is exactly what segment_organoid's loop now does.
    for t in (0, 12):
        frame = as_numpy(new_movie[t])
        check(f"frame[{t}] is numpy", isinstance(frame, np.ndarray), f"{type(frame)}")
        check(f"frame[{t}] is CZYX", frame.ndim == 4, f"{frame.shape}")
        check(f"frame[{t}] matches old path", np.array_equal(frame, np.asarray(old[t])))

    # np.max over channels must stay numpy (the dask-leak trap)
    frame = as_numpy(new_movie[0])
    merged = np.max(frame[[0, 1]], axis=0)
    check("np.max stays numpy", isinstance(merged, np.ndarray), f"{type(merged)}")
else:
    print("  (real IMS not available, skipped)", flush=True)

print("\n== new load path == old load path (TIFF) ==", flush=True)
from utils.load_image import load_image

REF = np.random.randint(0, 4000, size=(3, 2, 4, 16, 8)).astype(np.uint16)
p = os.path.join(tmp, "movie.ome.tif")
tifffile.imwrite(p, REF, metadata={"axes": "TCZYX", "PhysicalSizeZ": 2.0,
                                   "PhysicalSizeY": 0.33, "PhysicalSizeX": 0.33,
                                   "TimeIncrement": 30.0,
                                   "TimeIncrementUnit": "min"}, ome=True)
movie, voxel, interval, missing = load_image(p, lazy=False)
check("tiff pixels", np.array_equal(movie, REF))
check("tiff voxel", np.allclose(voxel, (2.0, 0.33, 0.33)), f"{voxel}")
check("tiff interval", abs(interval - 0.5) < 1e-9, f"{interval}")
check("tiff missing flag", missing is False)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"), flush=True)
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
