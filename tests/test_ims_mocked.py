import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile
import types

import numpy as np
import tifffile

ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}")


REF = np.random.randint(0, 4000, size=(3, 2, 4, 16, 8)).astype(np.uint16)
tmp = tempfile.mkdtemp()

print("\n== TIFF metadata parsing (now living in load_image) ==")
from utils.load_image import _to_tczyx, load_image


def load_tiff_movie_and_metadata(path):
    # tiff_metadata.py is gone; load_image is the single entry point now.
    return load_image(path, lazy=False)


p = os.path.join(tmp, "m.ome.tif")
tifffile.imwrite(p, REF, metadata={"axes": "TCZYX", "PhysicalSizeZ": 2.0,
                                   "PhysicalSizeY": 0.33, "PhysicalSizeX": 0.33,
                                   "TimeIncrement": 30.0, "TimeIncrementUnit": "min"},
                 ome=True)
movie, voxel, interval, missing = load_tiff_movie_and_metadata(p)
check("old API shape", movie.shape == REF.shape, f"got {movie.shape}")
check("old API pixels", np.array_equal(movie, REF))
check("old API voxel", np.allclose(voxel, (2.0, 0.33, 0.33)), f"got {voxel}")
check("old API time", abs(interval - 0.5) < 1e-9, f"got {interval}")
check("old API missing flag", missing is False)

p2 = os.path.join(tmp, "plain.tif")
tifffile.imwrite(p2, REF[0, 0])
movie2, voxel2, _, missing2 = load_tiff_movie_and_metadata(p2)
check("old API plain stack", movie2.shape == (1, 1, 4, 16, 8), f"got {movie2.shape}")
check("old API plain missing", missing2 is True)

print("\n== modules importing _to_tczyx still import ==")
for mod in ("utils.max_project", "utils.crop", "utils.load_image"):
    try:
        __import__(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, repr(e))

print("\n== mocked IMS: lazy path reads only what is indexed ==")


import utils.load_image as li


def install_fake(array, resolution, interval=0.25):
    """Stand in for imaris_ims_file_reader: h5py-like lazy slicing.

    Mirrors the real module, which exposes the ims_reader class alongside the
    ims() convenience function. load_image subclasses the class to silence its
    prints, so a factory function is not enough here.
    """
    holder = {}

    class FakeImsReader:
        def __init__(self, path, **kwargs):
            self._a = array
            self.shape = array.shape
            self.dtype = array.dtype
            self.ndim = array.ndim
            self.resolution = resolution
            self.reads = []
            holder["obj"] = self

        def __getitem__(self, key):
            self.reads.append(key)
            return self._a[key]

        def close(self):
            pass

    pkg = types.ModuleType("imaris_ims_file_reader")
    sub = types.ModuleType("imaris_ims_file_reader.ims")
    sub.ims_reader = FakeImsReader
    sub.ims = FakeImsReader
    pkg.ims = FakeImsReader
    sys.modules["imaris_ims_file_reader"] = pkg
    sys.modules["imaris_ims_file_reader.ims"] = sub

    gt = types.ModuleType("utils.get_time_interval")
    gt.get_time_interval = lambda path: interval
    sys.modules["utils.get_time_interval"] = gt

    # load_image caches the subclass it builds; drop it so this stub is used.
    li._QUIET_IMS_CLASS = None
    return holder

# 5D IMS
holder = install_fake(REF, (2.0, 0.33, 0.33))
fake_path = os.path.join(tmp, "sample.ims")
open(fake_path, "wb").close()

r = li.load_image(fake_path, lazy=True)
check("ims lazy shape", r.data.shape == REF.shape, f"got {r.data.shape}")
check("ims lazy is dask", type(r.data).__module__.startswith("dask"), f"{type(r.data)}")
check("ims voxel", r.voxel_size == (2.0, 0.33, 0.33), f"got {r.voxel_size}")
# Timing is now read from the file itself (load_image._ims_time_interval_hours),
# not from the patched utils.get_time_interval. The stub file is empty, so the
# documented 1.0 hour fallback is the correct result here.
check("ims time interval falls back to 1.0", r.time_interval == 1.0,
      f"got {r.time_interval}")
check("ims nothing read on open", holder["obj"].reads == [],
      f"reads={holder['obj'].reads}")

one = li.as_numpy(r.data[1, 0])
check("ims single (t,c) correct", np.array_equal(one, REF[1, 0]))
n_reads = len(holder["obj"].reads)
check("ims partial read only", 0 < n_reads <= 2, f"{n_reads} reads for 1 of 6 planes")

check("ims full readback", np.array_equal(li.as_numpy(r.data), REF))

# 4D IMS (C,Z,Y,X) -> must gain a leading T axis
holder = install_fake(REF[0], (2.0, 0.33, 0.33))
r = li.load_image(fake_path, lazy=True)
check("ims 4D padded to 5D", r.data.shape == (1, 2, 4, 16, 8), f"got {r.data.shape}")
check("ims 4D pixels", np.array_equal(li.as_numpy(r.data)[0], REF[0]))

# 3D IMS (Z,Y,X)
holder = install_fake(REF[0, 0], (2.0, 0.33, 0.33))
r = li.load_image(fake_path, lazy=True)
check("ims 3D padded to 5D", r.data.shape == (1, 1, 4, 16, 8), f"got {r.data.shape}")

# eager path
holder = install_fake(REF, (2.0, 0.33, 0.33))
r = li.load_image(fake_path, lazy=False)
check("ims eager is numpy", isinstance(r.data, np.ndarray), f"{type(r.data)}")
check("ims eager pixels", np.array_equal(r.data, REF))

# missing resolution -> metadata_missing
holder = install_fake(REF, None)
r = li.load_image(fake_path, lazy=True)
check("ims no resolution -> missing", r.metadata_missing is True)
check("ims no resolution -> default voxel", r.voxel_size == (1.0, 1.0, 1.0))

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
