"""ND2 and CZI coverage for utils/load_image.py.

CZI is tested end-to-end against a real file written with pylibCZIrw.
ND2 has no writer available, so its loader is driven with a stub that mimics
the nd2 package's documented API shapes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile
from collections import namedtuple

import numpy as np

ROOT = PROJECT_ROOT
sys.path.insert(0, ROOT)

import utils.load_image as li
from utils.load_image import as_numpy, load_image

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}")


T, C, Z, Y, X = 3, 2, 4, 16, 8
REF = np.random.randint(0, 4000, size=(T, C, Z, Y, X)).astype(np.uint16)
tmp = tempfile.mkdtemp()

# ── CZI: real file ────────────────────────────────────────────────────────────
print("\n== CZI (real file via pylibCZIrw) ==")
from pylibCZIrw import czi as pyczi

czi_path = os.path.join(tmp, "sample.czi")
with pyczi.create_czi(czi_path) as doc:
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                doc.write(data=REF[t, c, z], plane={"T": t, "C": c, "Z": z})
    # pylibCZIrw documents these as um but writes them into the CZI's metres
    # field verbatim, so pass metres to get a file that says 0.33 um / 2.0 um.
    doc.write_metadata(
        document_name="test", scale_x=0.33e-6, scale_y=0.33e-6, scale_z=2.0e-6
    )

print(f"      wrote {os.path.getsize(czi_path)} bytes")

for lazy in (True, False):
    r = load_image(czi_path, lazy=lazy)
    check(f"czi shape lazy={lazy}", r.data.shape == REF.shape, f"got {r.data.shape}")
    check(f"czi pixels lazy={lazy}", np.array_equal(as_numpy(r.data), REF))
    check(f"czi is dask lazy={lazy}",
          type(r.data).__module__.startswith("dask") == lazy, f"{type(r.data)}")
    print(f"      voxel_size={r.voxel_size}  time_interval={r.time_interval} "
          f"metadata_missing={r.metadata_missing}")
    check(f"czi voxel lazy={lazy}", np.allclose(r.voxel_size, (2.0, 0.33, 0.33)),
          f"got {r.voxel_size} (expected um)")
    check(f"czi md flag lazy={lazy}", r.metadata_missing is False)

print("\n== CZI without scaling metadata ==")
czi_bare = os.path.join(tmp, "bare.czi")
with pyczi.create_czi(czi_bare) as doc:
    for z in range(Z):
        doc.write(data=REF[0, 0, z], plane={"Z": z})
r = load_image(czi_bare)
print(f"      shape={r.data.shape} voxel={r.voxel_size} missing={r.metadata_missing}")
check("bare czi loads 5D", r.data.ndim == 5, f"got {r.data.shape}")
check("bare czi keeps Z", r.data.shape[2] == Z, f"got {r.data.shape}")
check("bare czi pixels", np.array_equal(as_numpy(r.data)[0, 0], REF[0, 0]))

# ── ND2: stubbed reader ───────────────────────────────────────────────────────
print("\n== ND2 (stubbed reader, real code path) ==")
VoxelSize = namedtuple("VoxelSize", "x y z")
Params = namedtuple("Params", "periodMs")
Loop = namedtuple("Loop", "type parameters")


class StubND2File:
    def __init__(self, sizes, voxel, loops):
        self.sizes = sizes
        self._voxel = voxel
        self.experiment = loops

    def voxel_size(self, channel=0):
        return self._voxel

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def install_nd2_stub(array, sizes, voxel=VoxelSize(0.33, 0.33, 2.0),
                     loops=(Loop("TimeLoop", Params(1800000.0)),)):
    import nd2

    nd2.ND2File = lambda path: StubND2File(sizes, voxel, list(loops))
    nd2.imread = lambda path, dask=False: (
        __import__("dask.array", fromlist=["array"]).from_array(array)
        if dask else array
    )


import nd2  # noqa: E402  (real package, patched below)

real_nd2file, real_imread = nd2.ND2File, nd2.imread
nd2_path = os.path.join(tmp, "sample.nd2")
open(nd2_path, "wb").close()

try:
    print("\n  -- canonical TCZYX order --")
    install_nd2_stub(REF, {"T": T, "C": C, "Z": Z, "Y": Y, "X": X})
    for lazy in (True, False):
        r = load_image(nd2_path, lazy=lazy)
        check(f"nd2 shape lazy={lazy}", r.data.shape == REF.shape, f"got {r.data.shape}")
        check(f"nd2 pixels lazy={lazy}", np.array_equal(as_numpy(r.data), REF))
        check(f"nd2 voxel (x,y,z)->(z,y,x) lazy={lazy}",
              r.voxel_size == (2.0, 0.33, 0.33), f"got {r.voxel_size}")
        check(f"nd2 time 1800000ms->0.5h lazy={lazy}",
              abs(r.time_interval - 0.5) < 1e-9, f"got {r.time_interval}")
        check(f"nd2 md flag lazy={lazy}", r.metadata_missing is False)

    print("\n  -- TZCYX order must be transposed --")
    tzcyx = np.transpose(REF, (0, 2, 1, 3, 4))
    install_nd2_stub(tzcyx, {"T": T, "Z": Z, "C": C, "Y": Y, "X": X})
    r = load_image(nd2_path)
    check("nd2 reordered to TCZYX", r.data.shape == REF.shape, f"got {r.data.shape}")
    check("nd2 reordered pixels", np.array_equal(as_numpy(r.data), REF))

    print("\n  -- ZYX only --")
    install_nd2_stub(REF[0, 0], {"Z": Z, "Y": Y, "X": X})
    r = load_image(nd2_path)
    check("nd2 3D padded", r.data.shape == (1, 1, Z, Y, X), f"got {r.data.shape}")
    check("nd2 3D pixels", np.array_equal(as_numpy(r.data)[0, 0], REF[0, 0]))

    print("\n  -- multi-position: first position, with warning --")
    positions = np.stack([REF, REF + 1])
    install_nd2_stub(positions, {"P": 2, "T": T, "C": C, "Z": Z, "Y": Y, "X": X})
    r = load_image(nd2_path)
    check("nd2 multi-position shape", r.data.shape == REF.shape, f"got {r.data.shape}")
    check("nd2 took first position", np.array_equal(as_numpy(r.data), REF))

    print("\n  -- missing voxel metadata --")
    install_nd2_stub(REF, {"T": T, "C": C, "Z": Z, "Y": Y, "X": X},
                     voxel=VoxelSize(0.0, 0.0, 0.0))
    r = load_image(nd2_path)
    check("nd2 zero voxel -> missing", r.metadata_missing is True)
    check("nd2 zero voxel -> default", r.voxel_size == (1.0, 1.0, 1.0), f"{r.voxel_size}")

    print("\n  -- no time loop --")
    install_nd2_stub(REF, {"T": T, "C": C, "Z": Z, "Y": Y, "X": X}, loops=())
    r = load_image(nd2_path)
    check("nd2 no timeloop -> 1.0h", r.time_interval == 1.0, f"got {r.time_interval}")
finally:
    nd2.ND2File, nd2.imread = real_nd2file, real_imread

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
