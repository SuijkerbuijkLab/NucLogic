import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile

import numpy as np
import tifffile


from utils.load_image import as_numpy, extension_of, is_supported, load_image

tmp = tempfile.mkdtemp()
failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if not condition:
        failures.append(f"{label}: {detail}")
    print(f"  [{status}] {label} {detail if not condition else ''}")


# reference data: T=3, C=2, Z=4, Y=16, X=8 (Y != X to catch transposes)
REF = np.random.randint(0, 4000, size=(3, 2, 4, 16, 8)).astype(np.uint16)

print("\n== OME-TIFF ==")
p = os.path.join(tmp, "movie.ome.tif")
tifffile.imwrite(
    p, REF, metadata={"axes": "TCZYX", "PhysicalSizeZ": 2.0, "PhysicalSizeY": 0.33,
                      "PhysicalSizeX": 0.33, "TimeIncrement": 30.0,
                      "TimeIncrementUnit": "min"}, ome=True,
)
for lazy in (True, False):
    r = load_image(p, lazy=lazy)
    check(f"ome shape lazy={lazy}", r.data.shape == REF.shape, f"got {r.data.shape}")
    check(f"ome pixels lazy={lazy}", np.array_equal(as_numpy(r.data), REF))
    check(f"ome voxel lazy={lazy}", np.allclose(r.voxel_size, (2.0, 0.33, 0.33)),
          f"got {r.voxel_size}")
    check(f"ome time lazy={lazy}", abs(r.time_interval - 0.5) < 1e-9,
          f"got {r.time_interval}")
    check(f"ome md_missing lazy={lazy}", r.metadata_missing is False)
    check(f"ome lazy is dask lazy={lazy}",
          (type(r.data).__module__.startswith("dask")) == lazy,
          f"got {type(r.data)}")

print("\n== tuple unpacking backward compat ==")
movie, voxel, interval, missing = load_image(p, lazy=False)
check("unpacks as 4-tuple", movie.shape == REF.shape and len(voxel) == 3)

print("\n== ImageJ TIFF (ZYX only) ==")
p = os.path.join(tmp, "stack.tif")
zyx = REF[0, 0]
tifffile.imwrite(p, zyx, imagej=True, resolution=(1 / 0.25, 1 / 0.25),
                 metadata={"spacing": 1.5, "unit": "um", "axes": "ZYX",
                           "finterval": 3600.0, "tunit": "s"})
for lazy in (True, False):
    r = load_image(p, lazy=lazy)
    check(f"ij shape lazy={lazy}", r.data.shape == (1, 1, 4, 16, 8), f"got {r.data.shape}")
    check(f"ij pixels lazy={lazy}", np.array_equal(as_numpy(r.data)[0, 0], zyx))
    check(f"ij voxel lazy={lazy}", np.allclose(r.voxel_size, (1.5, 0.25, 0.25)),
          f"got {r.voxel_size}")
    check(f"ij time lazy={lazy}", abs(r.time_interval - 1.0) < 1e-9, f"got {r.time_interval}")

print("\n== plain TIFF (no metadata) ==")
p = os.path.join(tmp, "plain.tif")
tifffile.imwrite(p, zyx)
r = load_image(p)
check("plain shape", r.data.shape == (1, 1, 4, 16, 8), f"got {r.data.shape}")
check("plain md_missing", r.metadata_missing is True)
check("plain voxel default", r.voxel_size == (1.0, 1.0, 1.0), f"got {r.voxel_size}")

print("\n== compressed TIFF (memmap would fail, aszarr must work) ==")
p = os.path.join(tmp, "compressed.ome.tif")
tifffile.imwrite(p, REF, metadata={"axes": "TCZYX"}, ome=True, compression="zlib")
r = load_image(p, lazy=True)
check("compressed lazy pixels", np.array_equal(as_numpy(r.data), REF))

print("\n== lazy really is lazy (partial read) ==")
p = os.path.join(tmp, "movie.ome.tif")
r = load_image(p, lazy=True)
frame = as_numpy(r.data[1])
check("single timepoint slice", np.array_equal(frame, REF[1]), f"got {frame.shape}")

print("\n== OME-Zarr ==")
import zarr

p = os.path.join(tmp, "image.ome.zarr")
root = zarr.open_group(p, mode="w")
root.create_array("0", shape=REF.shape, chunks=(1, 1, 4, 16, 8), dtype=REF.dtype)
root["0"][...] = REF
root.attrs["multiscales"] = [{
    "version": "0.4",
    "axes": [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ],
    "datasets": [{"path": "0", "coordinateTransformations": [
        {"type": "scale", "scale": [1800.0, 1.0, 2.0, 0.33, 0.33]}]}],
}]
for lazy in (True, False):
    r = load_image(p, lazy=lazy)
    check(f"zarr shape lazy={lazy}", r.data.shape == REF.shape, f"got {r.data.shape}")
    check(f"zarr pixels lazy={lazy}", np.array_equal(as_numpy(r.data), REF))
    check(f"zarr voxel lazy={lazy}", np.allclose(r.voxel_size, (2.0, 0.33, 0.33)),
          f"got {r.voxel_size}")
    check(f"zarr time lazy={lazy}", abs(r.time_interval - 0.5) < 1e-9,
          f"got {r.time_interval}")

print("\n== OME-Zarr in nanometres (unit conversion) ==")
p2 = os.path.join(tmp, "nm.ome.zarr")
root = zarr.open_group(p2, mode="w")
root.create_array("0", shape=REF.shape, chunks=(1, 1, 4, 16, 8), dtype=REF.dtype)
root["0"][...] = REF
root.attrs["multiscales"] = [{
    "version": "0.4",
    "axes": [{"name": n, "type": t, **({"unit": u} if u else {})} for n, t, u in
             [("t", "time", "second"), ("c", "channel", None), ("z", "space", "nanometer"),
              ("y", "space", "nanometer"), ("x", "space", "nanometer")]],
    "datasets": [{"path": "0", "coordinateTransformations": [
        {"type": "scale", "scale": [600.0, 1.0, 2000.0, 330.0, 330.0]}]}],
}]
r = load_image(p2)
check("zarr nm->um", np.allclose(r.voxel_size, (2.0, 0.33, 0.33)), f"got {r.voxel_size}")

print("\n== error handling ==")
try:
    load_image(os.path.join(tmp, "nope.tif"))
    check("missing file raises", False)
except FileNotFoundError:
    check("missing file raises", True)

p3 = os.path.join(tmp, "thing.png")  # .lif used to live here, before we read it
open(p3, "wb").close()
try:
    load_image(p3)
    check("unsupported raises", False)
except ValueError as e:
    check("unsupported raises", "Unsupported image format" in str(e))

p4 = os.path.join(tmp, "sample.nd2")
open(p4, "wb").close()
try:
    load_image(p4)
    check("corrupt nd2 raises", False, "no error raised")
except FileNotFoundError:
    check("corrupt nd2 raises", False, "wrong error type")
except Exception as e:
    # nd2 is installed now, so this surfaces the reader's own error rather than
    # the ImportError guard.
    check("corrupt nd2 raises", True)
    print(f"      {type(e).__name__}: {str(e)[:60]}")

print("\n== extension helpers ==")
check("ome.tif beats tif", extension_of("a.ome.tif") == ".ome.tif")
check("ome.zarr beats zarr", extension_of("a.ome.zarr") == ".ome.zarr")
check("zarr dir trailing slash", extension_of("a.zarr/") == ".zarr")
check("uppercase", is_supported("A.TIF") is True)
check("lif is supported", is_supported("a.lif") is True)
check("unsupported", is_supported("a.png") is False)

print("\n== implausible voxel size warning ==")
p5 = os.path.join(tmp, "metres.ome.tif")
tifffile.imwrite(p5, REF, metadata={"axes": "TCZYX", "PhysicalSizeZ": 0.000002,
                                    "PhysicalSizeY": 3.3e-7, "PhysicalSizeX": 3.3e-7},
                 ome=True)
r = load_image(p5)
check("still loads with odd voxel size", r.data.shape == REF.shape)

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"))
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
