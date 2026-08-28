"""save_as_ims -> load_image round trip for timing metadata."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import os
import sys
import tempfile
from datetime import datetime

import numpy as np


from utils.load_image import ims_acquisition_start, load_image
from utils.save_as_ims import save_as_ims

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}",
          flush=True)


tmp = tempfile.mkdtemp()
movie = np.random.randint(0, 500, size=(4, 2, 3, 32, 32)).astype(np.uint16)
origin = datetime(2025, 11, 21, 14, 47, 58)

print("== writing 4x2x3x32x32 test file ==", flush=True)
out = os.path.join(tmp, "roundtrip.ims")
save_as_ims(input_movie=movie, output_filename=out,
            voxel_size=(2.479, 0.642, 0.642), time_interval=2.0, start_time=origin)
print("  written", flush=True)

back = load_image(out, lazy=False)
print(f"  shape        : {back.data.shape}", flush=True)
print(f"  voxel_size   : {back.voxel_size}", flush=True)
print(f"  time_interval: {back.time_interval} h", flush=True)
print(f"  start        : {ims_acquisition_start(out)}", flush=True)

check("round trip shape", back.data.shape == movie.shape, f"{back.data.shape}")
check("round trip pixels", np.array_equal(back.data, movie))
check("round trip voxel", np.allclose(back.voxel_size, (2.479, 0.642, 0.642)),
      f"{back.voxel_size}")
check("round trip interval is 2 h", abs(back.time_interval - 2.0) < 0.01,
      f"got {back.time_interval}")
check("start time preserved", ims_acquisition_start(out) == origin,
      f"got {ims_acquisition_start(out)}")

print("\n== fractional interval (10 min) ==", flush=True)
out2 = os.path.join(tmp, "tenmin.ims")
save_as_ims(input_movie=movie, output_filename=out2, voxel_size=(1.0, 0.5, 0.5),
            time_interval=1 / 6, start_time=origin)
r2 = load_image(out2, lazy=True)
print(f"  interval: {r2.time_interval} h ({r2.time_interval * 60:.1f} min)", flush=True)
check("10 min round trip", abs(r2.time_interval - 1 / 6) < 0.02, f"{r2.time_interval}")

print("\n== defensive: unusable interval must not crash the writer ==", flush=True)
out3 = os.path.join(tmp, "bad.ims")
save_as_ims(input_movie=movie, output_filename=out3, voxel_size=(1.0, 0.5, 0.5),
            time_interval=0.0)
r3 = load_image(out3, lazy=True)
check("zero interval -> 1 h", abs(r3.time_interval - 1.0) < 0.01, f"{r3.time_interval}")

out4 = os.path.join(tmp, "none.ims")
save_as_ims(input_movie=movie, output_filename=out4, voxel_size=(1.0, 0.5, 0.5),
            time_interval=None)
check("None interval still writes", os.path.exists(out4))

print("\n== no start_time: spacing still correct ==", flush=True)
out5 = os.path.join(tmp, "nostart.ims")
save_as_ims(input_movie=movie, output_filename=out5, voxel_size=(1.0, 0.5, 0.5),
            time_interval=3.0)
r5 = load_image(out5, lazy=True)
check("spacing kept without start_time", abs(r5.time_interval - 3.0) < 0.01,
      f"{r5.time_interval}")

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"), flush=True)
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
