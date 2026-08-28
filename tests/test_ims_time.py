"""Read-only half: the time-interval fix against the real files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, require, sample_path  # noqa: E402

import sys


from utils.load_image import (
    _ims_time_interval_hours,
    ims_acquisition_start,
    load_image,
)

D = None  # resolved below
BIG = require("NUCLOGIC_TEST_IMS")
CROPPED = require("NUCLOGIC_TEST_IMS_CROPPED")

failures = []


def check(label, cond, detail=""):
    if not cond:
        failures.append(f"{label}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {detail if not cond else ''}",
          flush=True)


print("== real files ==", flush=True)
big = _ims_time_interval_hours(BIG)
print(f"  original interval : {big} h   (old code said 2000.0)", flush=True)
check("original is 2 h", abs(big - 2.0) < 0.01, f"got {big}")

start = ims_acquisition_start(BIG)
print(f"  acquisition start : {start}", flush=True)
check("start parsed (has milliseconds)",
      start is not None and start.year == 2025 and start.hour == 14, f"{start}")

cropped = _ims_time_interval_hours(CROPPED)
print(f"  cropped interval  : {cropped} h   (old code said 0.0)", flush=True)
check("broken cropped file -> 1.0 fallback, not 0.0", cropped == 1.0, f"got {cropped}")

r = load_image(BIG, lazy=True)
check("load_image reports 2 h", abs(r.time_interval - 2.0) < 0.01, f"{r.time_interval}")

print("\n" + ("ALL PASS" if not failures else f"{len(failures)} FAILURES:"), flush=True)
for f in failures:
    print("  -", f)
sys.exit(1 if failures else 0)
