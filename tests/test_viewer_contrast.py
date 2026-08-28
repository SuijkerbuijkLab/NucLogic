"""Contrast limits for napari layers.

Reproduces the saturated-image bug (napari auto-scaling a lazy array from one
plane) and checks the limits we compute instead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, Results, sample_path  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "PySide6"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402

import app as nuclogic  # noqa: E402

r = Results()


class Headless(nuclogic.MainWindow):
    def _start_update_check(self):
        pass

    def _start_prewarm(self):
        pass


from PySide6.QtWidgets import QApplication  # noqa: E402

qapp = QApplication.instance() or QApplication([])
window = Headless()

# A realistic organoid stack: dim background, bright nuclei in the middle Z
# planes, empty planes top and bottom (the case that broke auto-scaling).
rng = np.random.default_rng(0)
T, Z, Y, X = 4, 20, 64, 64
movie = rng.integers(90, 120, (T, Z, Y, X)).astype(np.uint16)
movie[:, :3] = 0          # empty planes at the start
movie[:, -3:] = 0         # and the end
movie[:, 8:12, 20:44, 20:44] = rng.integers(2500, 4000, (T, 4, 24, 24))

print("== the bug: scaling from a single plane ==")
first_plane = movie[0, 0]
print(f"  first plane min/max: {first_plane.min()}/{first_plane.max()}", flush=True)
r.check("first plane is empty (would collapse the range)", first_plane.max() == 0)

print("\n== our limits, computed from a sampled timepoint ==")
for label, stack in (("numpy", movie), ("dask", da.from_array(movie, chunks=(1, Z, Y, X)))):
    limits, value_range = window._contrast_from_sample(stack)
    print(f"  {label:5s} limits={limits} range={value_range}", flush=True)

    r.check(f"{label}: slider starts at 0", value_range[0] == 0.0, f"{value_range}")
    r.check(f"{label}: slider ends at data max",
            abs(value_range[1] - movie.max()) < 1, f"{value_range[1]} vs {movie.max()}")
    r.check(f"{label}: black point near background",
            90 <= limits[0] <= 130, f"{limits[0]}")
    r.check(f"{label}: white point near foreground",
            limits[1] > 2000, f"{limits[1]}")
    r.check(f"{label}: not saturated (white point well above black)",
            limits[1] > limits[0] * 5, f"{limits}")
    r.check(f"{label}: limits inside slider range",
            value_range[0] <= limits[0] < limits[1] <= value_range[1], f"{limits}")

print("\n== degenerate inputs ==")
blank = np.zeros((2, 4, 8, 8), dtype=np.uint16)
limits, value_range = window._contrast_from_sample(blank)
r.check("all-zero stack does not crash", limits[1] > limits[0], f"{limits}")

flat = np.full((2, 4, 8, 8), 500, dtype=np.uint16)
limits, value_range = window._contrast_from_sample(flat)
r.check("uniform stack gives a usable range", limits[1] > limits[0], f"{limits}")
r.check("uniform stack slider still reaches the value", value_range[1] >= 500,
        f"{value_range}")

print("\n== napari applies them ==")
# The Image layer is built directly: napari.Viewer needs an OpenGL canvas,
# which the offscreen Qt platform used for testing cannot provide.
from napari.layers import Image  # noqa: E402

limits, value_range = window._contrast_from_sample(movie)
layer = Image(movie, contrast_limits=limits)
layer.contrast_limits_range = value_range
layer.contrast_limits = limits
r.check("layer keeps our contrast limits",
        np.allclose(layer.contrast_limits, limits), f"{layer.contrast_limits}")
r.check("layer keeps our slider range",
        np.allclose(layer.contrast_limits_range, value_range),
        f"{layer.contrast_limits_range}")

# What napari does on its own with the lazy array: the bug being fixed.
auto = Image(da.from_array(movie, chunks=(1, Z, Y, X)))
print(f"  napari's own limits on the lazy array: {auto.contrast_limits}", flush=True)
r.check("our white point beats napari's auto-scaling",
        limits[1] > auto.contrast_limits[1],
        f"ours {limits} vs napari {auto.contrast_limits}")

print("\n== real IMS ==")
BIG = sample_path("NUCLOGIC_TEST_IMS")
if BIG:
    from utils.load_image import load_image

    real, _, _, _ = load_image(BIG)
    limits, value_range = window._contrast_from_sample(real[:, 0])
    print(f"  channel 0: limits={limits} range={value_range}", flush=True)
    r.check("real: slider starts at 0", value_range[0] == 0.0)
    r.check("real: white point above black point", limits[1] > limits[0], f"{limits}")
    r.check("real: limits within slider range", limits[1] <= value_range[1],
            f"{limits} vs {value_range}")
else:
    print("  (no sample IMS configured, skipped)", flush=True)

r.finish()
