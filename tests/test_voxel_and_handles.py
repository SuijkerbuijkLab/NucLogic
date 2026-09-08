"""Per-axis voxel size override, and not holding source files open.

Two fixes that travel together: an override that only applied when the file had
no calibration at all (and turned blank axes into 1.0), and readers that locked
the file they had been asked about.
"""

import gc
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import PROJECT_ROOT, Results, sample_path  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "PySide6"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from utils.load_image import as_numpy, load_image  # noqa: E402
from utils.voxel_size import parse_override, resolve_voxel_size  # noqa: E402

r = Results()
tmp = tempfile.mkdtemp()

FILE_VOXEL = (2.0, 0.33, 0.33)

print("== parsing the three boxes ==")
r.check("all blank means no override", parse_override("", "", "") == (None, None, None))
r.check("one filled leaves the others alone",
        parse_override("5", "", "") == (5.0, None, None),
        f"{parse_override('5', '', '')}")
r.check("whitespace counts as blank", parse_override("  ", "0.1", " ") == (None, 0.1, None),
        f"{parse_override('  ', '0.1', ' ')}")
for bad, why in (("abc", "not a number"), ("0", "zero"), ("-1", "negative")):
    try:
        parse_override(bad, "", "")
        r.check(f"{why} rejected", False, "no error raised")
    except ValueError as error:
        r.check(f"{why} rejected", "Z" in str(error), str(error)[:60])

print("\n== each axis resolves independently ==")
cases = {
    "nothing filled keeps the file": ((None, None, None), FILE_VOXEL),
    "Z only overrides Z": ((5.0, None, None), (5.0, 0.33, 0.33)),
    "XY only keeps the measured Z": ((None, 0.1, 0.1), (2.0, 0.1, 0.1)),
    "all three win": ((5.0, 0.1, 0.2), (5.0, 0.1, 0.2)),
}
for label, (override, expected) in cases.items():
    got = resolve_voxel_size(FILE_VOXEL, override)
    r.check(label, got == expected, f"got {got}, expected {expected}")

print("\n== the case that used to corrupt sizes ==")
# A merged MetaMorph sample: real Z from the stack, no XY calibration. Filling
# in only XY used to replace the whole tuple and overwrite Z with 1.0.
merged = (0.245, 1.0, 1.0)
got = resolve_voxel_size(merged, (None, 0.1, 0.1), metadata_missing=True)
r.check("measured Z survives an XY-only override", abs(got[0] - 0.245) < 1e-9,
        f"{got}")
r.check("XY takes the typed value", got[1:] == (0.1, 0.1), f"{got}")

# And the override now applies even when the file did have calibration.
got = resolve_voxel_size(FILE_VOXEL, (5.0, None, None), metadata_missing=False)
r.check("an override is no longer ignored on a calibrated file", got[0] == 5.0,
        f"{got}")

print("\n== bad input never propagates as a crash ==")
r.check("None override", resolve_voxel_size(FILE_VOXEL, None) == FILE_VOXEL)
r.check("short tuple", resolve_voxel_size(FILE_VOXEL, (1.0,)) == FILE_VOXEL,
        f"{resolve_voxel_size(FILE_VOXEL, (1.0,))}")
r.check("junk measured falls back to 1.0",
        resolve_voxel_size("nonsense", None) == (1.0, 1.0, 1.0),
        f"{resolve_voxel_size('nonsense', None)}")

print("\n== the UI produces a per-axis override ==")
import app as nuclogic  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

qapp = QApplication.instance() or QApplication([])


class Headless(nuclogic.MainWindow):
    def _start_update_check(self):
        pass

    def _start_prewarm(self):
        pass


window = Headless()
window.voxel_size_z_input.setText("5")
window.voxel_size_x_input.setText("")
window.voxel_size_y_input.setText("")
settings = window.get_advanced_statistics_settings()
r.check("Z-only override reaches the pipeline as (5, None, None)",
        settings[5] == (5.0, None, None), f"{settings[5]}")

window.voxel_size_z_input.setText("")
r.check("all blank reaches the pipeline as no override",
        window.get_advanced_statistics_settings()[5] == (None, None, None))

window.voxel_size_z_input.setText("oops")
try:
    window.get_advanced_statistics_settings()
    r.check("bad input still raises for the caller to report", False, "no error")
except ValueError:
    r.check("bad input still raises for the caller to report", True)
window.voxel_size_z_input.setText("")


def locked(path):
    alternative = path + ".probe"
    try:
        os.rename(path, alternative)
        os.rename(alternative, path)
        return False
    except OSError:
        return True


print("\n== reading metadata must not lock the file ==")
movie = np.random.default_rng(0).integers(0, 4000, (2, 2, 3, 8, 6)).astype(np.uint16)
tif_path = os.path.join(tmp, "movie.ome.tif")
tifffile.imwrite(tif_path, movie, ome=True, metadata={"axes": "TCZYX"})

loaded = load_image(tif_path)
r.check("tiff: metadata read leaves the file free", not locked(tif_path))
del loaded
gc.collect()

ims_path = sample_path("NUCLOGIC_TEST_IMS_CROPPED") or sample_path("NUCLOGIC_TEST_IMS")
if ims_path:
    copied = os.path.join(tmp, os.path.basename(ims_path))
    shutil.copy2(ims_path, copied)

    loaded = load_image(copied)
    r.check("ims: metadata read leaves the file free", not locked(copied),
            "the reader used to stay open for the life of the array")
    r.check("ims: shape still correct", len(loaded.data.shape) == 5,
            f"{loaded.data.shape}")

    plane = as_numpy(loaded.data[0, 0])
    r.check("ims: pixels still readable after the deferred open", plane.size > 0,
            f"{plane.shape}")

    del loaded, plane
    gc.collect()
    r.check("ims: released once the array is dropped", not locked(copied))
else:
    print("  (no sample IMS configured, skipped)", flush=True)

print("\n== file_to_folder is reached with handles released ==")
# _create_dirs collects first; this checks the collect actually frees a file
# whose array is unreferenced but not yet collected.
loose = os.path.join(tmp, "loose")
os.makedirs(loose)
loose_file = os.path.join(loose, "sample.ome.tif")
tifffile.imwrite(loose_file, movie, ome=True, metadata={"axes": "TCZYX"})

held = load_image(loose_file)
_ = as_numpy(held.data[0, 0])
r.check("a live, read array does lock the file", locked(loose_file))
del held, _
gc.collect()
r.check("collecting frees it again", not locked(loose_file))

from utils.file_to_folder import file_to_folder  # noqa: E402

failures = file_to_folder(loose)
r.check("the sample folder gets made", os.path.exists(
    os.path.join(loose, "sample", "sample.ome.tif")), f"{failures}")

r.finish()
