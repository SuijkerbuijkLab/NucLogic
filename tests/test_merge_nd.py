"""Merging MetaMorph .nd acquisitions into one OME-TIFF per sample.

Synthetic .nd/.STK fixtures cover the parsing and the failure modes; the real
acquisition, when configured, checks the pixels actually survive.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Results, sample_path  # noqa: E402

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from utils.load_image import load_image  # noqa: E402
from utils.merge_nd import (  # noqa: E402
    describe, merge_file, parse_nd, scan_directory, time_interval_of,
    voxel_size_of,
)

r = Results()
tmp = tempfile.mkdtemp()

ND_TEMPLATE = """"NDInfoFile", Version 1.0
"Description", test
"StartTime1", 20251216 17:42:13
"DoTimelapse", {do_time}
"NTimePoints", {timepoints}
"DoStage", {do_stage}
"NStagePositions", {positions}
"DoWave", TRUE
"NWavelengths", {channels}
{wave_names}"DoZSeries", TRUE
"NZSteps", {z}
"ZStepSize", 0
"WaveInFileName", TRUE
"NEvents", 0
"EndFile"
"""


def make_acquisition(directory, stem, channels=2, z=4, timepoints=1,
                     positions=1, y=8, x=6, write_stacks=True):
    """Write a .nd plus one STK per wavelength, and return the pixel data."""
    os.makedirs(directory, exist_ok=True)
    names = [f"CSU{500 + i}" for i in range(channels)]
    nd_text = ND_TEMPLATE.format(
        do_time="TRUE" if timepoints > 1 else "FALSE",
        timepoints=timepoints,
        do_stage="TRUE" if positions > 1 else "FALSE",
        positions=positions,
        channels=channels,
        wave_names="".join(
            f'"WaveName{i + 1}", "{name}"\n"WaveDoZ{i + 1}", TRUE\n'
            for i, name in enumerate(names)
        ),
        z=z,
    )
    nd_path = os.path.join(directory, stem + ".nd")
    with open(nd_path, "w", encoding="latin-1") as handle:
        handle.write(nd_text)

    data = {}
    if write_stacks:
        for index, name in enumerate(names):
            for timepoint in range(timepoints):
                stack = (np.random.default_rng(index * 10 + timepoint)
                         .integers(0, 4000, (z, y, x)).astype(np.uint16))
                suffix = f"_t{timepoint + 1}" if timepoints > 1 else ""
                path = os.path.join(
                    directory, f"{stem}_w{index + 1}{name}{suffix}.STK"
                )
                tifffile.imwrite(path, stack)
                data[(index, timepoint)] = stack
    return nd_path, data


print("== parsing the .nd index ==")
simple = os.path.join(tmp, "simple")
nd_path, pixels = make_acquisition(simple, "acq")
info = parse_nd(nd_path)
r.check("booleans coerced", info["DoWave"] is True and info["DoTimelapse"] is False,
        f"{info['DoWave']}, {info['DoTimelapse']}")
r.check("integers coerced", info["NZSteps"] == 4 and info["NWavelengths"] == 2,
        f"{info['NZSteps']}, {info['NWavelengths']}")
r.check("quoted strings unwrapped", info["WaveName1"] == "CSU500",
        f"{info['WaveName1']}")
r.check("stops at EndFile", "EndFile" not in info)

print("\n== describing an acquisition ==")
entry = describe(nd_path)
r.check("shape is T,C,Z,Y,X", entry.shape == (1, 2, 4, 8, 6), f"{entry.shape}")
r.check("channels named from the nd", entry.channel_names == ["CSU500", "CSU501"],
        f"{entry.channel_names}")
r.check("sources list the nd and both stacks", len(entry.sources) == 3,
        f"{entry.sources}")
r.check("not merged yet", entry.already_merged is False)

print("\n== merging ==")
folder = merge_file(nd_path)
written = os.path.join(folder, os.path.basename(folder) + ".ome.tif")
r.check("sample folder is named after the nd", os.path.basename(folder) == "acq",
        os.path.basename(folder))
data, voxel, interval, missing = load_image(written, lazy=False)
r.check("merged shape", data.shape == (1, 2, 4, 8, 6), f"{data.shape}")
for channel in range(2):
    r.check(f"channel {channel} pixels intact",
            np.array_equal(data[0, channel], pixels[(channel, 0)]))
r.check("no .part left", not [f for f in os.listdir(folder) if f.endswith(".part")])
with tifffile.TiffFile(written) as tif:
    r.check("written as OME BigTIFF", tif.is_ome and tif.is_bigtiff,
            f"ome={tif.is_ome} bigtiff={tif.is_bigtiff}")

print("\n== uncalibrated XY must be reported as missing ==")
# The fixture stacks carry no MetaMorph calibration, which is exactly the case
# the real data has: writing 1.0 um as fact would skip the voxel size override.
(z_um, y_um, x_um), xy_missing = voxel_size_of(entry)
r.check("xy reported missing", xy_missing is True)
r.check("xy left unset rather than invented", y_um is None and x_um is None,
        f"{y_um}, {x_um}")
r.check("merged file reports metadata_missing", missing is True)

print("\n== rerunning is a no-op ==")
again = merge_file(nd_path)
r.check("same folder returned", again == folder, f"{again}")
r.check("already_merged now true", describe(nd_path).already_merged is True)
r.check("scan reports it as merged",
        [e.already_merged for e in scan_directory(simple)] == [True])

print("\n== cancelling leaves nothing behind ==")
cancel_dir = os.path.join(tmp, "cancel")
cancel_nd, _ = make_acquisition(cancel_dir, "stopme")
result = merge_file(cancel_nd, should_stop=lambda: True)
r.check("returns None when cancelled", result is None, f"{result}")
folder_path = os.path.join(cancel_dir, "stopme")
leftover = os.listdir(folder_path) if os.path.isdir(folder_path) else []
r.check("no partial file kept", not leftover, f"{leftover}")
r.check("still reported as unmerged",
        describe(cancel_nd).already_merged is False)

print("\n== failure modes are explicit ==")
missing_dir = os.path.join(tmp, "missing")
missing_nd, _ = make_acquisition(missing_dir, "gone", write_stacks=False)
try:
    describe(missing_nd)
    r.check("missing stacks raise", False, "no error raised")
except FileNotFoundError as error:
    r.check("missing stacks raise", "gone_w1CSU500" in str(error), str(error)[:70])

partial_dir = os.path.join(tmp, "partial")
partial_nd, _ = make_acquisition(partial_dir, "half")
os.remove(os.path.join(partial_dir, "half_w2CSU501.STK"))
try:
    describe(partial_nd)
    r.check("a dropped channel is an error, not a silent 1-channel sample",
            False, "no error raised")
except FileNotFoundError:
    r.check("a dropped channel is an error, not a silent 1-channel sample", True)

stage_dir = os.path.join(tmp, "stage")
stage_nd, _ = make_acquisition(stage_dir, "plate", positions=4)
try:
    describe(stage_nd)
    r.check("multiposition nd refused clearly", False, "no error raised")
except ValueError as error:
    r.check("multiposition nd refused clearly", "stage positions" in str(error),
            str(error)[:70])

mismatch_dir = os.path.join(tmp, "mismatch")
mismatch_nd, _ = make_acquisition(mismatch_dir, "wrongz", z=4)
tifffile.imwrite(  # 7 planes where the nd promised 4
    os.path.join(mismatch_dir, "wrongz_w1CSU500.STK"),
    np.zeros((7, 8, 6), dtype=np.uint16),
)
try:
    describe(mismatch_nd)
    r.check("plane count mismatch refused", False, "no error raised")
except ValueError as error:
    r.check("plane count mismatch refused", "planes" in str(error), str(error)[:70])

print("\n== timelapse across per-timepoint files ==")
time_dir = os.path.join(tmp, "timelapse")
time_nd, time_pixels = make_acquisition(time_dir, "movie", channels=2, z=3,
                                        timepoints=3)
time_entry = describe(time_nd)
r.check("timepoints picked up", time_entry.shape == (3, 2, 3, 8, 6),
        f"{time_entry.shape}")
time_folder = merge_file(time_nd)
time_written = os.path.join(time_folder, "movie.ome.tif")
time_data = load_image(time_written, lazy=False).data
r.check("timelapse shape", time_data.shape == (3, 2, 3, 8, 6), f"{time_data.shape}")
ordered = all(
    np.array_equal(time_data[t, c], time_pixels[(c, t)])
    for t in range(3) for c in range(2)
)
r.check("every timepoint and channel landed in the right place", ordered)

print("\n== scanning a directory ==")
r.check("finds the acquisition", [e.name for e in scan_directory(time_dir)]
        == ["movie.nd"], f"{[e.name for e in scan_directory(time_dir)]}")
r.check("unreadable acquisitions are skipped, not fatal",
        scan_directory(stage_dir) == [], f"{scan_directory(stage_dir)}")

nd_sample = sample_path("NUCLOGIC_TEST_ND")
if nd_sample:
    print("\n== real MetaMorph acquisition ==")
    import shutil

    real_dir = os.path.join(tmp, "real")
    os.makedirs(real_dir)
    stem = os.path.splitext(os.path.basename(nd_sample))[0]
    for name in os.listdir(os.path.dirname(nd_sample)):
        if name.startswith(stem):
            shutil.copy2(os.path.join(os.path.dirname(nd_sample), name), real_dir)

    real_nd = os.path.join(real_dir, os.path.basename(nd_sample))
    real_entry = describe(real_nd)
    print(f"  shape={real_entry.shape} channels={real_entry.channel_names}",
          flush=True)
    r.check("real acquisition has channels to merge", real_entry.shape[1] > 1,
            f"{real_entry.shape}")

    (z_um, y_um, x_um), xy_missing = voxel_size_of(real_entry)
    r.check("real Z step recovered from the stack", z_um > 0, f"{z_um}")

    real_folder = merge_file(real_nd)
    real_written = os.path.join(real_folder, stem + ".ome.tif")
    merged = load_image(real_written)
    r.check("real merged shape", tuple(merged.data.shape) == real_entry.shape,
            f"{merged.data.shape}")
    r.check("real Z calibration preserved",
            abs(merged.voxel_size[0] - z_um) < 1e-6, f"{merged.voxel_size}")

    for channel, path in sorted(real_entry.files.items()):
        raw = tifffile.imread(path)
        got = np.asarray(merged.data[0, channel[0]].compute())
        r.check(f"real channel {channel[0]} pixels intact", np.array_equal(raw, got))
    shutil.rmtree(real_dir, ignore_errors=True)
else:
    print("\n  (no sample .nd configured, skipped)", flush=True)

r.finish()
