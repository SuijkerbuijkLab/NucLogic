"""Multiposition splitting: the load_image flag, the writer, resume and cancel.

Synthetic ND2 files cannot be written, so the position axis is exercised through
a fake reader and through the real .nd2 sample when one is configured.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _harness import Results, sample_path  # noqa: E402

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from utils import split_positions  # noqa: E402
from utils.file_to_folder import file_to_folder  # noqa: E402
from utils.load_image import count_positions, load_image  # noqa: E402
from utils.split_positions import (  # noqa: E402
    existing_position_folders, position_folder_name, position_folder_names,
    sanitise, scan_directory, split_file, stem_of,
)

r = Results()
tmp = tempfile.mkdtemp()

REF = np.random.default_rng(0).integers(0, 4000, (3, 2, 4, 16, 8)).astype(np.uint16)
VOXEL = (2.0, 0.33, 0.33)
INTERVAL = 0.5


def write_ome(path, array=REF):
    tifffile.imwrite(
        path, array, ome=True,
        metadata={"axes": "TCZYX", "PhysicalSizeZ": VOXEL[0],
                  "PhysicalSizeY": VOXEL[1], "PhysicalSizeX": VOXEL[2],
                  "TimeIncrement": 30.0, "TimeIncrementUnit": "min"},
    )
    return path


print("== the flag leaves single-position formats alone ==")
single = write_ome(os.path.join(tmp, "single.ome.tif"))
default = load_image(single)
flagged = load_image(single, all_positions=True)
r.check("default stays 5D", default.data.ndim == 5, f"{default.data.shape}")
r.check("flag gives 6D", flagged.data.ndim == 6, f"{flagged.data.shape}")
r.check("P is 1", flagged.data.shape[0] == 1, f"{flagged.data.shape}")
r.check("pixels identical to default",
        np.array_equal(np.asarray(flagged.data[0]), np.asarray(default.data)))
r.check("count_positions is 1", count_positions(single) == 1)

plain = os.path.join(tmp, "plain.tif")
tifffile.imwrite(plain, REF[0, 0])
r.check("plain tiff 6D", load_image(plain, all_positions=True).data.shape
        == (1, 1, 1, 4, 16, 8), f"{load_image(plain, all_positions=True).data.shape}")

print("\n== _to_tczyx keeps a position axis in front ==")
from utils.load_image import _drop_extra_axes, _to_tczyx  # noqa: E402

fake = np.zeros((5, 3, 2, 16, 8), dtype=np.uint16)  # P, T, C, Y, X
kept, axes = _drop_extra_axes(fake, "PTCYX", keep_positions=True)
r.check("position axis survives", kept.shape == fake.shape and "P" in axes, f"{axes}")
r.check("PTCZYX order", _to_tczyx(kept, axes, "PTCZYX").shape == (5, 3, 2, 1, 16, 8),
        f"{_to_tczyx(kept, axes, 'PTCZYX').shape}")

dropped, axes2 = _drop_extra_axes(fake, "PTCYX")
r.check("position axis dropped by default", dropped.shape == (3, 2, 16, 8), f"{dropped.shape}")
r.check("scenes normalise to P",
        _drop_extra_axes(fake, "STCYX", keep_positions=True)[1] == "PTCYX",
        _drop_extra_axes(fake, "STCYX", keep_positions=True)[1])

print("\n== folder naming ==")
r.check("stem strips .ome.tif", stem_of("/x/experiment.ome.tif") == "experiment",
        stem_of("/x/experiment.ome.tif"))
r.check("pads to the highest number",
        position_folder_name("a.nd2", 0, 46) == "a_p01", position_folder_name("a.nd2", 0, 46))
r.check("three digits past 99",
        position_folder_name("a.nd2", 99, 150) == "a_p100",
        position_folder_name("a.nd2", 99, 150))

print("\n== named positions (the .lif case) ==")
named = position_folder_names("exp.lif", labels=["Org1", "Org2", "Org3"], total=3)
r.check("scene name follows the stem", named == ["exp_Org1", "exp_Org2", "exp_Org3"],
        f"{named}")

illegal = position_folder_names("exp.lif", labels=["Grp/Org 1", "a:b*c?"], total=2)
r.check("illegal characters replaced",
        all(not set(n) & set('<>:"/\\|?*') for n in illegal), f"{illegal}")

duplicated = position_folder_names("exp.lif", labels=["Org", "Org", "Other"], total=3)
r.check("repeated names fall back to numbers",
        duplicated == ["exp_p01", "exp_p02", "exp_Other"], f"{duplicated}")

partial_labels = position_folder_names("exp.lif", labels=["Org1", None, ""], total=3)
r.check("missing names fall back individually",
        partial_labels == ["exp_Org1", "exp_p02", "exp_p03"], f"{partial_labels}")

r.check("sanitise trims trailing dots and spaces",
        sanitise("name. ") == "name", repr(sanitise("name. ")))
r.check("sanitise caps the length", len(sanitise("x" * 300)) <= 80,
        len(sanitise("x" * 300)))

print("\n== the writer round-trips through load_image ==")


class FakeMulti:
    """Stands in for a multiposition acquisition without needing a real .nd2.

    Positions have different Z depths, like the scenes of a real .lif, so the
    per-position read path is what gets exercised.
    """

    def __init__(self, depths=(4, 6, 5), labels=None):
        self.positions = [
            np.random.default_rng(i).integers(0, 4000, (3, 2, depth, 16, 8))
            .astype(np.uint16)
            for i, depth in enumerate(depths)
        ]
        self.labels = labels or [None] * len(depths)

    def __call__(self, path, lazy=True, all_positions=False, position=None):
        from utils.load_image import ImageData
        if all_positions:
            raise ValueError("scenes of differing shapes")
        return ImageData(self.positions[position or 0], VOXEL, INTERVAL, False)


def install_fake(loader, *paths):
    """Point split_positions at `loader` for `paths`, leaving other files real."""
    targets = {str(p) for p in paths}
    split_positions.load_image = lambda path, **kw: (
        loader(path, **kw) if str(path) in targets else real_load_image(path, **kw)
    )
    split_positions.count_positions = lambda path: (
        len(loader.positions) if str(path) in targets else real_count(path)
    )
    split_positions.position_labels = lambda path: (
        loader.labels if str(path) in targets else real_labels(path)
    )


source = os.path.join(tmp, "multi", "movie.nd2")
os.makedirs(os.path.dirname(source), exist_ok=True)
open(source, "wb").close()

real_load_image = split_positions.load_image
real_count = split_positions.count_positions
real_labels = split_positions.position_labels

fake_loader = FakeMulti()
install_fake(fake_loader, source)

seen = []
folders = split_file(source, progress=lambda f, label: seen.append((round(f, 3), label)))
r.check("one folder per position", len(folders) == 3, f"{folders}")
r.check("named p01..p03",
        [os.path.basename(f) for f in folders] == ["movie_p01", "movie_p02", "movie_p03"],
        f"{[os.path.basename(f) for f in folders]}")
r.check("progress reaches 1.0", seen and seen[-1][0] == 1.0, f"{seen[-1] if seen else None}")

for index, folder in enumerate(folders):
    written = os.path.join(folder, os.path.basename(folder) + ".ome.tif")
    data, voxel, interval, missing = load_image(written, lazy=False)
    r.check(f"p{index + 1} pixels", np.array_equal(data, fake_loader.positions[index]))
    r.check(f"p{index + 1} keeps its own depth",
            data.shape[2] == fake_loader.positions[index].shape[2],
            f"{data.shape}")
    r.check(f"p{index + 1} voxel", np.allclose(voxel, VOXEL), f"{voxel}")
    r.check(f"p{index + 1} interval", abs(interval - INTERVAL) < 1e-9, f"{interval}")
    r.check(f"p{index + 1} metadata present", missing is False)
    with tifffile.TiffFile(written) as tif:
        r.check(f"p{index + 1} is OME BigTIFF", tif.is_ome and tif.is_bigtiff,
                f"ome={tif.is_ome} bigtiff={tif.is_bigtiff}")

r.check("no .part files left",
        not [f for folder in folders for f in os.listdir(folder) if f.endswith(".part")])

print("\n== resume skips finished positions ==")
again = split_file(source, progress=None)
r.check("same folders returned", len(again) == 3, f"{again}")
r.check("resume rewrote nothing",
        all(os.path.exists(os.path.join(f, os.path.basename(f) + ".ome.tif"))
            for f in again))

print("\n== cancel removes only the partial file ==")
cancel_source = os.path.join(tmp, "multi", "cancelme.nd2")
open(cancel_source, "wb").close()
cancel_loader = FakeMulti()
install_fake(cancel_loader, source, cancel_source)

state = {"calls": 0}


def stop_after_first_position():
    # Let position 1 finish, then refuse the next one.
    state["calls"] += 1
    return state["calls"] > 5


partial = split_file(cancel_source, should_stop=stop_after_first_position)
r.check("stopped early", len(partial) < 3, f"{partial}")
leftover = []
for name in os.listdir(os.path.dirname(cancel_source)):
    folder = os.path.join(os.path.dirname(cancel_source), name)
    if os.path.isdir(folder):
        leftover += [f for f in os.listdir(folder) if f.endswith(".part")]
r.check("no .part left after cancel", not leftover, f"{leftover}")
for folder in partial:
    written = os.path.join(folder, os.path.basename(folder) + ".ome.tif")
    r.check(f"{os.path.basename(folder)} is a valid sample",
            load_image(written).data.ndim == 5)

print("\n== scanning and already-split detection ==")
scan_dir = os.path.join(tmp, "scan")
os.makedirs(scan_dir, exist_ok=True)
write_ome(os.path.join(scan_dir, "ordinary.ome.tif"))
r.check("single-position files are not listed", scan_directory(scan_dir) == [],
        f"{scan_directory(scan_dir)}")

r.check("finds generated folders",
        existing_position_folders(source) == ["movie_p01", "movie_p02", "movie_p03"],
        f"{existing_position_folders(source)}")
r.check("unrelated names ignored", existing_position_folders(
    os.path.join(tmp, "scan", "ordinary.ome.tif")) == [])

# From here on the real readers are needed again.
split_positions.load_image = real_load_image
split_positions.count_positions = real_count
split_positions.position_labels = real_labels

print("\n== file_to_folder leaves multiposition files alone ==")
move_dir = os.path.join(tmp, "move")
os.makedirs(move_dir, exist_ok=True)
ordinary = write_ome(os.path.join(move_dir, "ordinary.ome.tif"))
file_to_folder(move_dir)
r.check("ordinary file moved into its folder",
        os.path.exists(os.path.join(move_dir, "ordinary", "ordinary.ome.tif")))

nd2_path = sample_path("NUCLOGIC_TEST_ND2")
if nd2_path:
    import shutil

    copied = os.path.join(move_dir, os.path.basename(nd2_path))
    shutil.copy2(nd2_path, copied)
    file_to_folder(move_dir)
    r.check("multiposition nd2 stayed put", os.path.isfile(copied))
    r.check("no folder made for it",
            not os.path.isdir(os.path.join(move_dir, stem_of(copied))))

    print("\n== real nd2 ==")
    r.check("counts 46 positions", count_positions(nd2_path) == 46,
            f"{count_positions(nd2_path)}")
    every = load_image(nd2_path, all_positions=True)
    one = load_image(nd2_path)
    r.check("6D with P first", every.data.shape[0] == 46, f"{every.data.shape}")
    r.check("position 0 equals the default read",
            np.array_equal(np.asarray(every.data[0]), np.asarray(one.data)))
    entries = scan_directory(os.path.dirname(copied))
    r.check("scan reports it", any(e.positions == 46 for e in entries),
            f"{[(e.name, e.positions) for e in entries]}")
    r.check("not yet split", all(not e.already_split for e in entries))
else:
    print("  (no sample ND2 configured, skipped)", flush=True)

lif_path = sample_path("NUCLOGIC_TEST_LIF")
if lif_path:
    print("\n== real lif (ragged scenes) ==")
    from utils.load_image import position_labels

    total = count_positions(lif_path)
    r.check("counts more than one scene", total > 1, f"{total}")

    labels = position_labels(lif_path)
    r.check("every scene is named", all(labels), f"{labels[:3]}")
    names = position_folder_names(lif_path, total=total)
    r.check("folders use the scene names",
            all(stem_of(lif_path) in n for n in names)
            and any(labels[0] in n for n in names), f"{names[:2]}")

    shapes = []
    for index in range(total):
        data, voxel, _, missing = load_image(lif_path, position=index)
        shapes.append(data.shape)
        r.check(f"scene {index} has calibration", missing is False, f"{voxel}")
    r.check("scenes really do differ in shape", len(set(shapes)) > 1,
            f"{sorted(set(shapes))}")

    # A ragged file cannot share one position axis; it must say so, not guess.
    try:
        load_image(lif_path, all_positions=True)
        r.check("ragged all_positions rejected", False, "no error raised")
    except ValueError as error:
        r.check("ragged all_positions rejected", "differing shapes" in str(error),
                str(error)[:80])

    try:
        load_image(lif_path, position=total + 5)
        r.check("out-of-range position rejected", False, "no error raised")
    except IndexError:
        r.check("out-of-range position rejected", True)

    try:
        load_image(lif_path, all_positions=True, position=0)
        r.check("both arguments rejected", False, "no error raised")
    except ValueError:
        r.check("both arguments rejected", True)
else:
    print("\n  (no sample LIF configured, skipped)", flush=True)

r.finish()
