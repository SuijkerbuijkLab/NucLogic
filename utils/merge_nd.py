"""Merge a MetaMorph .nd acquisition into a single OME-TIFF sample.

MetaMorph splits one acquisition across files: a small .nd text index plus one
.STK per wavelength (and per timepoint, when it is a timelapse). The pipeline
expects one file per sample, so those parts are joined up front:

    20251216_gapstrain_1_1.nd
    20251216_gapstrain_1_1_w1CSU561.STK
    20251216_gapstrain_1_1_w2CSU488.STK
        ->  20251216_gapstrain_1_1/20251216_gapstrain_1_1.ome.tif

The originals are left alone; deleting them is offered afterwards, once the
merged file has been read back successfully.

The .nd is the authority on what the planes mean. tifffile reports an STK's
plane axis as "T" whether the planes are timepoints or Z slices, so a 61-plane
Z-stack read without the .nd would silently become a 61-frame timelapse.

Nothing here imports Qt: the UI drives it through the `progress` and
`should_stop` callbacks, like utils/split_positions.py.
"""

import os
import re

import numpy as np

from utils.load_image import load_image
from utils.save_as_tiff import Cancelled, save_as_tiff_stream

ND_EXTENSIONS = (".nd",)
STACK_EXTENSIONS = (".STK", ".stk", ".TIF", ".tif", ".TIFF", ".tiff")
OUTPUT_SUFFIX = ".ome.tif"

# MetaMorph writes the Z step into the STK, not the .nd, and calls it microns.
_UNCALIBRATED = {"", "0", "none"}


class NdFile:
    """One .nd acquisition and the stacks belonging to it."""

    def __init__(self, path, info, files, shape):
        self.path = path
        self.info = info
        self.files = files          # {(channel, timepoint): stack path}
        self.shape = shape          # T, C, Z, Y, X

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def stem(self):
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def channel_names(self):
        count = self.shape[1]
        return [
            str(self.info.get(f"WaveName{i + 1}", f"w{i + 1}")) for i in range(count)
        ]

    @property
    def sources(self):
        """Every file this sample is built from, the .nd included."""
        return [self.path] + sorted(set(self.files.values()))

    @property
    def estimated_bytes(self):
        return sum(
            os.path.getsize(p) for p in self.files.values() if os.path.exists(p)
        )

    @property
    def destination(self):
        folder = os.path.join(os.path.dirname(os.path.abspath(self.path)), self.stem)
        return os.path.join(folder, self.stem + OUTPUT_SUFFIX)

    @property
    def already_merged(self):
        return _is_readable(self.destination)


def parse_nd(path):
    """Read a .nd index into a dict.

    The format is one `"Key", value` pair per line, ending at "EndFile".
    """
    info = {}
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith('"'):
                continue
            key, _, value = line.partition(",")
            key = key.strip().strip('"')
            if key == "EndFile":
                break
            info[key] = _coerce(value.strip().strip('"').strip())
    return info


def _coerce(value):
    if value.upper() == "TRUE":
        return True
    if value.upper() == "FALSE":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def describe(path):
    """Work out what one .nd acquisition contains, or raise saying why not.

    Reads headers only: the .nd text plus the first page of one stack.
    """
    info = parse_nd(path)

    positions = int(info.get("NStagePositions", 1) or 1)
    if info.get("DoStage") and positions > 1:
        raise ValueError(
            f"{os.path.basename(path)} holds {positions} stage positions. "
            f"Multiposition MetaMorph acquisitions are not supported yet."
        )

    channels = max(1, int(info.get("NWavelengths", 1) or 1))
    timepoints = max(1, int(info.get("NTimePoints", 1) or 1))
    z_steps = max(1, int(info.get("NZSteps", 1) or 1))

    files = _resolve_files(path, info, channels, timepoints)
    planes, height, width = _stack_geometry(files[(0, 0)])

    if len(files) == channels:  # one file per channel, holding every plane
        if planes != z_steps * timepoints:
            raise ValueError(
                f"{os.path.basename(path)} says {z_steps} Z steps x "
                f"{timepoints} timepoints, but its stack holds {planes} planes. "
                f"Refusing to guess how they are ordered."
            )
        if timepoints > 1:
            raise ValueError(
                f"{os.path.basename(path)} stores {timepoints} timepoints inside "
                f"one stack per channel. Only per-timepoint files (_t1, _t2, ...) "
                f"are supported, because the plane order is otherwise ambiguous."
            )
    elif planes != z_steps:
        raise ValueError(
            f"{os.path.basename(path)} says {z_steps} Z steps but its stacks hold "
            f"{planes} planes each."
        )

    return NdFile(path, info, files, (timepoints, channels, z_steps, height, width))


def _resolve_files(path, info, channels, timepoints):
    """Map every (channel, timepoint) to its stack file.

    Missing parts are an error, never a silently dropped channel.
    """
    directory = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    in_name = info.get("WaveInFileName", True)

    files = {}
    missing = []
    for channel in range(channels):
        wave = str(info.get(f"WaveName{channel + 1}", "")) if in_name else ""
        for timepoint in range(timepoints):
            found = _find_stack(directory, stem, channel, wave, timepoint, timepoints)
            if found is None:
                missing.append(f"{stem}_w{channel + 1}{wave}")
            else:
                files[(channel, timepoint)] = found

    if missing:
        raise FileNotFoundError(
            f"{os.path.basename(path)} refers to stacks that are not next to it: "
            + ", ".join(sorted(set(missing)))
        )

    # A timelapse may store every timepoint in one file per channel instead.
    if timepoints > 1 and len(set(files.values())) == channels:
        files = {(c, 0): files[(c, 0)] for c in range(channels)}
    return files


def _find_stack(directory, stem, channel, wave, timepoint, timepoints):
    bases = [f"{stem}_w{channel + 1}{wave}", f"{stem}_w{channel + 1}"]
    suffixes = [f"_t{timepoint + 1}", ""] if timepoints > 1 else [""]

    for base in bases:
        for suffix in suffixes:
            for extension in STACK_EXTENSIONS:
                candidate = os.path.join(directory, base + suffix + extension)
                if os.path.isfile(candidate):
                    return candidate
    return None


def _stack_geometry(path):
    """(planes, height, width) of a stack, without reading pixels."""
    import tifffile

    with tifffile.TiffFile(path) as stack:
        shape = stack.series[0].shape
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    return shape[-3], shape[-2], shape[-1]


def voxel_size_of(entry):
    """(z, y, x) in micrometres, and whether the file actually said.

    MetaMorph leaves XCalibration at 1.0 with empty units when the objective
    calibration was never set, which must be reported as missing so the voxel
    size from the advanced settings is used instead.
    """
    import tifffile

    with tifffile.TiffFile(entry.files[(0, 0)]) as stack:
        metadata = stack.stk_metadata or {}

    units = str(metadata.get("CalibrationUnits", "")).strip().lower()
    x_um = float(metadata.get("XCalibration", 0) or 0)
    y_um = float(metadata.get("YCalibration", 0) or 0)

    xy_missing = units in _UNCALIBRATED or x_um <= 0 or y_um <= 0 or x_um == 1.0
    if xy_missing:
        # None, not 1.0: the merged file must not claim a calibration the
        # acquisition never had, or the voxel size override is never applied.
        x_um = y_um = None

    z_um = _z_step_um(metadata, entry.info)
    return (z_um, y_um, x_um), xy_missing


def _z_step_um(metadata, info):
    distances = np.asarray(metadata.get("ZDistance", []), dtype=float).ravel()
    distances = distances[distances > 0]
    if distances.size:
        return float(np.median(distances))

    step = float(info.get("ZStepSize", 0) or 0)
    return step if step > 0 else 1.0


def time_interval_of(entry):
    """Hours between timepoints, or 1.0 when the acquisition is not a timelapse."""
    import tifffile

    timepoints = entry.shape[0]
    if timepoints < 2 or (0, 1) not in entry.files:
        return 1.0

    stamps = []
    for timepoint in (0, 1):
        with tifffile.TiffFile(entry.files[(0, timepoint)]) as stack:
            stamps.append((stack.stk_metadata or {}).get("CreateTime"))

    if all(stamps) and stamps[1] > stamps[0]:
        return (stamps[1] - stamps[0]).total_seconds() / 3600.0
    return 1.0


def scan_directory(directory):
    """Every .nd acquisition sitting loose in `directory`."""
    found = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or not name.lower().endswith(ND_EXTENSIONS):
            continue
        try:
            found.append(describe(path))
        except Exception as error:
            print(f"Warning: skipping {name}: {error}")
    return found


def merge_file(path, progress=None, should_stop=None):
    """Write one .nd acquisition out as a single OME-TIFF sample folder.

    Returns the sample folder, or None when the merge was cancelled.
    """
    entry = path if isinstance(path, NdFile) else describe(path)
    destination = entry.destination
    folder = os.path.dirname(destination)

    if entry.already_merged:
        if progress:
            progress(1.0, f"{entry.stem} (already done)")
        return folder

    voxel_size, _ = voxel_size_of(entry)
    time_interval = time_interval_of(entry)

    os.makedirs(folder, exist_ok=True)
    completed = save_as_tiff_stream(
        destination,
        _planes(entry, progress, should_stop),
        entry.shape,
        _dtype_of(entry),
        voxel_size,
        time_interval,
    )
    if not completed:
        return None

    _verify(destination, entry.shape, voxel_size, time_interval)
    if progress:
        progress(1.0, entry.stem)
    return folder


def _planes(entry, progress, should_stop):
    """Yield (Y, X) planes in T, C, Z order, one stack in memory at a time."""
    import tifffile

    timepoints, channels, z_steps = entry.shape[:3]

    done = 0
    total = timepoints * channels
    for timepoint in range(timepoints):
        for channel in range(channels):
            if should_stop and should_stop():
                raise Cancelled()

            # Falls back to (channel, 0) when one file holds every timepoint.
            key = (channel, timepoint if (channel, timepoint) in entry.files else 0)
            with tifffile.TiffFile(entry.files[key]) as stack:
                data = stack.series[0].asarray()
                if data.ndim == 2:
                    data = data[None]
                if data.shape[0] != z_steps:
                    data = data[timepoint * z_steps:(timepoint + 1) * z_steps]

            for plane in data:
                yield plane

            done += 1
            if progress:
                progress(done / total, entry.stem)


def _dtype_of(entry):
    import tifffile

    with tifffile.TiffFile(entry.files[(0, 0)]) as stack:
        return stack.series[0].dtype


def _verify(destination, expected_shape, voxel_size, time_interval):
    """Re-open what we wrote and confirm it reads back as the same sample."""
    data, voxel, interval, metadata_missing = load_image(destination)

    if tuple(data.shape) != tuple(expected_shape):
        raise ValueError(
            f"{os.path.basename(destination)} read back as {tuple(data.shape)}, "
            f"expected {tuple(expected_shape)}"
        )

    known = [(a, b) for a, b in zip(voxel_size, voxel) if a is not None]
    if not all(abs(a - b) <= 1e-6 for a, b in known):
        raise ValueError(
            f"{os.path.basename(destination)} read back voxel size {voxel}, "
            f"expected {voxel_size}"
        )
    if any(v is None for v in voxel_size) and not metadata_missing:
        raise ValueError(
            f"{os.path.basename(destination)} does not report its missing "
            f"calibration, so the voxel size override would be skipped."
        )
    if abs(interval - time_interval) > 1e-6:
        raise ValueError(
            f"{os.path.basename(destination)} read back a time interval of "
            f"{interval} h, expected {time_interval} h"
        )


def _is_readable(destination):
    if not os.path.exists(destination):
        return False
    try:
        load_image(destination)
        return True
    except Exception:
        return False
