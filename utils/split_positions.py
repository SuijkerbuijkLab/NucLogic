"""Turn a multiposition acquisition into one sample folder per position.

ND2 files and CZI files can hold many stage positions in a single container.
The pipeline analyses one sample per folder, so those positions are written out
as separate OME-TIFFs before analysis:

    experiment.nd2
    experiment_p01/experiment_p01.ome.tif
    experiment_p02/experiment_p02.ome.tif
    ...

The original file is left where it is; deleting it is the user's decision.

Nothing here imports Qt: the UI in PySide6/app.py drives it through the
`progress` and `should_stop` callbacks so the same code can be tested headless.
"""

import os
import re

from utils.load_image import as_numpy, count_positions, extension_of, is_supported, \
    load_image, position_labels
from utils.save_as_tiff import Cancelled, PART_SUFFIX, remove_quietly, \
    save_as_tiff_stream

OUTPUT_SUFFIX = ".ome.tif"

# Anything a Windows path cannot hold, plus the separators Leica puts in the
# series names it takes from its project tree.
_ILLEGAL_IN_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


class MultipositionFile:
    """One loose file that holds more than one position."""

    def __init__(self, path, positions, existing_folders):
        self.path = path
        self.positions = positions
        self.existing_folders = existing_folders

    @property
    def name(self):
        return os.path.basename(self.path)

    @property
    def already_split(self):
        return len(self.existing_folders) >= self.positions

    @property
    def estimated_bytes(self):
        """Rough output size: the source file size, before compression differs."""
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0


def stem_of(path):
    """File name without its image extension, handling .ome.tif correctly."""
    base = os.path.basename(str(path).rstrip("/\\"))
    extension = extension_of(base)
    return base[: -len(extension)] if extension else os.path.splitext(base)[0]


def sanitise(label):
    """A position name reduced to something a filesystem will accept."""
    cleaned = _ILLEGAL_IN_NAME.sub("_", str(label)).strip(" .")
    return cleaned[:80]


def position_folder_names(path, labels=None, total=None):
    """Folder name for every position of `path`, in order.

    `<stem>_<scene name>` when the file names its positions, which Leica .lif
    always does, and `<stem>_p01` when it does not. Keeping the stem in front
    matters: two .lif files in one directory can easily use the same scene
    names, and bare names would collide into a single sample folder.

    Names that survive sanitising but repeat within the file fall back to the
    numbered form, so one file can never produce two identical folders.
    """
    if total is None:
        total = count_positions(path)
    if labels is None:
        labels = position_labels(path)

    stem = stem_of(path)
    width = max(2, len(str(total)))
    numbered = [f"{stem}_p{index + 1:0{width}d}" for index in range(total)]

    cleaned = [sanitise(label) if label else "" for label in labels[:total]]
    cleaned += [""] * (total - len(cleaned))

    names = []
    for index, name in enumerate(cleaned):
        duplicated = name and cleaned.count(name) > 1
        names.append(f"{stem}_{name}" if name and not duplicated else numbered[index])
    return names


def position_folder_name(path, index, total):
    """Folder name for one position of `path`."""
    return position_folder_names(path, total=total)[index]


def existing_position_folders(path, names=None):
    """Sample folders already generated for `path`, by name.

    The expected names come from the file itself rather than from a pattern
    match, so scene-named folders are recognised as readily as numbered ones.
    Pass `names` when the caller already has them, to save re-opening the file.
    """
    parent = os.path.dirname(os.path.abspath(str(path)))
    return [
        name
        for name in (names if names is not None else position_folder_names(path))
        if os.path.isdir(os.path.join(parent, name))
    ]


def scan_directory(directory):
    """Loose files in `directory` that hold more than one position.

    Header reads only, so this is cheap enough to run whenever the user picks a
    folder (a 46-position ND2 answers in well under a tenth of a second).
    """
    found = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or not is_supported(name):
            continue
        positions = count_positions(path)
        if positions > 1:
            names = position_folder_names(path, total=positions)
            found.append(
                MultipositionFile(
                    path, positions, existing_position_folders(path, names)
                )
            )
    return found


def split_file(path, progress=None, should_stop=None):
    """Write one sample folder per position of `path`.

    progress(fraction, label) is called as work completes, with fraction in
    0..1. should_stop() is polled between timepoints; when it returns True the
    partially written file is deleted and the split stops, leaving every
    already-completed position in place.

    Returns the list of folders that hold a finished sample.
    """
    total = count_positions(path)
    names = position_folder_names(path, total=total)
    parent = os.path.dirname(os.path.abspath(str(path)))

    written = []
    for index in range(total):
        if should_stop and should_stop():
            break

        name = names[index]
        folder = os.path.join(parent, name)
        destination = os.path.join(folder, name + OUTPUT_SUFFIX)

        if _is_readable(destination):
            # A previous run finished this one; resuming must not redo the work.
            written.append(folder)
            if progress:
                progress((index + 1) / total, f"{name} (already done)")
            continue

        # One position at a time, with its own calibration: .lif scenes differ
        # in depth and in voxel size, so nothing here may assume they match.
        position, voxel_size, time_interval, _ = load_image(path, position=index)
        os.makedirs(folder, exist_ok=True)

        def on_timepoint(done, timepoints, _index=index, _name=name):
            if progress:
                progress((_index + done / timepoints) / total, _name)

        completed = _write_position(
            position, destination, voxel_size, time_interval,
            should_stop, on_timepoint,
        )
        if not completed:
            break

        _verify(destination, position.shape, voxel_size, time_interval)
        written.append(folder)
        if progress:
            progress((index + 1) / total, name)

    return written


def _write_position(position, destination, voxel_size, time_interval,
                    should_stop, on_timepoint):
    """Stream one position to an OME-BigTIFF. False if it was cancelled."""
    return save_as_tiff_stream(
        destination,
        _planes(position, should_stop, on_timepoint),
        position.shape,
        position.dtype,
        voxel_size,
        time_interval,
    )


def _planes(position, should_stop, on_timepoint):
    """Yield (Y, X) planes in the declared T, C, Z order.

    One (Z, Y, X) stack is materialised at a time: that is the chunk both the
    ND2 and IMS readers hand back, so peak memory stays flat no matter how long
    the movie is.
    """
    timepoints, channels = position.shape[0], position.shape[1]
    for t in range(timepoints):
        if should_stop and should_stop():
            raise Cancelled()
        for c in range(channels):
            stack = as_numpy(position[t, c])
            for plane in stack:
                yield plane
        on_timepoint(t + 1, timepoints)


def _verify(destination, expected_shape, voxel_size, time_interval):
    """Re-open what we wrote and confirm it reads back as the same sample."""
    data, voxel, interval, _ = load_image(destination)

    if tuple(data.shape) != tuple(expected_shape):
        raise ValueError(
            f"{os.path.basename(destination)} read back as {tuple(data.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    if not all(abs(a - b) <= 1e-6 for a, b in zip(voxel, voxel_size)):
        raise ValueError(
            f"{os.path.basename(destination)} read back voxel size {voxel}, "
            f"expected {voxel_size}"
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


_remove = remove_quietly
