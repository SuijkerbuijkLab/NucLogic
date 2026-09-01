"""Single entry point for reading microscopy images into NucLogic.

Every pipeline step should call `load_image()` instead of branching on the file
extension itself. The loader returns one canonical shape for all formats:

    data                ndarray-like, always 5D in T, C, Z, Y, X
    voxel_size          (z_um, y_um, x_um)      -- micrometres
    time_interval       float                   -- hours between timepoints
    metadata_missing    bool                    -- True if voxel size was not
                                                   in the file (pipeline then
                                                   falls back to the user
                                                   override, see
                                                   segment_organoid._resolve_voxel_size)

Adding a format means writing one `_load_<fmt>` function and adding it to
`_LOADERS` -- no pipeline code changes.

Laziness
--------
`load_image(path, lazy=True)` (the default) returns a dask array for every
format. Downstream code that needs real
numpy (Cellpose, OpenCV, scikit-image) must materialise a slice first:

    movie = load_image(path).data          # lazy, nothing read yet
    frame = as_numpy(movie[timepoint])     # reads just this timepoint

`lazy=False` returns eager numpy arrays.

Heavy imports (tifffile, dask, nd2, ...) are done inside the loaders so that
importing this module stays cheap for app startup.
"""

import contextlib
import io
import os
import warnings
from datetime import datetime

import numpy as np

# Extensions are matched longest-first so ".ome.tif" wins over ".tif".
SUPPORTED_EXTENSIONS = (
    ".ome.tiff",
    ".ome.tif",
    ".ome.zarr",
    ".tiff",
    ".tif",
    ".lsm",
    ".ims",
    ".nd2",
    ".czi",
    ".lif",
    ".zarr",
)

# Voxel sizes outside this range are almost certainly a unit-conversion mistake
# rather than a real acquisition, and silently wrong voxel sizes corrupt every
# downstream morphometric. Warn loudly instead of failing.
PLAUSIBLE_VOXEL_UM = (0.005, 50.0)
MAX_CHANNELS = 6

# How readers name the stage-position axis: "P" in ND2, "S" (scene) elsewhere.
POSITION_AXES = "PS"

class ImageData(tuple):
    """movie, voxel_size, time_interval, metadata_missing = load_image(path)
    """

    __slots__ = ()

    def __new__(cls, data, voxel_size, time_interval, metadata_missing):
        return super().__new__(
            cls, (data, voxel_size, time_interval, metadata_missing)
        )

    data = property(lambda self: self[0])
    voxel_size = property(lambda self: self[1])
    time_interval = property(lambda self: self[2])
    metadata_missing = property(lambda self: self[3])


def extension_of(path):
    """Return the supported extension of `path`, or None if unsupported."""
    lowered = str(path).lower().rstrip("/\\")
    for extension in SUPPORTED_EXTENSIONS:
        if lowered.endswith(extension):
            return extension
    return None


def is_supported(path):
    """True if load_image can read this path. Replaces scattered endswith checks."""
    return extension_of(path) is not None


def as_numpy(array):
    """Materialise a (possibly lazy) array into real numpy memory."""
    compute = getattr(array, "compute", None)  # dask
    if callable(compute):
        return np.asarray(compute())
    return np.asarray(array)  # numpy, memmap, h5py-backed reader


def load_image(path, lazy=True, all_positions=False, position=None):
    """Load any supported image into canonical T, C, Z, Y, X form.

    Args:
        path: file (or .zarr directory) to read.
        lazy: return a dask array that reads on demand (default), instead of
              loading the whole movie into RAM.
        all_positions: keep every stage position / scene, returning a 6D
              P, T, C, Z, Y, X array. Formats that cannot hold more than one
              position still return P = 1, so callers never branch on format.
              Only works when every position has the same shape; a .lif whose
              scenes differ needs `position` instead.
        position: read exactly this position (0-based), still returning 5D and
              that position's own voxel size. Works whatever the other
              positions look like, so this is what splitting uses.

    The default reads the first position, which is what the pipeline wants;
    only the sample-folder generation in utils/split_positions.py passes
    either of the last two arguments.

    Returns:
        ImageData(data, voxel_size, time_interval, metadata_missing)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    extension = extension_of(path)
    if extension is None:
        raise ValueError(
            f"Unsupported image format: {os.path.basename(str(path))}. "
            f"Supported formats: {', '.join(sorted(set(SUPPORTED_EXTENSIONS)))}"
        )

    if all_positions and position is not None:
        raise ValueError("Pass either all_positions or position, not both.")

    if position is not None:
        total = count_positions(path)
        if not 0 <= int(position) < total:
            raise IndexError(
                f"Position {position} is out of range: "
                f"{os.path.basename(str(path))} has {total}."
            )

    data, voxel_size, time_interval, metadata_missing = _LOADERS[extension](
        path, lazy, all_positions, position
    )

    data = _ensure_canonical(data, all_positions)
    voxel_size = _check_voxel_size(voxel_size, path)
    if not lazy:
        data = as_numpy(data)

    return ImageData(data, voxel_size, float(time_interval), bool(metadata_missing))


def count_positions(path):
    """Number of stage positions / scenes in a file, without reading pixels.

    Cheap enough to run over a whole directory while the user waits: it opens
    headers only. Anything that cannot hold positions answers 1.
    """
    extension = extension_of(path)
    counter = _POSITION_COUNTERS.get(extension)
    if counter is None:
        return 1
    try:
        return max(1, int(counter(path)))
    except Exception as error:
        print(
            f"Warning: could not read the position count of "
            f"{os.path.basename(str(path))}: {error}"
        )
        return 1


def position_labels(path):
    """Each position's own name, or None where the file does not name them.

    Leica .lif names every scene ("WT_WT_Full_Org3"); ND2 stage positions are
    usually unnamed. Callers use a name when there is one and fall back to a
    position number otherwise.
    """
    extension = extension_of(path)
    reader = _POSITION_LABELLERS.get(extension)
    if reader is None:
        return [None] * count_positions(path)
    try:
        labels = list(reader(path))
    except Exception as error:
        print(
            f"Warning: could not read position names from "
            f"{os.path.basename(str(path))}: {error}"
        )
        return [None] * count_positions(path)
    return [str(label) if label else None for label in labels]


def _count_nd2_positions(path):
    import nd2

    with nd2.ND2File(path) as reader:
        return reader.sizes.get("P", 1)


def _nd2_position_labels(path):
    import nd2

    with nd2.ND2File(path) as reader:
        for loop in reader.experiment:
            points = getattr(getattr(loop, "parameters", None), "points", None)
            if points:
                return [getattr(point, "name", None) for point in points]
    return [None] * _count_nd2_positions(path)


def _count_bioio_positions(path):
    from bioio import BioImage

    return _bioio_scene_count(BioImage(path))


def _bioio_position_labels(path):
    from bioio import BioImage

    return list(BioImage(path).scenes)


_POSITION_COUNTERS = {
    ".nd2": _count_nd2_positions,
    ".czi": _count_bioio_positions,
    ".lif": _count_bioio_positions,
}

_POSITION_LABELLERS = {
    ".nd2": _nd2_position_labels,
    ".czi": _bioio_position_labels,
    ".lif": _bioio_position_labels,
}


# ── shared helpers ────────────────────────────────────────────────────────────


def _ensure_canonical(data, all_positions):
    expected = 6 if all_positions else 5
    if data.ndim != expected:
        order = "P, T, C, Z, Y, X" if all_positions else "T, C, Z, Y, X"
        raise ValueError(
            f"Loader returned a {data.ndim}D array; expected {expected}D ({order})"
        )
    return data


def _check_voxel_size(voxel_size, path):
    voxel_size = tuple(float(v) for v in voxel_size)
    if len(voxel_size) != 3:
        raise ValueError(f"voxel_size must be (z, y, x), got {voxel_size}")

    # Writers divide by the voxel size to store a TIFF resolution, so a zero or
    # negative value from a malformed file must never reach them.
    if any(v <= 0 for v in voxel_size):
        print(f"Warning: non-positive voxel size {voxel_size} um read from "
              f"{os.path.basename(str(path))}; falling back to 1.0 um.")
        voxel_size = tuple(v if v > 0 else 1.0 for v in voxel_size)

    low, high = PLAUSIBLE_VOXEL_UM
    implausible = [v for v in voxel_size if v != 1.0 and not (low <= v <= high)]
    if implausible:
        print(
            f"Warning: voxel size {voxel_size} um read from "
            f"{os.path.basename(str(path))} is outside the plausible range "
            f"{low}-{high} um. This usually means the file stores a different "
            f"unit (e.g. metres). Check the voxel size override in the advanced "
            f"settings before trusting size-based statistics."
        )
    return voxel_size


def _drop_extra_axes(array, axes, keep_positions=False):
    """Index away any axis that is not T/C/Z/Y/X (positions, scenes).

    With keep_positions the position axis survives instead, for the callers
    that split a multiposition file into one sample per position. Otherwise we
    take the first entry and say so rather than silently mangling the dimension
    order downstream.

    Only safe for readers whose axis names are authoritative (ND2, NGFF). Do
    not use it on tifffile axes, which label a plain Z,Y,X stack as "SYX".
    """
    axes = (axes or "").upper()
    keep = set("TCZYX") | (set(POSITION_AXES) if keep_positions else set())
    for name in [a for a in axes if a not in keep]:
        index = axes.index(name)
        if array.shape[index] > 1:
            print(
                f"Warning: file contains {array.shape[index]} entries along "
                f"axis '{name}'; using the first one."
                + (
                    " Generate sample folders to analyse every position."
                    if name in POSITION_AXES
                    else ""
                )
            )
        array = array[(slice(None),) * index + (0,)]
        axes = axes[:index] + axes[index + 1:]
    if keep_positions:
        # _to_tczyx only knows "P" as the position axis; scenes mean the same.
        axes = axes.replace("S", "P")
    return array, axes


def _scale_to_um(value, unit):
    """Convert a length in `unit` to micrometres."""
    unit = str(unit or "micrometer").strip().lower()
    factors = {
        "": 1.0,
        "micrometer": 1.0,
        "micron": 1.0,
        "um": 1.0,
        "µm": 1.0,
        "nanometer": 1e-3,
        "nm": 1e-3,
        "millimeter": 1e3,
        "mm": 1e3,
        "centimeter": 1e4,
        "cm": 1e4,
        "meter": 1e6,
        "m": 1e6,
    }
    return float(value) * factors.get(unit, 1.0)

def _time_to_hours(time_value, time_unit):
    unit = str(time_unit or "h").strip().lower()
    value = float(time_value)
    factors = {
        "": 1.0 / 3600.0,
        "h": 1.0,
        "hr": 1.0,
        "hrs": 1.0,
        "hour": 1.0,
        "hours": 1.0,
        "min": 1.0 / 60.0,
        "mins": 1.0 / 60.0,
        "minute": 1.0 / 60.0,
        "minutes": 1.0 / 60.0,
        "s": 1.0 / 3600.0,
        "sec": 1.0 / 3600.0,
        "secs": 1.0 / 3600.0,
        "second": 1.0 / 3600.0,
        "seconds": 1.0 / 3600.0,
        "ms": 1.0 / 3.6e6,
        "msec": 1.0 / 3.6e6,
        "millisecond": 1.0 / 3.6e6,
        "milliseconds": 1.0 / 3.6e6,
        "us": 1.0 / 3.6e9,
        "usec": 1.0 / 3.6e9,
        "microsecond": 1.0 / 3.6e9,
        "microseconds": 1.0 / 3.6e9,
        "ns": 1.0 / 3.6e12,
        "nsec": 1.0 / 3.6e12,
        "nanosecond": 1.0 / 3.6e12,
        "nanoseconds": 1.0 / 3.6e12,
    }
    return float(value) * factors.get(unit, 1.0 / 3600.0)

def _xy_voxel_size_um_from_tiff_page(page, imagej_metadata):
    xres = page.tags.get("XResolution")
    yres = page.tags.get("YResolution")
    if not xres or not yres:
        return (1.0, 1.0)

    def _ratio_to_float(ratio):
        num, den = ratio
        return float(num) / float(den) if float(den) != 0 else 0.0

    x_ppu = _ratio_to_float(xres.value)
    y_ppu = _ratio_to_float(yres.value)

    res_unit_tag = page.tags.get("ResolutionUnit")
    res_unit = str(res_unit_tag.value).lower() if res_unit_tag else ""

    if "inch" in res_unit:
        um_per_resolution_unit = 25400.0
    elif "centimeter" in res_unit:
        um_per_resolution_unit = 10000.0
    else:
        ij_unit = str((imagej_metadata or {}).get("unit", "um")).lower()
        um_per_resolution_unit = 1.0 if ij_unit in {"um", "µm"} else 1.0

    x_size = (1.0 / x_ppu) * um_per_resolution_unit if x_ppu > 0 else 1.0
    y_size = (1.0 / y_ppu) * um_per_resolution_unit if y_ppu > 0 else 1.0
    return (y_size, x_size)

def read_tiff_metadata(tif):
    """Read voxel size and time interval from an already-open TiffFile."""
    import xml.etree.ElementTree as ET
    metadata_missing = False

    if tif.is_ome:
        root = ET.fromstring(tif.ome_metadata)
        ns = root.tag.split("}")[0].lstrip("{")
        pixels = root.find(f".//{{{ns}}}Pixels")
        z_size = pixels.get("PhysicalSizeZ") if pixels is not None else None
        y_size = pixels.get("PhysicalSizeY") if pixels is not None else None
        x_size = pixels.get("PhysicalSizeX") if pixels is not None else None
        metadata_missing = pixels is None or any(
            value is None for value in (z_size, y_size, x_size)
        )
        voxel_size = (
            float(z_size or 1.0),
            float(y_size or 1.0),
            float(x_size or 1.0),
        )
        time_raw = (
            float(pixels.get("TimeIncrement", 1.0)) if pixels is not None else 1.0
        )
        time_unit = pixels.get("TimeIncrementUnit", "h") if pixels is not None else "h"
        time_interval = _time_to_hours(time_raw, time_unit)

    elif tif.is_imagej:
        ij = tif.imagej_metadata or {}
        z_size = float(ij.get("spacing", 1.0))

        # Some ImageJ files store finterval=0 even when a valid TimeIncrement exists.
        finterval_value = ij.get("finterval", None)
        time_increment_value = ij.get("TimeIncrement", 1.0)

        try:
            finterval_float = (
                float(finterval_value) if finterval_value is not None else None
            )
        except (TypeError, ValueError):
            finterval_float = None

        try:
            time_increment_float = float(time_increment_value)
        except (TypeError, ValueError):
            time_increment_float = 1.0

        if finterval_float is None or finterval_float <= 0:
            time_raw = time_increment_float if time_increment_float > 0 else 1.0
            time_unit = ij.get("TimeIncrementUnit", ij.get("tunit", "h"))
        else:
            time_raw = finterval_float
            time_unit = ij.get("tunit", ij.get("TimeIncrementUnit", "h"))

        time_interval = _time_to_hours(time_raw, time_unit)

        y_size, x_size = _xy_voxel_size_um_from_tiff_page(tif.pages[0], ij)
        voxel_size = (z_size, y_size, x_size)

    else:
        metadata_missing = True
        voxel_size = (1.0, 1.0, 1.0)
        time_interval = 1.0

    return voxel_size, time_interval, metadata_missing


def _canonical_order(all_positions):
    return "PTCZYX" if all_positions else "TCZYX"


def _to_tczyx(arr, axes, order="TCZYX"):
    """Reorder an array to canonical T, C, Z, Y, X (or P, T, C, Z, Y, X).

    Uses tifffile's axes string (e.g. "TZYX", "CZYX", "TCZYX") when it cleanly
    describes the array, inserting length-1 axes for any of the target axes
    that are absent. Falls back to a size heuristic when the axes metadata is
    unusable.

    Works on dask arrays as well as numpy arrays: squeeze/expand_dims/transpose
    all dispatch through __array_function__, so a lazy array stays lazy.
    """
    axes = (axes or "").upper()
    known = set(order)

    usable = len(axes) == arr.ndim and all(
        a in known or arr.shape[i] == 1 for i, a in enumerate(axes)
    )
    if usable:
        # Drop any axes outside the target order (guaranteed length-1 above).
        for a in [x for x in axes if x not in known]:
            i = axes.index(a)
            arr = np.squeeze(arr, axis=i)
            axes = axes[:i] + axes[i + 1:]
        # Insert missing axes as length-1 (order fixed by the transpose).
        for a in order:
            if a not in axes:
                arr = np.expand_dims(arr, axis=0)
                axes = a + axes
        return np.transpose(arr, [axes.index(a) for a in order])

    # No reliable axes metadata: infer from shape. Only formats that cannot
    # hold positions reach this, so a length-1 P is always the right answer.
    if arr.ndim == 2:  # Y, X — a single plane, e.g. a squeezed projection
        arr = arr[None, None, None]
    elif arr.ndim == 3:  # Z, Y, X
        arr = arr[None, None]
    elif arr.ndim == 4:  # ambiguous C,Z,Y,X vs T,Z,Y,X -> few leading planes = channels
        arr = arr[None] if arr.shape[0] <= MAX_CHANNELS else arr[:, None]
    elif arr.shape[1] > arr.shape[2]:  # 5D: T,Z,C,Y,X -> T,C,Z,Y,X
        arr = np.transpose(arr, (0, 2, 1, 3, 4))
    return arr[None] if "P" in order else arr


# ── TIFF (OME-TIFF, ImageJ TIFF, plain TIFF, Zeiss LSM) ───────────────────────


def _load_tiff(path, lazy, all_positions=False, position=None):
    import tifffile

    with tifffile.TiffFile(path) as tif:
        voxel_size, time_interval, metadata_missing = read_tiff_metadata(tif)
        series = tif.series[0]
        axes = series.axes

        if lazy:
            # aszarr() works for compressed/tiled TIFFs too, where memmap fails.
            import dask.array as da

            data = da.from_zarr(series.aszarr())
        else:
            data = series.asarray()

    if data.ndim < 2:
        raise ValueError("TIFF must have at least 2 dimensions (Y, X)")

    # No _drop_extra_axes here on purpose: tifffile labels a plain Z,Y,X stack
    # as "SYX" (S = samples), and dropping that axis would discard the Z planes.
    # _to_tczyx's shape heuristic already resolves those cases correctly.
    data = _to_tczyx(data, axes, _canonical_order(all_positions))
    return data, voxel_size, time_interval, metadata_missing


# ── Imaris .ims ───────────────────────────────────────────────────────────────


_IMS_TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")


def _ims_attr_to_str(value):
    """IMS stores string attributes as arrays of single bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    return b"".join(bytes(v) for v in np.asarray(value).ravel()).decode(
        "utf-8", "replace"
    )


def _parse_ims_timestamp(text):
    for fmt in _IMS_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognised IMS timestamp: {text!r}")


def ims_acquisition_start(path):
    """Datetime of the first timepoint, or None when the file does not say."""
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            attrs = handle["DataSetInfo/TimeInfo"].attrs
            points = _sorted_timepoint_attrs(attrs)
            if points:
                return _parse_ims_timestamp(_ims_attr_to_str(attrs[points[0]]))
    except Exception:
        pass
    return None


def _sorted_timepoint_attrs(attrs):
    points = []
    for key in attrs:
        if not key.startswith("TimePoint"):
            continue
        try:
            points.append((int(key[len("TimePoint"):]), key))
        except ValueError:
            continue
    return [key for _, key in sorted(points)]


def _ims_time_interval_hours(path):
    """Hours between timepoints.

    DataSetTimes holds nanosecond timestamps and is the more precise source;
    DataSetInfo/TimeInfo holds formatted strings and is what PyImarisWriter
    writes. Falls back to 1.0 hour when neither is usable.
    """
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            rows = handle["DataSetTimes/Time"][:]
        if len(rows) >= 2:
            stamps = np.array([row[2] for row in rows], dtype=np.float64)
            interval = _time_to_hours(np.median(np.diff(stamps)), "ns")
            if interval > 0:
                return interval
    except Exception:
        pass

    try:
        with h5py.File(path, "r") as handle:
            attrs = handle["DataSetInfo/TimeInfo"].attrs
            points = _sorted_timepoint_attrs(attrs)
            if len(points) >= 2:
                first = _parse_ims_timestamp(_ims_attr_to_str(attrs[points[0]]))
                second = _parse_ims_timestamp(_ims_attr_to_str(attrs[points[1]]))
                interval = _time_to_hours((second - first).total_seconds(), "s")
                if interval > 0:
                    return interval
    except Exception:
        pass

    print(
        f"Warning: no usable time interval in {os.path.basename(str(path))}; "
        f"assuming 1.0 hour between timepoints."
    )
    return 1.0


@contextlib.contextmanager
def _quiet_ims():
    """Silence imaris_ims_file_reader's console noise.

    It prints on every open and close, and warns about missing HistogramMin/Max
    while scanning metadata for every resolution level -- including the
    downsampled ones the pipeline never reads.
    """
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings(
            "ignore", message=r"Histogram(Max|Min) value is not present"
        )
        yield


_QUIET_IMS_CLASS = None


def _quiet_ims_reader(path):
    """Open an .ims file without the reader's per-open/close chatter."""
    global _QUIET_IMS_CLASS

    if _QUIET_IMS_CLASS is None:
        # `ims` is a convenience function; ims_reader is the class behind it.
        from imaris_ims_file_reader.ims import ims_reader

        class QuietImsReader(ims_reader):
            def close(self):
                try:
                    with _quiet_ims():
                        super().close()
                except Exception:
                    # close() also runs from __del__ at interpreter shutdown,
                    # when redirecting stdout may no longer work.
                    super().close()

        _QUIET_IMS_CLASS = QuietImsReader

    with _quiet_ims():
        return _QUIET_IMS_CLASS(path)


class _DimensionPreservingReader:
    """Wrap the Imaris reader so sliced reads keep their dimensionality.

    imaris_ims_file_reader squeezes length-1 slices: reader[0:1, 0:1, :, :, :]
    on a 5D file returns (Z, Y, X), not (1, 1, Z, Y, X). dask requires each
    chunk to come back with the full number of dimensions, so without this any
    multi-chunk read fails.

    It also owns the file handle, and only opens it when pixels are first
    asked for. _load_ims reads the metadata with its own short-lived reader and
    closes it, so merely inspecting an .ims -- scanning a folder, reading a
    voxel size -- leaves the file unlocked. On Windows an open file cannot be
    moved, which otherwise blocks sorting samples into their folders.
    """

    def __init__(self, path, shape, dtype):
        self._path = path
        self._reader = None
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)

    def _open(self):
        if self._reader is None:
            self._reader = _quiet_ims_reader(self._path)
        return self._reader

    def close(self):
        reader, self._reader = self._reader, None
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))

        block = np.asarray(self._open()[key])
        # Integer keys are meant to drop their axis; slices are not.
        expected = tuple(
            len(range(*part.indices(size)))
            for part, size in zip(key, self.shape)
            if isinstance(part, slice)
        )
        return block.reshape(expected) if block.shape != expected else block


def _load_ims(path, lazy, all_positions=False, position=None):
    # Read the metadata with a reader of our own and let it go again: holding it
    # open would lock the file for as long as the returned array lives, even if
    # no pixel is ever read from it.
    reader = _quiet_ims_reader(path)  # HDF5-backed, slices lazily
    try:
        voxel_size = getattr(reader, "resolution", None)
        shape, dtype = tuple(reader.shape), reader.dtype
    finally:
        with _quiet_ims():
            reader.close()

    metadata_missing = voxel_size is None
    if metadata_missing:
        voxel_size = (1.0, 1.0, 1.0)

    time_interval = _ims_time_interval_hours(path)

    import dask.array as da

    # One chunk per (T, C) plane stack: matches how the pipeline iterates and how
    # max_project already streams IMS data. Built even for lazy=False, so the
    # eager path reads in chunks instead of one huge request (and because the
    # reader does not accept Ellipsis indexing).
    chunks = tuple(1 if i < len(shape) - 3 else size for i, size in enumerate(shape))
    # meta= keeps dask from probing the file with a zero-size read on open.
    data = da.from_array(
        _DimensionPreservingReader(path, shape, dtype),
        chunks=chunks,
        asarray=False,
        name=str(path),
        meta=np.empty((0,) * len(shape), dtype=dtype),
    )

    # IMS is stored T, C, Z, Y, X but drops leading singleton dimensions.
    while data.ndim < 5:
        data = data[None]

    if all_positions:  # IMS holds a single position
        data = data[None]

    return data, voxel_size, time_interval, metadata_missing


# ── Nikon .nd2 ────────────────────────────────────────────────────────────────


def _load_nd2(path, lazy, all_positions=False, position=None):
    import nd2

    with nd2.ND2File(path) as reader:
        axes = "".join(reader.sizes.keys()).upper()

        voxel = reader.voxel_size()  # namedtuple (x, y, z), micrometres
        voxel_size = (float(voxel.z), float(voxel.y), float(voxel.x))
        metadata_missing = any(v <= 0 for v in voxel_size)
        if metadata_missing:
            voxel_size = (1.0, 1.0, 1.0)

        time_interval = _nd2_time_interval_hours(reader)

    # nd2.imread(dask=True) opens the file per chunk, so the dask graph stays
    # valid after the ND2File above is closed.
    data = nd2.imread(path, dask=True) if lazy else nd2.imread(path)

    if position is not None and "P" in axes:
        index = axes.index("P")
        data = data[(slice(None),) * index + (int(position),)]
        axes = axes[:index] + axes[index + 1:]

    data, axes = _drop_extra_axes(data, axes, keep_positions=all_positions)
    data = _to_tczyx(data, axes, _canonical_order(all_positions))
    return data, voxel_size, time_interval, metadata_missing


def _nd2_time_interval_hours(reader):
    """Best-effort period between timepoints from the ND2 experiment loops."""
    try:
        for loop in reader.experiment:
            period_ms = getattr(getattr(loop, "parameters", None), "periodMs", None)
            if period_ms:
                return _time_to_hours(period_ms, "ms")
    except Exception as error:
        print(f"Warning: could not read ND2 time interval: {error}")
    return 1.0


# ── Zeiss .czi ────────────────────────────────────────────────────────────────


def _load_bioio(path, lazy, all_positions=False, position=None):
    """Zeiss .czi and Leica .lif, both read through bioio.

    Positions are scenes here, not an array axis. For CZI they are usually one
    acquisition at several stage positions; a .lif is more of a project file,
    so its scenes routinely differ in Z depth and even in channel count. That
    is why `position` exists: it never needs the scenes to agree.
    """
    from bioio import BioImage

    # reconstruct_mosaic defaults to True, so tiled acquisitions arrive already
    # stitched and the mosaic axis never reaches us.
    image = BioImage(path)
    scene_count = _bioio_scene_count(image)

    if all_positions:
        data = _stack_bioio_scenes(path, image, lazy)
        return (data,) + _bioio_metadata(image)

    if position is not None and int(position) > 0:
        image.set_scene(int(position))
    elif position is None and scene_count > 1:
        print(
            f"Warning: {os.path.basename(str(path))} contains {scene_count} "
            f"scenes; using the first one. Generate sample folders to analyse "
            f"every position."
        )

    # bioio guarantees the TCZYX order, so no _to_tczyx guessing is needed.
    data = image.dask_data if lazy else image.data
    return (data,) + _bioio_metadata(image)


def _bioio_metadata(image):
    """Voxel size, time interval and missing flag for the current scene.

    Read after any set_scene call: .lif scenes each carry their own calibration.
    """
    pixel_sizes = image.physical_pixel_sizes  # (Z, Y, X) in micrometres
    voxel_size = (
        float(pixel_sizes.Z or 0.0),
        float(pixel_sizes.Y or 0.0),
        float(pixel_sizes.X or 0.0),
    )
    metadata_missing = any(v <= 0 for v in voxel_size)
    if metadata_missing:
        voxel_size = (1.0, 1.0, 1.0)

    return voxel_size, _bioio_time_interval_hours(image), metadata_missing


def _bioio_scene_count(image):
    """Scene count, or 1 when the file carries no usable scene metadata.

    bioio_czi raises UnsupportedMetadataError rather than reporting a single
    scene for files written without scene records, so this must never be the
    thing that stops an otherwise readable file from loading.
    """
    try:
        return len(image.scenes)
    except Exception:
        return 1


def _stack_bioio_scenes(path, image, lazy):
    """Stack every scene into a leading position axis.

    Scenes are not an array axis in bioio, they are a reader mode, so each one
    is opened on its own BioImage rather than by re-pointing a shared reader.
    """
    from bioio import BioImage

    arrays = []
    for index in range(_bioio_scene_count(image)):
        scene_image = image if index == 0 else BioImage(path)
        scene_image.set_scene(index)
        arrays.append(scene_image.dask_data if lazy else scene_image.data)

    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) > 1:
        raise ValueError(
            f"{os.path.basename(str(path))} has scenes of differing shapes "
            f"({sorted(shapes)}); they cannot share one position axis. Read "
            f"them one at a time with load_image(path, position=i) instead."
        )

    if lazy:
        import dask.array as da

        return da.stack(arrays)
    return np.stack(arrays)


def _bioio_time_interval_hours(image):
    """Best-effort timepoint interval from a bioio TimeInterval, if present."""
    try:
        interval = getattr(image, "time_interval", None)
        if interval:
            return _time_to_hours(float(interval), "s")
    except Exception as error:
        print(f"Warning: could not read time interval: {error}")
    return 1.0


# ── OME-Zarr / NGFF ───────────────────────────────────────────────────────────


def _load_ome_zarr(path, lazy, all_positions=False, position=None):
    """Read the full-resolution level of an OME-NGFF multiscale pyramid.

    Voxel size and time interval come from the `coordinateTransformations`
    scale of that level, converted with the units declared in `axes`.
    """
    import dask.array as da
    import zarr

    node = zarr.open(str(path), mode="r")
    attributes = dict(node.attrs)
    multiscales = attributes.get("multiscales")

    if not multiscales:
        # A bare zarr array with no NGFF metadata: read it, guess the axes.
        array = node if hasattr(node, "shape") else node[list(node.array_keys())[0]]
        data = da.from_zarr(array) if lazy else np.asarray(array)
        data = _to_tczyx(data, "", _canonical_order(all_positions))
        return data, (1.0, 1.0, 1.0), 1.0, True

    multiscale = multiscales[0]
    dataset = multiscale["datasets"][0]  # level 0 == full resolution
    array = node[dataset["path"]]
    data = da.from_zarr(array) if lazy else np.asarray(array)

    axes_meta = multiscale.get("axes") or []
    # NGFF v0.3 stores axes as plain strings, v0.4+ as dicts with name/unit.
    axes = "".join(
        (axis if isinstance(axis, str) else axis.get("name", "")) for axis in axes_meta
    ).upper()
    units = [
        "" if isinstance(axis, str) else (axis.get("unit") or "") for axis in axes_meta
    ]

    scale = _ngff_scale(multiscale, dataset, len(axes))

    voxel_size, metadata_missing = _ngff_voxel_size(axes, units, scale)
    time_interval = _ngff_time_interval(axes, units, scale)

    data, axes = _drop_extra_axes(data, axes, keep_positions=all_positions)
    data = _to_tczyx(data, axes, _canonical_order(all_positions))
    return data, voxel_size, time_interval, metadata_missing


def _ngff_scale(multiscale, dataset, n_axes):
    """Per-axis scale, combining the dataset transform with any global one."""
    scale = [1.0] * n_axes
    for transforms in (
        multiscale.get("coordinateTransformations") or [],
        dataset.get("coordinateTransformations") or [],
    ):
        for transform in transforms:
            if transform.get("type") == "scale":
                values = transform.get("scale") or []
                for i, value in enumerate(values[:n_axes]):
                    scale[i] *= float(value)
    return scale


def _ngff_voxel_size(axes, units, scale):
    sizes = {}
    for i, name in enumerate(axes):
        if name in "ZYX":
            sizes[name] = _scale_to_um(scale[i], units[i] or "micrometer")

    metadata_missing = not sizes or all(value == 1.0 for value in sizes.values())
    return (
        sizes.get("Z", 1.0),
        sizes.get("Y", 1.0),
        sizes.get("X", 1.0),
    ), metadata_missing


def _ngff_time_interval(axes, units, scale):
    if "T" not in axes:
        return 1.0
    i = axes.index("T")
    interval = _time_to_hours(scale[i], units[i] or "second")
    return interval if interval > 0 else 1.0


_LOADERS = {
    ".tif": _load_tiff,
    ".tiff": _load_tiff,
    ".ome.tif": _load_tiff,
    ".ome.tiff": _load_tiff,
    ".lsm": _load_tiff,
    ".ims": _load_ims,
    ".nd2": _load_nd2,
    ".czi": _load_bioio,
    ".lif": _load_bioio,
    ".zarr": _load_ome_zarr,
    ".ome.zarr": _load_ome_zarr,
}
