import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def _time_to_hours(time_value, time_unit):
    unit = str(time_unit or "h").strip().lower()
    value = float(time_value)
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return value
    if unit in {"min", "mins", "minute", "minutes"}:
        return value / 60.0
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return value / 3600.0
    if unit in {"ms", "msec", "millisecond", "milliseconds"}:
        return value / 3600000.0
    return value


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


MAX_CHANNELS = 4  # more than this leading planes -> treat as timepoints, not channels


def _to_tczyx(arr, axes):
    """Reorder an array to canonical T, C, Z, Y, X.

    Uses tifffile's axes string (e.g. "TZYX", "CZYX", "TCZYX") when it cleanly
    describes the array, inserting length-1 axes for any of T/C/Z that are
    absent. Falls back to a size heuristic when the axes metadata is unusable.

    Works on dask arrays as well as numpy arrays: squeeze/expand_dims/transpose
    all dispatch through __array_function__, so a lazy array stays lazy.
    """
    axes = (axes or "").upper()
    known = set("TCZYX")

    usable = len(axes) == arr.ndim and all(
        a in known or arr.shape[i] == 1 for i, a in enumerate(axes)
    )
    if usable:
        # Drop any non-TCZYX axes (guaranteed length-1 by the check above).
        for a in [x for x in axes if x not in known]:
            i = axes.index(a)
            arr = np.squeeze(arr, axis=i)
            axes = axes[:i] + axes[i + 1:]
        # Insert missing T/C/Z as length-1 axes (order fixed by the transpose).
        for a in "TCZYX":
            if a not in axes:
                arr = np.expand_dims(arr, axis=0)
                axes = a + axes
        return np.transpose(arr, [axes.index(a) for a in "TCZYX"])

    # No reliable axes metadata: infer from shape.
    if arr.ndim == 3:  # Z, Y, X
        return arr[None, None]
    if arr.ndim == 4:  # ambiguous C,Z,Y,X vs T,Z,Y,X -> few leading planes = channels
        return arr[None] if arr.shape[0] <= MAX_CHANNELS else arr[:, None]
    if arr.shape[1] > arr.shape[2]:  # 5D: T,Z,C,Y,X -> T,C,Z,Y,X
        arr = np.transpose(arr, (0, 2, 1, 3, 4))
    return arr


def read_tiff_metadata(tif):
    """Read voxel size and time interval from an already-open TiffFile.

    Split out of load_tiff_movie_and_metadata so utils.load_image can reuse this
    parsing for its lazy path instead of duplicating it.

    Returns:
        voxel_size: tuple (z_um, y_um, x_um)
        time_interval_hours: float
        metadata_missing: bool
    """
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


def load_tiff_movie_and_metadata(input_file):
    """Load TIFF movie and metadata.

    Returns:
        loaded_movie: ndarray in T, C, Z, Y, X
        voxel_size: tuple (z_um, y_um, x_um)
        time_interval_hours: float
        metadata_missing: bool
    """
    with tifffile.TiffFile(input_file) as tif:
        voxel_size, time_interval, metadata_missing = read_tiff_metadata(tif)

        series = tif.series[0]
        loaded_movie = series.asarray()
        movie_axes = series.axes

    if loaded_movie.ndim < 3:
        raise ValueError("Loaded TIFF movie must have at least 3 dimensions (Z, Y, X)")

    loaded_movie = _to_tczyx(loaded_movie, movie_axes)

    return loaded_movie, voxel_size, time_interval, metadata_missing
