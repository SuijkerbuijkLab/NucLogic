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


def load_tiff_movie_and_metadata(input_file):
    """Load TIFF movie and metadata.

    Returns:
        loaded_movie: ndarray in T, C, Z, Y, X
        voxel_size: tuple (z_um, y_um, x_um)
        time_interval_hours: float
        metadata_missing: bool
    """
    metadata_missing = False

    with tifffile.TiffFile(input_file) as tif:
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
            time_unit = (
                pixels.get("TimeIncrementUnit", "h") if pixels is not None else "h"
            )
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

        loaded_movie = tif.asarray()

    while loaded_movie.ndim < 5:
        loaded_movie = np.expand_dims(loaded_movie, axis=0)
    loaded_movie = np.transpose(
        loaded_movie, (0, 2, 1, 3, 4)
    )  # from T,Z,C,Y,X to T,C,Z,Y,X

    return loaded_movie, voxel_size, time_interval, metadata_missing
