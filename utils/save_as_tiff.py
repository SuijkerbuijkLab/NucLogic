"""Single writer for every TIFF the pipeline produces.

Counterpart to utils/load_image.py: anything written here reads back through
load_image with its voxel size and time interval intact.

Files are OME-BigTIFF. The earlier writers combined imagej=True with
bigtiff=True, which tifffile accepts only under protest ("writing nonconformant
BigTIFF ImageJ") because the ImageJ format is capped at 4 GB and has no BigTIFF
variant. OME carries the same calibration, has no size limit, and is what Fiji
reaches for through Bio-Formats anyway.
"""

import os

import tifffile

PART_SUFFIX = ".part"


class Cancelled(Exception):
    """Raised inside a plane generator to unwind a cancelled write."""


def save_as_tiff_stream(destination, planes, shape, dtype, voxel_size,
                        time_interval, axes="TCZYX"):
    """Write an OME-BigTIFF from an iterator of 2D planes, atomically.

    `planes` must yield the planes of `shape` in the order `axes` declares,
    slowest axis first, so nothing larger than a single plane stack is ever
    held in memory. Raising Cancelled from the iterator aborts cleanly.

    A voxel_size component may be None where the source file did not say. That
    axis is then left out of the OME metadata rather than written as 1.0, so
    load_image reports metadata_missing and the pipeline applies the voxel size
    from the advanced settings instead of a made-up one.

    The file is built under a .part name and renamed only once complete, so an
    interrupted run can never leave a truncated file that looks finished.

    Returns True, or False when the write was cancelled.
    """
    part = str(destination) + PART_SUFFIX

    metadata = {
        "axes": axes,
        "TimeIncrement": float(time_interval),
        "TimeIncrementUnit": "h",
    }
    for name, value in zip(("Z", "Y", "X"), voxel_size):
        if value is not None and float(value) > 0:
            metadata["PhysicalSize" + name] = float(value)

    try:
        with tifffile.TiffWriter(part, bigtiff=True, ome=True) as writer:
            writer.write(planes, shape=tuple(shape), dtype=dtype, metadata=metadata)
    except Cancelled:
        remove_quietly(part)
        return False
    except BaseException:
        remove_quietly(part)
        raise

    os.replace(part, destination)
    return True


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def save_as_tiff(path, array, axes, voxel_size, time_interval,
                 compression="zlib", compression_level=8):
    """Write `array` as an OME-BigTIFF.

    Args:
        path: destination file.
        axes: dimension names of `array`, e.g. "TCZYX", "TZCYX", "TZYX", "ZYX".
        voxel_size: (z, y, x) in micrometres.
        time_interval: hours between timepoints.
    """
    z_um, y_um, x_um = (float(v) if float(v) > 0 else 1.0 for v in voxel_size)

    tifffile.imwrite(
        path,
        array,
        bigtiff=True,
        ome=True,
        resolution=(1 / x_um, 1 / y_um),
        metadata={
            "axes": axes,
            "PhysicalSizeZ": z_um,
            "PhysicalSizeY": y_um,
            "PhysicalSizeX": x_um,
            "TimeIncrement": float(time_interval),
            "TimeIncrementUnit": "h",
        },
        compression=compression,
        compressionargs=(
            {"level": compression_level} if compression == "zlib" else None
        ),
    )
