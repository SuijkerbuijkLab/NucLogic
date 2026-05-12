import os

import numpy as np
import tifffile
from imaris_ims_file_reader.ims import ims
from alive_progress import alive_bar


def max_project(file, name="nuclei"):
    # Load in either the ims or tif file
    if file.endswith(".ims"):
        movie = ims(file)  # T,C,Z,Y,X
        shape = movie.shape

        # Ensure shape has 5 dimensions by padding at the front
        full_shape = shape
        while len(full_shape) < 5:
            full_shape = (1,) + full_shape

        output_shape = (full_shape[0], full_shape[1], full_shape[3], full_shape[4])
        max_data = np.zeros(output_shape, dtype=movie.dtype)
        ndim_diff = 5 - movie.ndim

        # IMS is HDF5-backed so each read triggers disk I/O — process one (T, C) at a time
        with alive_bar(
            full_shape[0] * full_shape[1], title="Generating Max Z projection"
        ) as bar:
            for t in range(full_shape[0]):
                for c in range(full_shape[1]):
                    idx = [t, c, slice(None), slice(None), slice(None)][ndim_diff:]
                    z_stack = movie[tuple(idx)]  # Z,Y,X
                    max_data[t, c] = np.max(z_stack, axis=0)
                    bar()

    elif file.endswith((".tif", ".tiff")):
        try:
            movie = tifffile.memmap(file, mode="r")  # T,Z,C,Y,X
        except ValueError:
            movie = tifffile.imread(file)

        shape = movie.shape
        while len(shape) < 5:
            shape = (1,) + shape
            movie = movie.reshape(shape)

        # T,Z,C,Y,X -> max over Z (axis 1) -> T,C,Y,X
        output_shape = (shape[0], shape[2], shape[3], shape[4])
        max_data = np.zeros(output_shape, dtype=movie.dtype)

        with alive_bar(shape[0], title="Generating Max Z projection") as bar:
            for t in range(shape[0]):
                # movie[t] is (Z,C,Y,X) — max over Z covers all channels in one call
                max_data[t] = np.max(movie[t], axis=0)
                bar()

    print(f"Saving max projection of shape {max_data.shape}")

    proj_XY_name = os.path.join(os.path.dirname(file), f"{name}_projXY.tif")
    tifffile.imwrite(
        proj_XY_name,
        max_data,
        imagej=True,
        metadata={"axes": "TCYX"},
        compression="zlib",
        compressionargs={"level": 8},
    )

    return max_data
