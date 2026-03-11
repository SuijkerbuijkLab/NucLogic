import os

import numpy as np
import tifffile
from imaris_ims_file_reader.ims import ims
from alive_progress import alive_bar


def max_project(file, name="nuclei", fixed=False):
    # Load in either the ims or tif file
    if file.endswith(".ims"):
        movie = ims(file)  # T,C,Z,Y,X
        # Don't load all data at once - get shape first
        shape = movie.shape

        # Ensure shape has 5 dimensions by padding at the front
        full_shape = shape
        while len(full_shape) < 5:
            full_shape = (1,) + full_shape

        # Output shape after removing Z axis (axis 2 in TCZYX)
        output_shape = (full_shape[0], full_shape[1], full_shape[3], full_shape[4])
        max_data = np.zeros(output_shape, dtype=movie.dtype)
        # Calculate how many dimensions were added
        ndim_diff = 5 - movie.ndim

        # Process one timepoint and channel at a time
        with alive_bar(
            full_shape[0] * full_shape[1], title="Generating Max Z projection"
        ) as bar:
            for t in range(full_shape[0]):
                for c in range(full_shape[1]):
                    # Dynamically build index, skipping padded dimensions
                    idx = [t, c, slice(None), slice(None), slice(None)][ndim_diff:]
                    z_stack = movie[tuple(idx)]  # Z,Y,X

                    # Compute max projection along Z
                    max_data[t, c, :, :] = np.max(z_stack, axis=0)
                    bar()

    elif file.endswith(".tif"):
        # For TIFF files, use memmap to avoid loading everything at once
        movie = tifffile.memmap(file, mode="r")  # T,Z,C,Y,X

        # Ensure movie has 5 dimensions by padding at the front
        shape = movie.shape
        while len(shape) < 5:
            shape = (1,) + shape
            # Reshape the memmap array to match
            movie = movie.reshape(shape)

        # For TIF: T,Z,C,Y,X -> need to project Z (axis 1)
        # Output will be T,C,Y,X
        output_shape = (shape[0], shape[2], shape[3], shape[4])
        max_data = np.zeros(output_shape, dtype=movie.dtype)

        # Process one timepoint and channel at a time
        with alive_bar(shape[0] * shape[2], title="Generating Max Z projection") as bar:
            for t in range(shape[0]):
                for c in range(shape[2]):
                    # Load just this T,C slice with all Z
                    z_stack = movie[t, :, c, :, :]  # Z,Y,X
                    # Compute max projection along Z
                    max_data[t, c, :, :] = np.max(z_stack, axis=0)
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
