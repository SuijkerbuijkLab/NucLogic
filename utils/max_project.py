# Function to make a maximum Z projection of the nuclei channel from an ims or tiff file

import os

import numpy as np
import tifffile
from imaris_ims_file_reader.ims import ims


def max_project(file, name="nuclei", fixed=False):
    # Load in either the ims or tif file
    if file.endswith(".ims"):
        movie = ims(file)  # T,C,Z,Y,X
        movie = movie[:]
        print(f"the movie shape is: {movie.shape}")
    elif file.endswith(".tif"):
        movie = tifffile.imread(file)  # T,Z,C,Y,X
        movie = np.moveaxis(movie, 2, 1)  # T,C,Z,Y,X
        print(f"the movie shape is: {movie.shape}")

    # Ensure movie has 5 dimensions
    while movie.ndim < 5:
        movie = np.expand_dims(movie, axis=0)  # add new dimension at the front
        print(f"added dimension, new shape: {movie.shape}")

    # Project axis Z, if fixed we have 1 time axis less
    max_data = np.max(movie, axis=2)  # T,C,Y,X
    print(f"the projected data shape is: {max_data.shape}")

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
