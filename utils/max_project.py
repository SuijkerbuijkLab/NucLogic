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
    elif file.endswith(".tif"):
        movie = tifffile.imread(file)  # T,Z,C,Y,X

    # Project axis Z, if fixed we have 1 time axis less
    if fixed:
        max_data = np.max(movie, axis=1)
    else:
        max_data = []
        for t in range(movie.shape[0]):
            max_data.append(np.max(movie[t], axis=1))
        max_data = np.array(max_data)

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
