# Function to make a maximum Z projection of the nuclei channel from an ims or tiff file

import os

import numpy as np
import tifffile
from imaris_ims_file_reader.ims import ims


def max_project(file, nuclei=2, name="nuclei", fixed=False):
    # Load in either the ims or tif file
    if file.endswith(".ims"):
        movie = ims(file)  # T,C,Z,Y,X

        # Get only the information of the nuclei channel
        movie = movie[:, nuclei, :, :, :]  # Remaining object is T Z Y X

    elif file.endswith(".tif"):
        movie = tifffile.imread(file)  # T,Z,C,Y,X

        # Get only the information of the nuclei channel
        movie = movie[:, :, nuclei, :, :]  # Remaining object is T Z Y X

    # Project axis Z, if fixed we have 1 time axis less
    if fixed:
        max_data = np.max(movie, axis=0)
    else:
        max_data = np.max(movie, axis=1)

    proj_XY_name = os.path.join(os.path.dirname(file), f"{name}_projXY.tif")
    tifffile.imwrite(
        proj_XY_name,
        max_data,
        compression="zlib",
        compressionargs={"level": 8},
    )

    return max_data
