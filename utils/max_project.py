# Function to make a maximum Z projection of the nuclei channel from an ims or tiff file

from imaris_ims_file_reader.ims import ims
import numpy as np
import tifffile

import os


def max_project(file, output_directory, nuclei=2):
    # Load in either the ims or tif file
    if file.endswith(".ims"):
        movie = ims(file)  # T,C,Z,Y,X
    elif file.endswith(".tif"):
        movie = tifffile.imread(file)

    # Get only the information of the nuclei channel
    movie = movie[:, nuclei, :, :, :]  # Remaining object is T Z Y X

    # Project axis Z
    max_data = np.max(movie, axis=1)

    # Save the new file
    name = os.path.basename(file).split(".")[0]
    tif = os.path.join(output_directory, f"{name}_projXY.tif")
    tifffile.imwrite(tif, max_data)

    print(f"Saved max Z-projection at {tif}")
