# Function to make a maximum Z projection from a ims file

from imaris_ims_file_reader.ims import ims
import numpy as np
import tifffile

import os


def max_project(movie, output_directory, nuclei=3):
    movie = ims(movie)  # T,C,Z,Y,X
    movie = movie[:, nuclei, :, :, :]

    max_data = np.max(movie, axis=3)

    tif = os.path.join(output_directory, f"projXY.tif")
    tifffile.imwrite(tif, max_data)

    print(f"Saved max Z-projection at {tif}")
