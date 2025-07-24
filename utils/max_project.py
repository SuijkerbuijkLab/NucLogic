# Function to make a maximum Z projection from a ims file

from imaris_ims_file_reader.ims import ims
import numpy as np
import tifffile

import os


def max_project(file, output_directory, nuclei=2):
    if file.endswith(".ims"):
        movie = ims(file)  # T,C,Z,Y,X
    elif file.endswith(".tif"):
        movie = tifffile.imread(file)

    movie = movie[:, nuclei, :, :, :] # Remaining object is T Z Y X
    
    max_data = np.max(movie, axis=1) # Project axis Z 
    
    name = os.path.basename(file).split(".")[0]
    tif = os.path.join(output_directory, f"{name}_projXY.tif")
    tifffile.imwrite(tif, max_data)

    print(f"Saved max Z-projection at {tif}")
