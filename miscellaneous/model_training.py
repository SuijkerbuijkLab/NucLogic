import tifffile
import scipy.ndimage as ndimage
from imaris_ims_file_reader.ims import ims
import numpy as np

import os

input_directory = r"C:\Users\6331823\Local SSD\Train model whole organoid segmentation"
output_directory = r"C:\Users\6331823\Local SSD\Train model whole organoid segmentation"

for file in os.listdir(input_directory):
    if file.endswith(".tif"):
        # image = ims(os.path.join(input_directory, file)) #T,C,Z,Y,X

        # image = image[0, 1, :, :, :]  # Remaining object is Z Y X

        # max_data = np.max(image, axis=0) # YX
        max_data = tifffile.imread(os.path.join(input_directory, file))
        # if "40x" in file.lower():
        #    resize_factors = (100 / max_data.shape[0], 100 / max_data.shape[1])
        # elif "20x" in file.lower():
        resize_factors = (200 / max_data.shape[0], 200 / max_data.shape[1])
        print(max_data.shape)
        XY = ndimage.zoom(max_data, resize_factors, order=0)
        XY = ndimage.gaussian_filter(XY, sigma=(2, 2))
        tifffile.imwrite(
            f"{os.path.join(output_directory,file.split('.')[0])}_small.tif", XY
        )
