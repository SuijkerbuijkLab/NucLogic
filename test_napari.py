import tifffile
from libpysal.weights import KNN
from esda.moran import Moran
import numpy as np
import napari

import os


image = tifffile.imread(
    r"Z:\users\6331823\Mario Pipeline\new\TL01_mario\frames\Channel-Ref-frame-39.tif"
)

XZ = np.max(image, axis=1)

for idx, row in enumerate(XZ):
    # Create weights for 1D data (e.g., k-nearest neighbors with k=2)
    coords = np.arange(len(row)).reshape(-1, 1)
    w_1d = KNN.from_array(coords, k=2)

    # Compute Moran's I
    moran = Moran(row, w_1d)
    print(f"Row {idx}, Moran's I: {moran.I}")


viewer = napari.Viewer()
viewer.add_image(XZ, name="max")


napari.run()
