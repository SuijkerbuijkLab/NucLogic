import tifffile
import napari
import numpy as np

import os


image = tifffile.imread(
    r"Z:\users\6331823\Mario Pipeline\new\TL27\frames\Channel-Ref-frame-39.tif"
)[:, 200:820, 170:800]

XZ = np.max(image, axis=1)

for idx, row in enumerate(XZ):
    # Create weights for 1D data (e.g., k-nearest neighbors with k=2)
    coords = np.arange(len(row)).reshape(-1, 1)
    w_1d = KNN.from_array(coords, k=2)

    # Compute Moran's I
    moran = Moran(row, w_1d)
    print(f"Row {idx}, Moran's I: {moran.I}")


# viewer = napari.Viewer()
# viewer.add_image(XZ, name="max")


# napari.run()
