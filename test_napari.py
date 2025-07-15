import tifffile
import esda
import libpysal
import numpy as np
import napari

import os


image = tifffile.imread(r"E:\Users\Sebastian_van_Dijk\TEMP\TL27\frames\Channel-Ref-frame-59.tif")[:, 200:820, 170:800]

XZ = np.max(image, axis=1)

for row in itterrows


# viewer = napari.Viewer()
# viewer.add_image(XZ, name="max")


# napari.run()
