import napari
import tifffile
import numpy as np

import os

input_directory = r"Z:\users\6331823\Mario Pipeline\new\TL01\segmented"

data = []
shapes = []
for frame in range(60):
    image = tifffile.imread(
        os.path.join(input_directory, f"Channel-ref-frame-{frame}_masks.tif")
    )
    data.append(image)
    shapes.append(image.shape)

# Find max shape for z, y, x
max_z = max(s[0] for s in shapes)
max_y = max(s[1] for s in shapes)
max_x = max(s[2] for s in shapes)

# Pad each image to max shape (centered)
padded_data = []
for image in data:
    pad_z = max_z - image.shape[0]
    pad_y = max_y - image.shape[1]
    pad_x = max_x - image.shape[2]
    pad_front = pad_z // 2
    pad_back = pad_z - pad_front
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    padded = np.pad(
        image,
        ((pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    padded_data.append(padded)

data = np.stack(padded_data, axis=0).astype(np.uint16)
print(data.shape)

viewer = napari.Viewer()
viewer.add_labels(data, name="TL01", scale=(1, 4.96 / 0.621, 1, 1))

napari.run()
