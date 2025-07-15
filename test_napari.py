import tifffile
import napari
import numpy as np

import os


input_directory = r"E:\Users\Sebastian_van_Dijk\TEMP\TL27\segmented"

frames = []
shapes = []

# Step 1: Load all frames and record their shapes
for frame in range(60):
    image = tifffile.imread(
        os.path.join(input_directory, f"Channel-ref-frame-{frame}_masks.tif")
    )
    frames.append(image)
    shapes.append(image.shape)

# Step 2: Find the maximum shape across all frames
max_shape = np.max(np.array(shapes), axis=0)

# Step 3: Pad each image symmetrically to match max_shape
def pad_to_shape(img, target_shape):
    pad_width = []
    for dim, target in zip(img.shape, target_shape):
        total_pad = target - dim
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(img, pad_width, mode='constant', constant_values=0)

padded_frames = [pad_to_shape(img, max_shape) for img in frames]

# Step 4: Stack into a single array
stacked = np.stack(padded_frames, axis=0)

viewer = napari.Viewer()
viewer.add_labels(stacked, name="max", scale = (1, 4.94/0.621, 1, 1))

napari.run()
