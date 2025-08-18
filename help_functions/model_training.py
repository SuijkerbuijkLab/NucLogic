import tifffile
import scipy.ndimage as ndimage

# from imaris_ims_file_reader.ims import ims
import numpy as np
import napari
from alive_progress import alive_it

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.load_model import load_model

# input_directory = r"E:\Users\Sebastian_van_Dijk\Merel plexin organoids"
# output_directory = (
#     r"E:\Users\Sebastian_van_Dijk\Train model intestinal organoid segmentation"
# )

# for file in os.listdir(input_directory):
#     if file.endswith(".ims"):
#         image = ims(os.path.join(input_directory, file))  # T,C,Z,Y,X

#         image = image[0, 1, :, :, :]  # Remaining object is Z Y X

#         max_data = np.max(image, axis=0)  # YX

#         if "40x" in file.lower():
#             resize_factors = (100 / max_data.shape[0], 100 / max_data.shape[1])
#         elif "20x" in file.lower():
#             resize_factors = (200 / max_data.shape[0], 200 / max_data.shape[1])
#         XY = ndimage.zoom(max_data, resize_factors, order=0)
#         # XY = ndimage.gaussian_filter(XY, sigma=(2, 2))
#         tifffile.imwrite(
#             f"{os.path.join(output_directory,file.split('.im')[0])}_small.tif", XY
#         )
organoid_model = load_model(
    r"C:\Users\6331823\Local SSD\Train model whole organoid segmentation\smoothed_XY\models\whole_liver_organoid_segmentation",
)
input_directory = r"C:\Users\6331823\Local SSD\Data_Anna\all\EXpansionLMix_3"
file = os.path.join(input_directory, "ExpansionLMix_3.tif")

image = tifffile.imread(os.path.join(input_directory, file))  # T,Z,C,Y,X

image = image[:, :, 2, :, :]  # Remaining object is T Z Y X

max_data = np.max(image, axis=1)  # TYX
frames = max_data.shape[0]

viewer = napari.Viewer()
factors = [100, 200]

for factor in factors:
    masks = []
    for frame in alive_it(range(frames), title="Segmenting frames"):
        frame = max_data[frame]
        resize_factors = (factor / frame.shape[0], factor / frame.shape[1])
        XY = ndimage.zoom(frame, resize_factors, order=0)
        XY = ndimage.gaussian_filter(XY, sigma=(2, 2))
        mask, _, _ = organoid_model.eval(
            XY,
            diameter=None,
            normalize=True,
            flow_threshold=0.4,
            invert=False,
            resample=True,
            do_3D=False,
        )
        mask = ndimage.zoom(
            mask, (1 / resize_factors[0], 1 / resize_factors[1]), order=0
        )  # Rescale back to original size
        masks.append(mask)

    masks = np.stack(masks, axis=0).astype(np.uint16)
    viewer.add_labels(masks, name=f"{factor}_seg", blending="additive", opacity=0.5)

viewer.add_image(max_data, name="max_data", blending="additive", colormap="gray")
napari.run()
