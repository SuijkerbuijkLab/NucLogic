import tifffile
import scipy.ndimage as ndimage
import cellpose.utils as cellpose_utils
from skimage.measure import regionprops
import pandas as pd

import os

from .load_model import load_model


def crop_tiff_stack(XY_path, output_directory):
    model = load_model(custom_model=True, model_path=r"")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original = tifffile.imread(XY_path)

    XY = ndimage.zoom(original, (1, 0.2, 0.2), order=0)

    XY = ndimage.gaussian_filter(XY, sigma=(0, 2, 2))

    timepoints, y, x = XY.shape

    masks = []
    for frame in range(timepoints):
        image = XY[frame, :, :]
        # tif = os.path.join(output_directory, f"frame-{frame}.tif")
        # tifffile.imwrite(tif, image)
        mask, _, _ = model.eval(image, diameter=None, do_3D=False)
        masks.append(mask)
    XY_mask = cellpose_utils.stack_to_3D(masks, axis=0)
    # In this mask try to find the one that is the organoid we want to keep.
    # Base this on the coordinates in XY and the size of the organoid.

    XY_mask = ndimage.zoom(masks, (1, 5, 5), order=0)

    props = regionprops(XY_mask)
    data = []
    for prop in props:
        label = prop.label
        centroid = prop.centroid  # (t, y, x)
        bounding_box = prop.bbox  # (min_t, min_y, min_x, max_t, max_y, max_x)
        volume = prop.area

        data.append(
            {
                "label": label,
                "t": centroid[0],
                "y": centroid[1],
                "x": centroid[2],
                "bounding_box": bounding_box,
                "volume": volume,
            }
        )

    df_props = pd.DataFrame(data)

    center_image = (XY_mask.shape[1] / 2, XY_mask.shape[2] / 2)
    df_props["distance_to_center"] = (
        (df_props["y"] - center_image[0]) ** 2 + (df_props["x"] - center_image[1]) ** 2
    ) ** 0.5
    # Filter for first timepoint
    df_first_timepoint = df_props[df_props["t"] == 0]

    # Find the row with the minimum distance to center
    closest_organoid = df_first_timepoint.loc[
        df_first_timepoint["distance_to_center"].idxmin()
    ]

    organoid = original.copy()
    organoid[organoid != closest_organoid["label"]] = 0

    tif = os.path.join(output_directory, f"XY_copped.tif")
    tifffile.imwrite(tif, organoid)


input_directory = r"C:\Users\6331823\Local SSD\TL02"
output_directory = os.path.join(input_directory, "smoothed_XY")

XY_path = [
    os.path.join(input_directory, f)
    for f in os.listdir(input_directory)
    if "projXY" in f and f.endswith(".tif")
][0]

crop_tiff_stack(XY_path, output_directory)
