import tifffile
import scipy.ndimage as ndimage
from skimage.measure import regionprops
import pandas as pd
import numpy as np
import trackpy as tp
from libpysal.weights import KNN
from esda.moran import Moran
from scipy.ndimage import zoom
from alive_progress import alive_bar

import os

from utils.load_model import load_model
from utils.threshold import threshold

# from load_model import load_model
# from threshold import threshold


def crop(XY_path, input_frames, output_directory, model):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original = tifffile.imread(XY_path)

    resize_factors = (1, 200 / original.shape[1], 200 / original.shape[2])

    XY = ndimage.zoom(original, resize_factors, order=0)

    XY = ndimage.gaussian_filter(XY, sigma=(0, 2, 2))

    timepoints, y, x = XY.shape

    masks = []
    for frame in range(timepoints):
        image = XY[frame, :, :]
        mask, _, _ = model.eval(image, diameter=None, do_3D=False)
        masks.append(mask)

    XY_mask = np.stack(masks, axis=0).astype(np.uint16)

    rescale_factors = (
        1,
        original.shape[1] / XY_mask.shape[1],
        original.shape[2] / XY_mask.shape[2],
    )
    XY_mask = ndimage.zoom(masks, rescale_factors, order=0)

    features = []
    for t in range(timepoints):
        props = regionprops(XY_mask[t])
        for prop in props:
            features.append(
                {
                    "frame": t,
                    "y": prop.centroid[0],
                    "x": prop.centroid[1],
                    "label": prop.label,
                }
            )
    features_df = pd.DataFrame(features)

    linked = tp.link_df(
        features_df, search_range=30, memory=2
    )  # adjust search_range as needed

    # Now 'particle' column gives the tracked ID for each organoid

    # Example: get the trajectory for the organoid closest to center at t=0
    center_y, center_x = XY_mask.shape[1] / 2, XY_mask.shape[2] / 2
    t0 = linked[linked["frame"] == 0].copy()
    t0["dist"] = ((t0["y"] - center_y) ** 2 + (t0["x"] - center_x) ** 2) ** 0.5
    closest_particle = t0.loc[t0["dist"].idxmin(), "particle"]

    # Get all frames for this tracked organoid
    organoid_track = linked[linked["particle"] == closest_particle]
    print(organoid_track)

    organoid_only = np.zeros_like(original)
    for _, row in organoid_track.iterrows():
        t = int(row["frame"])
        label = int(row["label"])
        organoid_only[t][XY_mask[t] == label] = original[t][XY_mask[t] == label]

    tifffile.imwrite(
        os.path.join(output_directory, "XY_tracked_cropped.tif"), organoid_only
    )

    with alive_bar(timepoints, title="Cropping frames") as bar:
        # Find coordinates of non-zero pixels
        for frame in range(timepoints):
            maskXY = organoid_only[frame] > 0
            coordsXY = np.where(maskXY)

            # Adjust cropping limits based on XY projection
            if coordsXY[0].size > 0:
                row_min, row_max = (
                    np.min(coordsXY[0]),
                    np.max(coordsXY[0]) + 1,
                )
                col_min, col_max = (
                    np.min(coordsXY[1]),
                    np.max(coordsXY[1]) + 1,
                )

            ref = tifffile.imread(
                os.path.join(input_frames, f"Channel-ref-frame-{frame}.tif")
            )[:, row_min:row_max, col_min:col_max]

            XZ = np.max(image, axis=1)
            z_list = []
            for idx, row in enumerate(XZ):
                # Create weights for 1D data (e.g., k-nearest neighbors with k=2)
                coords = np.arange(len(row)).reshape(-1, 1)
                w_1d = KNN.from_array(coords, k=2)

                # Compute Moran's I
                moran = Moran(row, w_1d)
                if moran.I > 0.15:
                    z_list.append(idx)

            z_vals_sorted = sorted(set(z_list))
            max_block = []
            current_block = [z_vals_sorted[0]]

            for i in range(1, len(z_vals_sorted)):
                if z_vals_sorted[i] == z_vals_sorted[i - 1] + 1:
                    current_block.append(z_vals_sorted[i])
                else:
                    if len(current_block) > len(max_block):
                        max_block = current_block
                    current_block = [z_vals_sorted[i]]

            # Final check in case the longest block ends the loop
            if len(current_block) > len(max_block):
                max_block = current_block

            slice_min = max(min(max_block) - 1, 0)
            slice_max = min(max(max_block) + 1, ref.shape[0])
            ref = ref[slice_min:slice_max, :, :]

            tifffile.imwrite(
                os.path.join(output_directory, f"Channel-ref-frame-{frame}.tif"), ref
            )

            for x in ["CRC", "WT"]:
                image = tifffile.imread(
                    os.path.join(input_frames, f"Channel-{x}-frame-{frame}.tif")
                )[slice_min:slice_max, row_min:row_max, col_min:col_max]

                tifffile.imwrite(
                    os.path.join(output_directory, f"Channel-{x}-frame-{frame}.tif"),
                    image,
                )
            bar()
