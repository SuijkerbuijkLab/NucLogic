import tifffile
import scipy.ndimage as ndimage
from skimage.measure import regionprops
import pandas as pd
import numpy as np
import trackpy as tp

import os

from load_model import load_model


def crop_tiff_stack(XY_path, output_directory):
    model = load_model(True, r"C:\Users\6331823\Desktop\TEMP\TL02\smoothed_XY\models\cpsam_20250709_110949")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    original = tifffile.imread(XY_path)

    XY = ndimage.zoom(original, (1, 0.2, 0.2), order=0)

    XY = ndimage.gaussian_filter(XY, sigma=(0, 2, 2))

    timepoints, y, x = XY.shape

    masks = []
    for frame in range(timepoints):
        print(frame)
        image = XY[frame, :, :]
        # tif = os.path.join(output_directory, f"frame-{frame}.tif")
        # tifffile.imwrite(tif, image)
        mask, _, _ = model.eval(image, diameter=None, do_3D=False)
        masks.append(mask)

    XY_mask = np.stack(masks, axis=0).astype(np.uint16)

    rescale_factors = (1, original.shape[1] / XY_mask.shape[1], original.shape[2] / XY_mask.shape[2])
    XY_mask = ndimage.zoom(masks, rescale_factors, order=0)
    tifffile.imwrite(os.path.join(output_directory, "mask.tif"), XY_mask)
    print(" saved mask")

    features = []
    for t in range(timepoints):
        props = regionprops(XY_mask[t])
        for prop in props:
            features.append({
                'frame': t,
                'y': prop.centroid[0],
                'x': prop.centroid[1],
                'label': prop.label
            })
    features_df = pd.DataFrame(features)

    linked = tp.link_df(features_df, search_range=30, memory=2)  # adjust search_range as needed

    # Now 'particle' column gives the tracked ID for each organoid
    print(linked.head())

    # Example: get the trajectory for the organoid closest to center at t=0
    center_y, center_x = XY_mask.shape[1] / 2, XY_mask.shape[2] / 2
    t0 = linked[linked['frame'] == 0].copy()
    t0['dist'] = ((t0['y'] - center_y)**2 + (t0['x'] - center_x)**2)**0.5
    closest_particle = t0.loc[t0['dist'].idxmin(), 'particle']

    # Get all frames for this tracked organoid
    organoid_track = linked[linked['particle'] == closest_particle]
    print(organoid_track)

    organoid_only = np.zeros_like(original)
    for _, row in organoid_track.iterrows():
        t = int(row['frame'])
        label = int(row['label'])
        organoid_only[t][XY_mask[t] == label] = original[t][XY_mask[t] == label]

    tifffile.imwrite(os.path.join(output_directory, "XY_tracked_cropped.tif"), organoid_only)

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

        cropped = organoid_only.copy()[
            frame, row_min:row_max, col_min:col_max
        ]
        tifffile.imwrite(os.path.join(output_directory, "tiffs",f"XY_cropped_{frame}.tif"), cropped)
        

input_directory = r"C:\Users\6331823\Desktop\TEMP\TL02"
output_directory = os.path.join(input_directory, "smoothed_XY")

XY_path = [
    os.path.join(input_directory, f)
    for f in os.listdir(input_directory)
    if "projXY" in f and f.endswith(".tif")
][0]

crop_tiff_stack(XY_path, input_directory)
