# Function that will crop a movie based on an existing XY max projection

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
from imaris_ims_file_reader.ims import ims

import os

from utils.load_model import load_model
from utils.threshold import threshold


def crop(XY_path, input_file, output_directory, model, nuclei=2):
    # Create the output directory to save the files
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the XY max projection, resize it smaller and apply a gaussian blur so cellpose can find the organoids
    original = tifffile.imread(XY_path)
    resize_factors = (1, 200 / original.shape[1], 200 / original.shape[2])
    XY = ndimage.zoom(original, resize_factors, order=0)
    XY = ndimage.gaussian_filter(XY, sigma=(0, 2, 2))

    # Get the dimensions of the movie XY max projection to loop over timepoints
    timepoints, y, x = XY.shape

    # Loop over every frame
    masks = []
    for frame in range(timepoints):
        image = XY[frame, :, :]
        # Apply cellpose to the frame to find where organoids are
        mask, _, _ = model.eval(
            image,
            diameter=None,
            normalize=True,
            flow_threshold=0.4,
            invert=False,
            resample=True,
            do_3D=False,
        )

        # Dilate the found masks to make sure we don't cut off outer membranes / edges of organoids
        labels = np.unique(mask)
        labels = labels[labels != 0]  # skip background
        dilated = np.zeros_like(mask)
        for label in labels:
            binary = mask == label
            dilated_binary = ndimage.binary_dilation(binary, iterations=3)
            dilated[dilated_binary] = label

        # Save the mask of this frame to the list of the whole movie
        masks.append(dilated)

    # Concatante the list of masks into a real movie of masks
    XY_mask = np.stack(masks, axis=0).astype(np.uint16)

    # Rescale the masks back to the original size
    rescale_factors = (
        1,
        original.shape[1] / XY_mask.shape[1],
        original.shape[2] / XY_mask.shape[2],
    )
    XY_mask = ndimage.zoom(masks, rescale_factors, order=0)

    # Get the data of every mask in the movie
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

    # Track the found masks over time during the movie
    linked = tp.link_df(
        features_df, search_range=40, memory=2
    )  # adjust search_range as needed

    # Get the trajectory for the organoid closest to center at t=0
    center_y, center_x = XY_mask.shape[1] / 2, XY_mask.shape[2] / 2
    t0 = linked[linked["frame"] == 0].copy()
    t0["dist"] = ((t0["y"] - center_y) ** 2 + (t0["x"] - center_x) ** 2) ** 0.5
    closest_particle = t0.loc[t0["dist"].idxmin(), "particle"]

    # Get all frames for this tracked organoid
    organoid_track = linked[linked["particle"] == closest_particle]
    print(organoid_track)

    # Make a movie that only shows the original movie at the place of the selected organoid
    organoid_only = np.zeros_like(original)
    for _, row in organoid_track.iterrows():
        t = int(row["frame"])
        label = int(row["label"])
        organoid_only[t][XY_mask[t] == label] = original[t][XY_mask[t] == label]

    # Save this tracked movie to the output folder
    name = os.path.basename(XY_path).replace(".tif", "")
    tifffile.imwrite(
        os.path.join(
            os.path.dirname(output_directory),
            f"{name}_tracked.tif",
        ),
        organoid_only,
    )

    # Load in the input image, which is either a ims or tiff
    if input_file.endswith(".ims"):
        movie = ims(input_file)  # T,C,Z,Y,X
    elif input_file.endswith(".tif"):
        movie = tifffile.imread(input_file)

    # Loop over every frame in the movie to crop that frame.
    with alive_bar(timepoints, title="Cropping frames") as bar:
        for frame in range(timepoints):

            # Find the XY coordinates of the mask
            maskXY = organoid_only[frame] > 0
            coordsXY = np.where(maskXY)

            # Adjust cropping limits based on XY projection, creating a bounding box around the organoid
            if coordsXY[0].size > 0:
                row_min, row_max = (
                    np.min(coordsXY[0]),
                    np.max(coordsXY[0]) + 1,
                )
                col_min, col_max = (
                    np.min(coordsXY[1]),
                    np.max(coordsXY[1]) + 1,
                )

            # Get the frame of the movie
            ref = movie[frame, nuclei, :, row_min:row_max, col_min:col_max]

            # Get a bounding box cropped version of the organoid
            ref_crop = maskXY[row_min:row_max, col_min:col_max]

            # In this bounding box image, create the Z axis of the same dimensions of the original movie
            ref_expanded = ref_crop[np.newaxis, :, :]
            ref_expanded = np.repeat(ref_expanded, ref.shape[0], axis=0)

            # Use the this XYZ mask on the original movie to crop as both a bounding box and to make all pixels (corners) where there is no organoid actually black
            ref_masked = np.where(ref_expanded > 0, ref, 0).astype(np.uint16)

            # Make an max XZ projection that we will use to see what Z slices are important
            XZ = np.max(ref_masked, axis=1)
            z_list = []
            moran_values = []  # Collect Moran’s I for all rows

            # Step 1: Calculate Moran's I for each row
            for idx, row in enumerate(XZ):
                coords = np.arange(len(row)).reshape(-1, 1)
                w_1d = KNN.from_array(coords, k=2)
                moran = Moran(row, w_1d)
                moran_values.append((idx, moran.I))  # store both index and value
                print(moran.I)

            # Step 2: Calculate dynamic threshold based on max moran I value found
            max_moran = max(val for _, val in moran_values)
            threshold = 0.2 * max_moran

            # Step 3: Filter Z slices using threshold
            z_list = [idx for idx, val in moran_values if val >= threshold]

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

            slice_min = max(min(max_block) - 2, 0)
            slice_max = min(max(max_block) + 2, ref.shape[0])

            # This time, crop the frame of the movie on all channels using corrext XYZ
            cropped_image = movie[
                frame, :, slice_min:slice_max, row_min:row_max, col_min:col_max
            ]

            # On this cropped frame, we again need to make it so that the pixels in the corners where no organoid is are actually black
            mask_crop = maskXY[
                row_min:row_max, col_min:col_max
            ]  # shape: (Y_crop, X_crop)
            # Expand mask to 4D: (C, Z, Y_crop, X_crop)
            mask_expanded = mask_crop[np.newaxis, np.newaxis, :, :]
            # Repeat over C and Z to match cropped_image shape
            mask_expanded = np.repeat(
                mask_expanded, cropped_image.shape[0], axis=0
            )  # C
            mask_expanded = np.repeat(
                mask_expanded, cropped_image.shape[1], axis=1
            )  # Z
            # Apply the mask
            cropped_image_masked = np.where(mask_expanded, cropped_image, 0).astype(
                np.uint16
            )

            # Save this final cropped frame in the output folder
            tifffile.imwrite(
                os.path.join(output_directory, f"Frame-{frame}.tif"),
                cropped_image_masked,
            )
            bar()
