# Function that will crop a movie based on an existing XY max projection using a SAM model

import tifffile
import pandas as pd
import numpy as np
from libpysal.weights import KNN
from esda.moran import Moran
from imaris_ims_file_reader.ims import ims
from PIL import Image
import scipy.ndimage as ndimage
import cv2
from .offset_image import offset_image
from utils.pad_to_shape import pad_to_shape


import os, shutil, tempfile


def crop(
    proj_XY,
    input_file,
    model,
    nuclei=2,
    name="projXY_tracked",
):

    # Convert movie to 8 bit for meta SAM
    # nuclei may be a list (single or dual); always max-project over channel axis -> (T, H, W)
    nuclei_list = nuclei if isinstance(nuclei, list) else [nuclei]
    proj_XY_nuclei = np.max(proj_XY[:, nuclei_list, :, :], axis=1)
    proj_XY_8bit = ((proj_XY_nuclei / proj_XY_nuclei.max()) * 255).astype(np.uint8)

    all_masks = segment_organoid(proj_XY_8bit, model)

    # Dilate the masks by 12 pixels to ensure the organoid is fully in there
    new_masks = []
    for _, mask in enumerate(all_masks):
        mask = ndimage.binary_dilation(mask, iterations=12)
        new_masks.append(mask)
    new_masks = np.stack(new_masks, axis=0).astype(np.uint16)

    # Expand masks to match channel dimension: (T, Y, X) -> (T, C, Y, X)
    new_masks = new_masks[:, np.newaxis, :, :]  # Add channel axis
    new_masks = np.repeat(new_masks, proj_XY.shape[1], axis=1)

    # Create new movie that only has the tracked organoid
    organoid_only = np.where(new_masks, proj_XY, 0)

    # Save this tracked movie to the output folder
    tifffile.imwrite(
        os.path.join(
            os.path.dirname(input_file),
            f"{name}_projXY_tracked.tif",
        ),
        organoid_only,
        imagej=True,
        metadata={"axes": "TCYX"},
        compression="zlib",
        compressionargs={"level": 8},
    )

    # Load in the input image, which is either a ims or tiff
    if input_file.endswith(".ims"):
        movie = ims(input_file)  # T,C,Z,Y,X
    elif input_file.endswith(".tif"):
        movie = tifffile.imread(input_file)  # T,Z,C,Y,X
        movie = np.transpose(movie, (0, 2, 1, 3, 4))  # T,C,Z,Y,X

    timepoints = proj_XY.shape[0]

    cropped_movie = []
    max_dims = [0, 0, 0, 0]

    # Loop over every frame in the movie to crop that frame.
    for frame in range(timepoints):

            maskXY = np.max(organoid_only[frame, nuclei], axis=0) > 0
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

            # Get the frame of the movie — always max-project over nuclei channels (handles 1 or 2)
            nuclei_list = nuclei if isinstance(nuclei, list) else [nuclei]
            ref = np.max(
                np.stack(
                    [
                        movie[frame, ch, :, row_min:row_max, col_min:col_max]
                        for ch in nuclei_list
                    ]
                ),
                axis=0,
            )

            # Get a bounding box cropped version of the organoid
            ref_crop = maskXY[row_min:row_max, col_min:col_max]

            # In this bounding box image, create the Z axis of the same dimensions of the original movie
            ref_expanded = ref_crop[np.newaxis, :, :]
            ref_expanded = np.repeat(ref_expanded, ref.shape[0], axis=0)

            # Use the this XYZ mask on the original movie to crop as both a bounding box and to make all pixels (corners) where there is no organoid actually black
            ref_masked = np.where(ref_expanded > 0, ref, 0).astype(np.uint16)

            # Make an max XZ projection that we will use to see what Z slices are important
            XZ = np.max(ref_masked, axis=1)

            # Apply offset to enhance contrast and make background and signal more distinct
            XZ = offset_image(XZ, type="median")

            z_list = []
            moran_values = []  # Collect Moran’s I for all rows

            # Step 1: Calculate Moran's I for each row
            for idx, row in enumerate(XZ):
                coords = np.arange(len(row)).reshape(-1, 1)
                w_1d = KNN.from_array(coords, k=2)
                moran = Moran(row, w_1d)
                if not np.isnan(
                    moran.I
                ):  # Make sure this value is not NA, which can happen in truly background / random data
                    moran_values.append((idx, moran.I))  # store both index and value
                else:
                    moran_values.append((idx, 0))

            # Step 2: Calculate dynamic threshold based on max moran I value found
            max_moran = max(val for _, val in moran_values)
            threshold = 0.4 * max_moran

            # Step 3: Filter Z slices using threshold
            z_list = [idx for idx, val in moran_values if val >= threshold]

            z_vals_sorted = sorted(set(z_list))
            max_block = []
            current_block = [z_vals_sorted[0]]

            for i in range(1, len(z_vals_sorted)):
                if z_vals_sorted[i] - z_vals_sorted[i - 1] <= 6:
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
            cropped_image_masked = np.transpose(
                cropped_image_masked, (1, 0, 2, 3)
            )  # Z C Y X for tiff

            cropped_movie.append(cropped_image_masked)
            for i in range(4):
                max_dims[i] = max(max_dims[i], cropped_image_masked.shape[i])

    cropped_movie = [pad_to_shape(stack, max_dims) for stack in cropped_movie]
    cropped_movie = np.stack(cropped_movie, axis=0)

    return cropped_movie


def get_coords(movie):
    first_frame = movie[0]
    # Normalize and convert to 8-bit
    frame = ((first_frame / first_frame.max()) * 255).astype(np.uint8)
    frame = ndimage.gaussian_filter(frame, sigma=(10, 10))

    # Apply threshold (you can tweak the value)
    median = np.median(frame)
    _, binary_mask = cv2.threshold(frame, median + 5, 255, cv2.THRESH_BINARY)

    binary_mask = ndimage.binary_dilation(binary_mask, iterations=10)
    filled_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    num_labels, labels, stats_array, centroids = cv2.connectedComponentsWithStats(
        filled_mask
    )

    center = np.array([frame.shape[0] // 2, frame.shape[1] // 2])

    df = {
        "label": np.arange(1, num_labels),
        "area": stats_array[1:, cv2.CC_STAT_AREA],
        "centroid": [c for c in centroids[1:]],  # keep as array
    }

    df = pd.DataFrame(df)
    # Compute distances to center
    df["distance_to_center"] = df["centroid"].apply(
        lambda c: np.linalg.norm(center - c)
    )

    # Filter and select
    filtered_df = df[df["area"] > 3000]
    closest = filtered_df.loc[filtered_df["distance_to_center"].idxmin()]
    selected_label = closest["label"]

    # Create a binary mask for the selected label
    selected_mask = (labels == selected_label).astype(np.uint8)

    # Apply erosion only to that mask
    eroded_mask = ndimage.binary_erosion(selected_mask, iterations=20)

    # Get coordinates from the eroded mask
    coords = np.column_stack(np.where(eroded_mask))

    sampled_points = coords[np.random.choice(len(coords), 5, replace=False)][:, ::-1]
    selected_centroid = closest["centroid"]
    selected_centroid = np.array(
        [[int(selected_centroid[0]), int(selected_centroid[1])]], dtype=np.int32
    )

    combined_points = np.vstack([sampled_points, selected_centroid])

    return combined_points


def segment_organoid(proj_XY_8bit, model):

    # Temporary directory for saving frames
    temp_dir = tempfile.mkdtemp()

    # Itterate over movie frames
    for i, frame in enumerate(proj_XY_8bit):
        # Convert image to RGB grayscale image
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        # Normalize the frame to 0-255 range
        frame = (frame / frame.max() * 255).astype(np.uint8)

        # Save the frame as a JPEG image
        Image.fromarray(frame).save(os.path.join(temp_dir, f"{i:05d}.jpeg"))

    # Get inference state
    inference_state = model.init_state(temp_dir)

    # Calculate center point where organoid should be
    coords = get_coords(proj_XY_8bit)

    labels = np.ones(len(coords))
    _, object_ids, mask_logits = model.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=coords,
        labels=labels,
    )

    # Get the masks of the organoid
    all_masks = []
    for frame_idx, object_ids, mask_logits in model.propagate_in_video(inference_state):
        masks = (mask_logits > 0.0).cpu().numpy()  # shape: (N, X, H, W)
        N, X, H, W = masks.shape
        masks = masks.reshape(N * X, H, W)
        all_masks.append(masks)

    shutil.rmtree(temp_dir)

    all_masks = np.stack(all_masks, axis=0)  # shape: (T, C, Y, X)
    all_masks = all_masks[:, 0]  # T Y X

    return all_masks
