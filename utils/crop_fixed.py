import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import tifffile
from esda.moran import Moran
from imaris_ims_file_reader.ims import ims
from libpysal.weights import KNN

from .offset_image import offset_image


def crop_fixed(
    proj_XY,
    input_file,
    output_directory,
    nuclei=2,
    name="projXY_tracked",
    voxel_size=(1.0, 1.0, 1.0),
    manual=False,
    dual_nuclei=False,
):
    # Create the output directory to save the files
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not manual:
        # Apply gaussian filter to smooth the image and make thresholding more robust
        frame = ndimage.gaussian_filter(proj_XY, sigma=(3, 3))

        # Apply threshold (you can tweak the value)
        median = np.median(frame)
        _, binary_mask = cv2.threshold(frame, median + 5, 255, cv2.THRESH_BINARY)

        binary_mask = ndimage.binary_dilation(binary_mask, iterations=5)

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

        organoid_only = np.where(selected_mask > 0, proj_XY, 0)

    else:
        # Normalize for the CV2 roi selector
        normalized_image = cv2.normalize(proj_XY, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Get polygon selection
        selected_mask, _, _ = polygon_selection(normalized_image)

        organoid_only = np.where(selected_mask > 0, proj_XY, 0)

    # Save this tracked movie to the output folder
    tifffile.imwrite(
        os.path.join(
            os.path.dirname(output_directory),
            f"{name}_projXY_tracked.tif",
        ),
        organoid_only,
        compression="zlib",
        compressionargs={"level": 8},
    )

    # Find the XY coordinates of the mask
    maskXY = selected_mask > 0
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

    # Load in the input image, which is either a ims or tiff
    if input_file.endswith(".ims"):
        movie = ims(input_file)  # T,C,Z,Y,X
    elif input_file.endswith(".tif"):
        movie = tifffile.imread(input_file)  # T,Z,C,Y,X
        movie = np.transpose(movie, (0, 2, 1, 3, 4))  # T,C,Z,Y,X

    # Get a bounding box cropped version of the organoid
    if not dual_nuclei:
        ref = movie[:, nuclei, :, row_min:row_max, col_min:col_max]
    else:
        ref = movie[:, nuclei, :, row_min:row_max, col_min:col_max]
        ref = np.max(ref, axis=1)  # Combine multiple nuclei channels by max projection

    ref_crop = selected_mask[row_min:row_max, col_min:col_max]

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
    cropped_image = movie[:, :, slice_min:slice_max, row_min:row_max, col_min:col_max]

    # On this cropped frame, we again need to make it so that the pixels in the corners where no organoid is are actually black
    mask_crop = maskXY[row_min:row_max, col_min:col_max]  # shape: (Y_crop, X_crop)
    # Expand mask to 4D: (C, Z, Y_crop, X_crop)
    mask_expanded = mask_crop[np.newaxis, np.newaxis, :, :]
    # Repeat over C and Z to match cropped_image shape
    mask_expanded = np.repeat(mask_expanded, cropped_image.shape[0], axis=0)  # C
    mask_expanded = np.repeat(mask_expanded, cropped_image.shape[1], axis=1)  # Z
    # Apply the mask
    cropped_image_masked = np.where(mask_expanded, cropped_image, 0).astype(np.uint16)

    cropped_image_masked = np.transpose(
        cropped_image_masked, (1, 0, 2, 3)
    )  # Z C Y X for tiff

    # Save this final cropped frame in the output folder
    tifffile.imwrite(
        os.path.join(output_directory, f"Frame-0.tif"),
        cropped_image_masked,
        bigtiff=True,
        resolution=(
            (1 / voxel_size[1]) * 25400,
            (1 / voxel_size[2]) * 25400,
        ),
        metadata={
            "unit": "um",
            "axes": "ZCYX",
            "PhysicalSizeX": voxel_size[2],
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": voxel_size[1],
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": voxel_size[0],
            "PhysicalSizeZUnit": "µm",
            "spacing": voxel_size[0],
        },
        compression="zlib",
        compressionargs={"level": 8},
    )


def polygon_selection(image, display_size=1024):
    """
    Allow user to select a polygon region by clicking points.
    Press 'c' to complete the selection.
    The image is resized for display, but the mask is created on the original resolution.

    Args:
        image: Original image (can be any size)
        display_size: Target size for display (default 1024)
    """
    # Store original dimensions
    original_h, original_w = image.shape[:2]

    # Calculate scaling factor to fit within display_size while maintaining aspect ratio
    scale = min(display_size / original_w, display_size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Make a copy and normalize for display
    draw_image = image.copy()
    p_low, p_high = np.percentile(draw_image, [0, 99])  # Clip top 1%
    draw_image = np.clip(draw_image, p_low, p_high)
    draw_image = cv2.normalize(draw_image, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # Resize for display
    draw_image_resized = cv2.resize(
        draw_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )

    points = []

    # Mouse callback function
    def draw_polygon(event, x, y, flags, param):
        nonlocal draw_image_resized

        # Left button click - add point
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw a small circle at clicked point
            cv2.circle(
                draw_image_resized,
                center=(x, y),
                radius=3,
                color=(255, 255, 0),
                thickness=-1,
            )

            # Connect lines between points
            if len(points) > 1:
                cv2.line(
                    draw_image_resized,
                    points[-2],
                    points[-1],
                    thickness=1,
                    color=(255, 0, 0),
                )

            # Show the image with selected points
            cv2.imshow("Select Polygon - Press C when done", draw_image_resized)

    # Create window and set mouse callback
    cv2.imshow("Select Polygon - Press C when done", draw_image_resized)
    cv2.setMouseCallback("Select Polygon - Press C when done", draw_polygon)

    # Wait for key press
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Press 'c' to complete
            break

    # Close the window
    cv2.destroyAllWindows()

    # Create mask from polygon if we have enough points
    if len(points) > 2:
        # Scale points back to original image coordinates
        points_original = [(int(x / scale), int(y / scale)) for x, y in points]

        # Create mask on original image dimensions
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        points_array = np.array([points_original], dtype=np.int32)
        cv2.fillPoly(mask, points_array, 255)

        # Apply mask to extract region from original image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Get bounding box of non-zero region (for cropping)
        x, y, w, h = cv2.boundingRect(mask)
        cropped = result[y : y + h, x : x + w]

        return mask, result, cropped
    else:
        print("Not enough points selected for polygon")
        return None, None, None
