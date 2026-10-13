import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from esda.moran import Moran
from libpysal.weights import KNN
from skimage.filters import threshold_triangle 

from .offset_image import offset_image
from utils.load_image import as_numpy, load_image
from utils.save_as_tiff import save_as_tiff


def crop_fixed(
    proj_XY,
    input_file,
    nuclei=2,
    name="projXY_tracked",
    manual=False,
):
    original_projXY = proj_XY.copy()

    while proj_XY.ndim > 2:
        proj_XY = np.max(
            proj_XY, axis=0
        )  # Collapse channel and or time dimensions if present

    if not manual:
        # Apply gaussian filter to smooth the image and make thresholding more robust
        proj_XY_8bit = ((proj_XY / np.percentile(proj_XY, 99.8)) * 255).astype(np.uint8)
        frame = ndimage.gaussian_filter(proj_XY_8bit, sigma=3)

        # Apply threshold (you can tweak the value)
        threshold_value = threshold_triangle(frame)
        _, binary_mask = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)

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

        while selected_mask.ndim < original_projXY.ndim:
            selected_mask = np.expand_dims(selected_mask, axis=0)
        # Apply mask to original projXY
        organoid_only = np.where(selected_mask > 0, original_projXY, 0)

    else:
        # Normalize for the CV2 roi selector
        normalized_image = cv2.normalize(proj_XY, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Get polygon selection.
        selected_mask, _, _ = polygon_selection(normalized_image)
        if selected_mask is None:
            raise ValueError(
                "Manual cropping was cancelled or no polygon was selected."
            )

        while selected_mask.ndim < original_projXY.ndim:
            selected_mask = np.expand_dims(selected_mask, axis=0)
        # Apply mask to original projXY
        organoid_only = np.where(selected_mask > 0, original_projXY, 0)

    # Load the input image; load_image normalises every format to T,C,Z,Y,X.
    movie_tczyx, voxel_size, time_interval, _ = load_image(input_file)
    print(f"Original movie shape: {movie_tczyx.shape}")

    # Save this tracked movie to the output folder. Z is projected away, so the
    # file carries no Z calibration.
    save_as_tiff(
        os.path.join(os.path.dirname(input_file), f"{name}_projXY_tracked.tif"),
        organoid_only,
        "TCYX",
        (1.0, voxel_size[1], voxel_size[2]),
        time_interval,
    )

    # Find the XY coordinates of the mask
    selected_mask = np.squeeze(selected_mask)  # Should be (Y, X)
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

    # Get a bounding box cropped version of the organoid.
    # Do channel slicing one-by-one because ims reader does not support list-based
    # indexing on the channel axis (e.g. movie[:, [0, 1], ...]).
    if isinstance(nuclei, (list, tuple, np.ndarray)):
        nuclei_list = [int(ch) for ch in nuclei]
    else:
        nuclei_list = [int(nuclei)]

    if not nuclei_list:
        raise ValueError(
            "No nuclei channel configured. Please mark at least one channel as 'Nuclei marker'."
        )

    channel_crops = []
    for ch in nuclei_list:
        if ch >= movie_tczyx.shape[1]:
            raise ValueError(
                f"Configured nuclei channel index {ch} is out of bounds for movie with {movie_tczyx.shape[1]} channels."
            )
        ch_crop = as_numpy(movie_tczyx[:, ch, :, row_min:row_max, col_min:col_max])
        channel_crops.append(ch_crop)

    ref = np.max(np.stack(channel_crops, axis=1), axis=1)  # (T, Z, Y, X)

    ref_crop = selected_mask[
        row_min:row_max, col_min:col_max
    ]  # Should be (Y, X) or (1, 1, Y, X)
    print(f"ref crop shape: {ref_crop.shape}")

    ref_expanded = ref_crop[np.newaxis, :, :]
    print(f"ref expanded shape before repeat: {ref_expanded.shape}")
    ref_expanded = np.repeat(ref_expanded, ref.shape[0], axis=0)

    print(f"ref expanded shape: {ref_expanded.shape}, ref shape {ref.shape}")

    # Use the this XYZ mask on the original movie to crop as both a bounding box and to make all pixels (corners) where there is no organoid actually black
    ref_masked = np.where(ref_expanded > 0, ref, 0).astype(np.uint16)

    # Make an XZ projection used to determine relevant Z slices.
    # ref_masked is (T, Z, Y, X) -> collapse T and Y to get (Z, X).
    XZ = np.max(ref_masked, axis=(0, 2))
    if XZ.ndim == 1:
        XZ = XZ[np.newaxis, :]

    # Apply offset to enhance contrast and make background and signal more distinct
    XZ = offset_image(XZ, type="median")

    z_list = []
    moran_values = []  # Collect Moran’s I for all rows

    # Step 1: Calculate Moran's I for each row
    for idx, row in enumerate(XZ):
        row = np.atleast_1d(row)
        if row.size < 2 or np.std(row) == 0:
            moran_values.append((idx, 0))
            continue
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
    if not moran_values:
        # Fallback to full Z-range if Moran computation yielded no usable values.
        slice_min, slice_max = 0, ref.shape[1]
    else:
        max_moran = max(val for _, val in moran_values)
        threshold = 0.4 * max_moran

        # Step 3: Filter Z slices using threshold
        z_list = [idx for idx, val in moran_values if val >= threshold]

        if not z_list:
            slice_min, slice_max = 0, ref.shape[1]
        else:
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
            slice_max = min(max(max_block) + 2, ref.shape[1])

    # This time, crop the frame of the movie on all channels using corrext XYZ
    cropped_image = as_numpy(
        movie_tczyx[:, :, slice_min:slice_max, row_min:row_max, col_min:col_max]
    )

    # On this cropped frame, make pixels outside the organoid black.
    # cropped_image has shape (T, C, Z, Y, X).
    mask_crop = maskXY[row_min:row_max, col_min:col_max]  # (Y_crop, X_crop)
    mask_expanded = mask_crop[np.newaxis, np.newaxis, np.newaxis, :, :]  # (1,1,1,Y,X)
    mask_expanded = np.repeat(mask_expanded, cropped_image.shape[0], axis=0)  # T
    mask_expanded = np.repeat(mask_expanded, cropped_image.shape[1], axis=1)  # C
    mask_expanded = np.repeat(mask_expanded, cropped_image.shape[2], axis=2)  # Z
    cropped_image_masked = np.where(mask_expanded, cropped_image, 0).astype(np.uint16)

    cropped_image_masked = np.transpose(
        cropped_image_masked, (0, 2, 1, 3, 4)
    )  # Z,C,Y,X

    return cropped_image_masked


def _matplotlib_polygon(image, snap_px=10):
    """Click points; the outline is drawn as you go. Undo/Backspace removes the
    last point, clicking near the first point closes the polygon, Done/Enter
    finishes. Returns the (x, y) points in image pixel coordinates."""
    from matplotlib.widgets import Button

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.1)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    (outline,) = ax.plot([], [], "-o", color="yellow", markersize=4, linewidth=1)
    points = []
    closed = False

    def redraw():
        xy = points + points[:1] if closed else points
        outline.set_data([p[0] for p in xy], [p[1] for p in xy])
        ax.set_title(
            "Outline closed: press Done (or Undo to edit)" if closed
            else "Click around the organoid; click the first point to close"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal closed
        # Ignore clicks outside the image, non-left clicks, and toolbar zoom/pan.
        if event.inaxes is not ax or event.button != 1 or closed:
            return
        if getattr(fig.canvas.toolbar, "mode", ""):
            return
        # Snap: a click on an existing point never adds a new one; on the first
        # point (with a polygon's worth of points) it closes the outline.
        for i, p in enumerate(points):
            px, py = ax.transData.transform(p)
            if np.hypot(px - event.x, py - event.y) <= snap_px:
                if i == 0 and len(points) > 2:
                    closed = True
                    redraw()
                return
        points.append((int(round(event.xdata)), int(round(event.ydata))))
        redraw()

    def undo(_=None):
        nonlocal closed
        if closed:
            closed = False
        elif points:
            points.pop()
        redraw()

    def done(_=None):
        fig.canvas.stop_event_loop()

    def on_key(event):
        if event.key in ("backspace", "ctrl+z"):
            undo()
        elif event.key == "enter":
            done()

    undo_button = Button(fig.add_axes([0.52, 0.02, 0.2, 0.05]), "Undo last point")
    done_button = Button(fig.add_axes([0.75, 0.02, 0.15, 0.05]), "Done")
    undo_button.on_clicked(undo)
    done_button.on_clicked(done)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", done)

    redraw()
    fig.show()
    fig.canvas.start_event_loop(timeout=-1)  # blocks until Done, Enter or window close
    plt.close(fig)
    return points


def polygon_selection(image, display_size=1024):
    """
    Allow user to select a polygon region by clicking points (see _matplotlib_polygon).
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

    # matplotlib, not OpenCV HighGUI: cellpose installs headless OpenCV, which has no windows.
    points = _matplotlib_polygon(draw_image_resized)

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
