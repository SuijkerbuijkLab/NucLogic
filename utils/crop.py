# Function that will crop a movie based on an existing XY max projection using a SAM model

import pandas as pd
import numpy as np
from libpysal.weights import KNN
from PIL import Image
import scipy.ndimage as ndimage
import cv2
from .offset_image import offset_image
from utils.pad_to_shape import pad_to_shape
from utils.load_image import as_numpy, load_image
from utils.save_as_tiff import save_as_tiff
from skimage.filters import threshold_triangle
from alive_progress import alive_bar

import os, shutil, tempfile


def _moran_rows(XZ, w_cache):
    """Vectorized Moran's I for every row of a 2D array.

    Equivalent to running esda.Moran(row, KNN(k=2)) per row (default row-
    standardized transform), but computed for all rows at once: with row-
    standardized weights S0 == n, so I reduces to zᵀWz / zᵀz. The KNN weight
    matrix depends only on the row length, so it is built once per length and
    cached in `w_cache`. Rows with zero variance get I = 0.
    """
    n = XZ.shape[1]
    W = w_cache.get(n)
    if W is None:
        coords = np.arange(n).reshape(-1, 1)
        S = KNN.from_array(coords, k=2).sparse.astype(float)  # binary adjacency
        rs = np.asarray(S.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        W = S.multiply((1.0 / rs)[:, None]).tocsr()           # row-standardized
        w_cache[n] = W
    Z = XZ.astype(float)
    Z -= Z.mean(axis=1, keepdims=True)
    num = (Z * (W @ Z.T).T).sum(axis=1)                       # z · (W z)
    den = (Z * Z).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


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
    proj_XY_8bit = ((proj_XY_nuclei / np.percentile(proj_XY_nuclei, 99.9)) * 255).astype(np.uint8)

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

    # Load the input image lazily; frames are read one at a time in the loop below.
    movie, voxel_size, time_interval, _ = load_image(input_file)

    # Save this tracked movie to the output folder. Z is projected away, so the
    # file carries no Z calibration.
    save_as_tiff(
        os.path.join(os.path.dirname(input_file), f"{name}_projXY_tracked.tif"),
        organoid_only,
        "TCYX",
        (1.0, voxel_size[1], voxel_size[2]),
        time_interval,
    )

    timepoints = proj_XY.shape[0]

    cropped_movie = []
    max_dims = [0, 0, 0, 0]
    w_cache = {}  # KNN spatial weights reused across rows/frames (keyed by row length)

    # Loop over every frame in the movie to crop that frame.
    with alive_bar(timepoints, title="Croppping frames") as bar:
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
                        as_numpy(movie[frame, ch, :, row_min:row_max, col_min:col_max])
                        for ch in nuclei_list
                    ]
                ),
                axis=0,
            )

            # Get a bounding box cropped version of the organoid
            ref_crop = maskXY[row_min:row_max, col_min:col_max]

            # Use the XY mask on the original movie to crop as a bounding box and
            # black out corners where there is no organoid (broadcast over Z).
            ref_masked = np.where(ref_crop[np.newaxis, :, :] > 0, ref, 0).astype(np.uint16)

            # Make an max XZ projection that we will use to see what Z slices are important
            XZ = np.max(ref_masked, axis=1)

            # Apply offset to enhance contrast and make background and signal more distinct
            XZ = offset_image(XZ, type="median")

            # Moran's I per row (vectorized; zero-variance rows -> 0)
            moran_values = np.nan_to_num(_moran_rows(XZ, w_cache), nan=0.0)

            # Dynamic threshold based on the max Moran I found, then keep those rows
            threshold = 0.4 * moran_values.max()
            z_list = np.nonzero(moran_values >= threshold)[0].tolist()

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
            cropped_image = as_numpy(
                movie[frame, :, slice_min:slice_max, row_min:row_max, col_min:col_max]
            )

            # On this cropped frame, we again need to make it so that the pixels in the corners where no organoid is are actually black
            mask_crop = maskXY[
                row_min:row_max, col_min:col_max
            ]  # shape: (Y_crop, X_crop)
            # Apply the mask, broadcasting (Y,X) over C and Z of the cropped image
            cropped_image_masked = np.where(
                mask_crop[np.newaxis, np.newaxis, :, :], cropped_image, 0
            ).astype(np.uint16)
            cropped_image_masked = np.transpose(
                cropped_image_masked, (1, 0, 2, 3)
            )  # Z C Y X for tiff

            cropped_movie.append(cropped_image_masked)
            for i in range(4):
                max_dims[i] = max(max_dims[i], cropped_image_masked.shape[i])

            bar()

    cropped_movie = [pad_to_shape(stack, max_dims) for stack in cropped_movie]
    cropped_movie = np.stack(cropped_movie, axis=0)

    return cropped_movie


def central_organoid_mask(frame_8bit):
    """Classical segmentation of the central organoid in a single 2D frame.

    Returns (selected_mask, centroid_xy) for the connected component closest to the
    image center (area > 3000), or (None, None) if nothing suitable is found. Shared
    by the frame-0 point prompt (get_coords) and the keyframe re-prompts in
    segment_organoid so both describe the same object the same way.
    """
    if frame_8bit.max() == 0:
        return None, None

    # Normalize and convert to 8-bit
    frame = ((frame_8bit / frame_8bit.max()) * 255).astype(np.uint8)
    frame = ndimage.gaussian_filter(frame, sigma=3)

    # Apply threshold (you can tweak the value)
    threshold_value = threshold_triangle(frame)
    _, binary_mask = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)

    binary_mask = ndimage.binary_dilation(binary_mask, iterations=10)
    filled_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    num_labels, labels, stats_array, centroids = cv2.connectedComponentsWithStats(
        filled_mask
    )
    if num_labels <= 1:
        return None, None

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
    if filtered_df.empty:
        return None, None
    closest = filtered_df.loc[filtered_df["distance_to_center"].idxmin()]
    selected_label = closest["label"]

    # Create a binary mask for the selected label
    selected_mask = (labels == selected_label).astype(np.uint8)

    return selected_mask


def segment_organoid(proj_XY_8bit, model, drop_frac=0.30, max_repairs=100):

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
    selected_mask = central_organoid_mask(proj_XY_8bit[0])
    if selected_mask is None or selected_mask.sum() == 0:
        raise ValueError("No central organoid found in frame 0 to seed SAM2")

    _, object_ids, mask_logits = model.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=selected_mask.astype(bool),   # (H, W) bool, matches the saved frames
    )

    T = len(proj_XY_8bit)

    def run_propagation(start=0):
        # Propagate from `start` forward; return {frame_idx: 2D bool mask of obj 1}
        out = {}
        for frame_idx, object_ids, mask_logits in model.propagate_in_video(
            inference_state, start_frame_idx=start
        ):
            masks = (mask_logits > 0.0).cpu().numpy()  # shape: (N, X, H, W)
            out[frame_idx] = masks.reshape(-1, masks.shape[-2], masks.shape[-1])[0]
        return out

    # Initial forward pass
    masks = run_propagation(0)

    # Drop-triggered re-prompting: when the area suddenly collapses (> drop_frac vs
    # the immediately previous frame), SAM has lost part of the organoid. Re-anchor
    # that frame with SAM's own last good mask (never the classical threshold mask,
    # which fails when organoids touch) and re-propagate the tail. SAM's refined
    # output for the re-anchored frame can still show the drop, so the search always
    # resumes after the repaired frame: each frame is repaired at most once, instead
    # of the same frame being retried until max_repairs.
    search_from = 1
    for _ in range(max_repairs):
        drop_t = next(
            (
                t
                for t in range(search_from, T)
                if masks[t - 1].sum() > 0
                and masks[t].sum() < (1 - drop_frac) * masks[t - 1].sum()
            ),
            None,
        )
        if drop_t is None:
            break

        good_mask = masks[drop_t - 1]  # SAM's own pre-drop mask
        if good_mask.sum() == 0:
            break
        model.add_new_mask(
            inference_state=inference_state,
            frame_idx=drop_t,
            obj_id=1,
            mask=good_mask.astype(bool),
        )
        masks.update(run_propagation(drop_t))  # redo tail, overwrite downstream
        search_from = drop_t + 1

    shutil.rmtree(temp_dir)

    all_masks = np.stack([masks[t] for t in range(T)], axis=0)  # T Y X

    return all_masks
