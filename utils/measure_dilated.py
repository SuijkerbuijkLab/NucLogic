from scipy.ndimage import distance_transform_edt
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from utils.properties_channel import properties_channel


def measure_dilated(
    mask,
    dilation_size=1,
    channels=[0, 1, 2, 3],
    channel_names=["mTmG", "DAPI", "AldoB", "Lyz"],
):
    dilated = np.zeros_like(mask)

    for z in range(mask.shape[0]):
        slice_mask = mask[z]
        distance, indices = distance_transform_edt(
            slice_mask == 0, return_indices=True  # Find distances from background
        )
        expanded_slice = slice_mask.copy()
        expand_region = (slice_mask == 0) & (distance <= dilation_size)

        # Use indices to find which label each background pixel is nearest to
        expanded_slice[expand_region] = slice_mask[
            indices[0][expand_region], indices[1][expand_region]
        ]

        dilated[z] = expanded_slice

    # If you need to preserve the original dtype
    mask = dilated.astype(mask.dtype)

    props = regionprops(mask)

    props_mask = []
    # For every mask found, get the label, centeroid, boundingbox, and volume
    for prop in props:
        label = prop.label
        centroid = prop.centroid  # (z, y, x)
        bounding_box = prop.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
        volume = prop.area

        width_x = bounding_box[5] - bounding_box[2]
        width_y = bounding_box[4] - bounding_box[1]
        height_pixel = bounding_box[3] - bounding_box[0]
        height = (bounding_box[3] - bounding_box[0]) * 8.125
        aspect_ratio = height / np.mean([width_x, width_y])

        props_mask.append(
            {
                "label": label,
                "z_center": centroid[0],
                "y_center": centroid[1],
                "x_center": centroid[2],
                "bounding_box": bounding_box,
                "width_x": width_x,
                "width_y": width_y,
                "height_pixel": height_pixel,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "volume": volume,
            }
        )
    props_mask = pd.DataFrame(props_mask)

    if len(channels) > 0:
        chan0 = properties_channel(mask, channels[0], type="DAPI")
        final_df = props_mask.merge(chan0, on="label")
    if len(channels) > 1:
        chan1 = properties_channel(mask, channels[1], type="mTmG")
        final_df = final_df.merge(chan1, on="label")
    if len(channels) > 2:
        chan2 = properties_channel(mask, channels[2], type="AldoB")
        final_df = final_df.merge(chan2, on="label")
    if len(channels) > 3:
        chan3 = properties_channel(mask, channels[3], type="Lyz")
        final_df = final_df.merge(chan3, on="label")

    final_df["log_ratio_lyz_dapi"] = np.log(final_df["raw_Lyz"] / final_df["raw_DAPI"])
    final_df["log_ratio_aldob_dapi"] = np.log(
        final_df["raw_AldoB"] / final_df["raw_DAPI"]
    )

    return final_df, mask
