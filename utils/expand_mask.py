from scipy.ndimage import distance_transform_edt
import napari
import numpy as np


def expand_mask(
    mask,
    dilation_size=1,
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
    return mask
