from scipy.ndimage import distance_transform_edt
import numpy as np


def expand_mask(
    mask,
    dilation_size=1,
):
    mask = np.asarray(mask)

    def _expand_zyx(mask_zyx):
        dilated_zyx = np.zeros_like(mask_zyx)
        for z in range(mask_zyx.shape[0]):
            slice_mask = mask_zyx[z]
            distance, indices = distance_transform_edt(
                slice_mask == 0, return_indices=True
            )
            expanded_slice = slice_mask.copy()
            expand_region = (slice_mask == 0) & (distance <= dilation_size)

            # Assign each background pixel the label of its nearest non-zero neighbor.
            expanded_slice[expand_region] = slice_mask[
                indices[0][expand_region], indices[1][expand_region]
            ]
            dilated_zyx[z] = expanded_slice
        return dilated_zyx

    if mask.ndim == 3:
        dilated = _expand_zyx(mask)
    elif mask.ndim == 4:
        # Expected shape T, Z, Y, X.
        dilated = np.zeros_like(mask)
        for t in range(mask.shape[0]):
            dilated[t] = _expand_zyx(mask[t])
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}; expected ZYX or TZYX.")

    return dilated.astype(mask.dtype)
