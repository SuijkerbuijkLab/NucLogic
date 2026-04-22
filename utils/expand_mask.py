from scipy.ndimage import distance_transform_edt
import numpy as np


def expand_mask_3d(
    mask,
    dilation_size_um=1.0,
    voxel_size=(1.0, 1.0, 1.0),  # (z, y, x) in micrometers
):
    mask = np.asarray(mask)

    def _expand_zyx(mask_zyx):
        # Background voxels are True, labeled voxels are False
        bg = mask_zyx == 0

        # 3D EDT in physical units (um), plus nearest labeled voxel indices
        distance, indices = distance_transform_edt(
            bg,
            sampling=voxel_size,
            return_indices=True,
        )

        expanded = mask_zyx.copy()
        expand_region = bg & (distance <= dilation_size_um)

        # Nearest labeled voxel for each expand_region voxel
        expanded[expand_region] = mask_zyx[
            indices[0][expand_region],
            indices[1][expand_region],
            indices[2][expand_region],
        ]
        return expanded

    if mask.ndim == 3:  # Z,Y,X
        out = _expand_zyx(mask)
    elif mask.ndim == 4:  # T,Z,Y,X
        out = np.empty_like(mask)
        for t in range(mask.shape[0]):
            out[t] = _expand_zyx(mask[t])
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}; expected ZYX or TZYX.")

    return out.astype(mask.dtype)
