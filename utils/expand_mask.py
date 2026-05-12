from scipy.ndimage import distance_transform_edt
import numpy as np
import math


def expand_mask_3d(
    mask,
    dilation_size_um=1.0,
    voxel_size=(1.0, 1.0, 1.0),
):
    mask = np.asarray(mask)

    if mask.ndim == 3:
        return _expand_zyx(mask, dilation_size_um, voxel_size).astype(mask.dtype)
    elif mask.ndim == 4:
        out = np.empty_like(mask)
        for t in range(mask.shape[0]):
            out[t] = _expand_zyx(mask[t], dilation_size_um, voxel_size)
        return out.astype(mask.dtype)
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}; expected ZYX or TZYX.")


def _expand_zyx(mask_zyx, dilation_size_um, voxel_size, tile_z=20):
    z_total = mask_zyx.shape[0]
    # How many Z-slices the dilation radius spans — needed as overlap between tiles
    z_overlap = max(1, math.ceil(dilation_size_um / voxel_size[0]))

    expanded = mask_zyx.copy()

    for z_start in range(0, z_total, tile_z):
        z_end = min(z_start + tile_z, z_total)

        # Extend the tile by z_overlap on each side so EDT sees enough context
        z0 = max(0, z_start - z_overlap)
        z1 = min(z_total, z_end + z_overlap)

        chunk = mask_zyx[z0:z1]
        if not np.any(chunk):
            continue

        bg = chunk == 0
        distance, indices = distance_transform_edt(bg, sampling=voxel_size, return_indices=True)

        # Work only on the non-overlap centre of this tile
        zl0 = z_start - z0
        zl1 = z_end - z0

        local_expand = bg[zl0:zl1] & (distance[zl0:zl1] <= dilation_size_um)
        if not np.any(local_expand):
            continue

        ez, ey, ex = np.where(local_expand)

        # indices[0] is chunk-local Z of the nearest labeled voxel → add z0 for global Z
        iz_global = indices[0, zl0:zl1][local_expand] + z0
        iy        = indices[1, zl0:zl1][local_expand]
        ix        = indices[2, zl0:zl1][local_expand]

        expanded[ez + z_start, ey, ex] = mask_zyx[iz_global, iy, ix]

    return expanded
