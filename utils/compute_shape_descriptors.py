import numpy as np
import pandas as pd
from skimage.measure import regionprops, marching_cubes, mesh_surface_area

# Derived shape descriptors handled by this module. These are NOT plain
# regionprops attributes: they are computed from the (voxel-size-scaled)
# inertia tensor and, for sphericity, a per-label surface mesh.
DERIVED_SHAPE_DESCRIPTORS = (
    "axis_medial_length",
    "elongation",
    "flatness",
    "anisotropy",
    "sphericity",
)


def _equivalent_ellipsoid_semi_axes(eigvals):
    """Semi-axes a >= b >= c of the uniform ellipsoid whose inertia tensor
    eigenvalues match ``eigvals`` (returned by regionprops in descending order).

    For a uniform solid ellipsoid the second moment about a principal axis is
    ``(other_axis_1**2 + other_axis_2**2) / 5``. Inverting that relation gives
    the semi-axes below. Tiny negative values from discretisation noise are
    clamped to zero before the square root.
    """
    l1, l2, l3 = (float(v) for v in eigvals)
    a = np.sqrt(max(2.5 * (l1 + l2 - l3), 0.0))
    b = np.sqrt(max(2.5 * (l1 - l2 + l3), 0.0))
    c = np.sqrt(max(2.5 * (-l1 + l2 + l3), 0.0))
    return a, b, c


def _sphericity(region, spacing):
    """Mesh-based 3D sphericity: pi**(1/3) * (6V)**(2/3) / A.

    1.0 for a perfect sphere, lower for anything rougher/less round. Returns
    ``np.nan`` for labels too small to build a closed surface mesh.
    """
    volume = float(region.area)  # spacing-aware => physical volume
    try:
        # Pad so the surface closes even when the label touches its bbox edge.
        padded = np.pad(region.image, 1)
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)
        surface_area = mesh_surface_area(verts, faces)
    except (RuntimeError, ValueError):
        return np.nan
    if surface_area <= 0:
        return np.nan
    return (np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0)) / surface_area


def compute_shape_descriptors(mask, voxel_size, descriptors):
    """Compute derived, voxel-size-scaled shape descriptors per label.

    Parameters
    ----------
    mask : ndarray
        Labelled 3D mask (Z, Y, X).
    voxel_size : sequence of float
        Physical voxel size (z, y, x); used as ``spacing`` so lengths are in
        physical units and ratios are corrected for anisotropic voxels.
    descriptors : sequence of str
        Subset of :data:`DERIVED_SHAPE_DESCRIPTORS` to compute.

    Returns
    -------
    pandas.DataFrame
        One row per label with a ``label`` column plus the requested descriptors.
    """
    requested = [d for d in descriptors if d in DERIVED_SHAPE_DESCRIPTORS]
    columns = ["label", *requested]
    if not requested:
        return pd.DataFrame(columns=columns)

    spacing = tuple(float(v) for v in voxel_size)
    need_axes = any(
        d in ("axis_medial_length", "elongation", "flatness", "anisotropy")
        for d in requested
    )
    need_sphericity = "sphericity" in requested

    rows = []
    for region in regionprops(mask, spacing=spacing):
        row = {"label": region.label}

        if need_axes:
            a, b, c = _equivalent_ellipsoid_semi_axes(region.inertia_tensor_eigvals)
            if "axis_medial_length" in requested:
                row["axis_medial_length"] = 2.0 * a  # longest full axis, physical units
            if "elongation" in requested:
                row["elongation"] = (1.0 - b / a) if a > 0 else np.nan
            if "flatness" in requested:
                row["flatness"] = (1.0 - c / b) if b > 0 else np.nan
            if "anisotropy" in requested:
                row["anisotropy"] = (1.0 - c / a) if a > 0 else np.nan

        if need_sphericity:
            row["sphericity"] = _sphericity(region, spacing)

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
