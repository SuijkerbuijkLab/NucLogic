def generate_ellipse_coordinates(
    row, col, r_radius, c_radius, rotation=0.0, max_size=None
):
    """
    Generates pixel coordinates for a 2D ellipse.

    Args:
        row: Center row (Y-coordinate)
        col: Center column (X-coordinate)
        r_radius: Radius along row axis (minor/major depending on rotation)
        c_radius: Radius along column axis
        rotation: Rotation angle in radians
        max_size: Optional [height, width] to clip coordinates

    Returns:
        List of (row, col) tuples representing ellipse pixels
    """

    import numpy as np

    # Normalize rotation to [0, π)
    rotation = rotation % np.pi
    if rotation < 0:
        rotation += np.pi

    # Calculate rotated bounding box
    r_radius_rot = abs(r_radius * np.cos(rotation)) + abs(c_radius * np.sin(rotation))
    c_radius_rot = abs(r_radius * np.sin(rotation)) + abs(c_radius * np.cos(rotation))

    # Bounding box limits
    upper_left = [int(np.ceil(row - r_radius_rot)), int(np.ceil(col - c_radius_rot))]
    lower_right = [int(np.floor(row + r_radius_rot)), int(np.floor(col + c_radius_rot))]

    # Generate grid of candidate points
    r_range = np.arange(upper_left[0], lower_right[0] + 1)
    c_range = np.arange(upper_left[1], lower_right[1] + 1)

    # Calculate distances from center
    r_grid, c_grid = np.meshgrid(r_range - row, c_range - col, indexing="ij")

    # Apply rotation and ellipse equation
    cos_a, sin_a = np.cos(rotation), np.sin(rotation)

    if r_radius > 0 and c_radius > 0:
        distances = ((r_grid * cos_a + c_grid * sin_a) / r_radius) ** 2 + (
            (r_grid * sin_a - c_grid * cos_a) / c_radius
        ) ** 2
    else:
        return []

    # Select pixels inside ellipse
    inside_ellipse = distances < 1
    rows_inside, cols_inside = np.where(inside_ellipse)

    # Convert to absolute coordinates
    ellipse_coords = [
        (r_range[r], c_range[c]) for r, c in zip(rows_inside, cols_inside)
    ]

    # Filter by image bounds if provided
    if max_size is not None:
        ellipse_coords = [
            (r, c)
            for r, c in ellipse_coords
            if 0 <= r < max_size[0] and 0 <= c < max_size[1]
        ]

    return ellipse_coords
