def draw_3D_ellipses(summary_df, dims, label_ellipses=True):
    """
    Draws 3D ellipses as 2D cross-sections using information from a summary DataFrame.
    """
    import numpy as np
    from OSCAR_stitching.generate_ellipse_coordinates import (
        generate_ellipse_coordinates,
    )

    # Initialize output image (3D labeled stack)
    # FIXED: Numpy uses [Z, Y, X] ordering, so dims should be (Z, Y, X)
    img = np.zeros(dims, dtype=int)

    for _, row in summary_df.iterrows():
        # Extract object properties
        cx, cy, cz = row["Xcenter"], row["Ycenter"], row["Zcenter"]
        a, b = row["A"], row["B"]
        n_slices = row["Vz"]

        # Convert angle from degrees to radians
        ang = np.radians(row["AngleXY"])

        vx, vy = row["Vx"], row["Vy"]
        v_mod = row["C"]

        # Determine object label
        label = int(row["Label"]) if label_ellipses else 1

        # Normalize direction vector
        if v_mod != 0:
            v = np.array([vx, vy]) / v_mod
        else:
            v = np.array([0.0, 0.0])

        # Handle NaN values in direction vector
        v[np.isnan(v)] = 0.0

        # Calculate Z-slice range for this object
        z_half = int(np.floor(n_slices / 2))
        z_start = max(0, int(np.round(cz)) - z_half)
        z_end = min(dims[0], int(np.round(cz)) + z_half)  # FIXED: dims[0] is Z

        # Update n_slices based on actual range
        n_slices_actual = z_end - z_start

        # Draw ellipses for each Z-slice
        for zi in range(z_start, z_end):
            # Distance from object center
            delta_z = zi - cz

            # Scale factor based on ellipsoid geometry
            if n_slices_actual > 0:
                factor = np.sqrt(max(0, 1 - (2 * delta_z / n_slices_actual) ** 2))
            else:
                factor = 1.0

            # Scale axes based on distance from center
            a_scaled = a * factor / 2
            b_scaled = b * factor / 2

            # Shift ellipse center according to trajectory direction
            x_shifted = cx + v[0] * delta_z
            y_shifted = cy + v[1] * delta_z

            # Generate ellipse coordinates
            ellipse_coords = generate_ellipse_coordinates(
                int(np.round(y_shifted)),  # row (Y)
                int(np.round(x_shifted)),  # col (X)
                b_scaled,  # minor axis radius
                a_scaled,  # major axis radius
                ang,
                max_size=[dims[1], dims[2]],  # FIXED: [Y, X] not [Z, Y]
            )

            # Filter valid coordinates
            valid_coords = [
                (r, c)
                for r, c in ellipse_coords
                if 0 <= r < dims[1] and 0 <= c < dims[2]  # FIXED: Y and X dims
            ]

            # FIXED: Draw ellipse with correct indexing [Z, Y, X]
            for r, c in valid_coords:
                img[zi, r, c] = label

    return img
