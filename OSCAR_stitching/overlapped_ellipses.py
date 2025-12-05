def overlapped_ellipses(a_A, b_A, x_A, y_A, angle_A, a_B, b_B, x_B, y_B, angle_B):
    """
    Determines the overlap between two ellipses using the focal point method.

    Args:
        a_A, b_A: Semi-major and semi-minor axis lengths of ellipse A
        x_A, y_A: Center coordinates of ellipse A
        angle_A: Rotation angle of ellipse A in degrees
        a_B, b_B: Semi-major and semi-minor axis lengths of ellipse B
        x_B, y_B: Center coordinates of ellipse B
        angle_B: Rotation angle of ellipse B in degrees

    Returns:
        Dictionary containing:
        - 'x_A': X-coordinates of points on ellipse A
        - 'y_A': Y-coordinates of points on ellipse A
        - 'x_B': X-coordinates of points on ellipse B
        - 'y_B': Y-coordinates of points on ellipse B
        - 'AonB': Number of points from A falling within B
        - 'BonA': Number of points from B falling within A
    """
    import numpy as np
    from OSCAR_stitching.generate_points import generate_points

    # ===== Part 1: Check how many points from Ellipse A fall within Ellipse B =====

    # Generate points on ellipse A
    x_A_points, y_A_points = generate_points(x_A, y_A, angle_A, a_A, b_A)

    # Convert angle B to radians
    angle_B_rad = (angle_B * np.pi) / 180

    # Calculate focal distance for ellipse B
    c_B = np.sqrt(np.abs(a_B**2 - b_B**2))

    # Calculate focal points of ellipse B
    f_1_x_B = x_B - c_B * np.cos(angle_B_rad)
    f_1_y_B = y_B - c_B * np.sin(angle_B_rad)
    f_2_x_B = x_B + c_B * np.cos(angle_B_rad)
    f_2_y_B = y_B + c_B * np.sin(angle_B_rad)

    # Count how many points from A fall within B
    q_A = 0
    for k in range(len(x_A_points)):
        # Distance from point to first focal point of B
        dist_1_to_B = np.sqrt(
            (x_A_points[k] - f_1_x_B) ** 2 + (y_A_points[k] - f_1_y_B) ** 2
        )
        # Distance from point to second focal point of B
        dist_2_to_B = np.sqrt(
            (x_A_points[k] - f_2_x_B) ** 2 + (y_A_points[k] - f_2_y_B) ** 2
        )

        # Check if point is inside ellipse B using focal property
        if (dist_1_to_B + dist_2_to_B) <= 2 * a_B:
            q_A += 1

    # ===== Part 2: Check how many points from Ellipse B fall within Ellipse A =====

    # Generate points on ellipse B
    x_B_points, y_B_points = generate_points(x_B, y_B, angle_B, a_B, b_B)

    # Convert angle A to radians
    angle_A_rad = (angle_A * np.pi) / 180

    # Calculate focal distance for ellipse A
    c_A = np.sqrt(np.abs(a_A**2 - b_A**2))

    # Calculate focal points of ellipse A
    f_1_x_A = x_A - c_A * np.cos(angle_A_rad)
    f_1_y_A = y_A - c_A * np.sin(angle_A_rad)
    f_2_x_A = x_A + c_A * np.cos(angle_A_rad)
    f_2_y_A = y_A + c_A * np.sin(angle_A_rad)

    # Count how many points from B fall within A
    q_B = 0
    for k in range(len(x_B_points)):
        # Distance from point to first focal point of A
        dist_1_to_A = np.sqrt(
            (x_B_points[k] - f_1_x_A) ** 2 + (y_B_points[k] - f_1_y_A) ** 2
        )
        # Distance from point to second focal point of A
        dist_2_to_A = np.sqrt(
            (x_B_points[k] - f_2_x_A) ** 2 + (y_B_points[k] - f_2_y_A) ** 2
        )

        # Check if point is inside ellipse A using focal property
        if (dist_1_to_A + dist_2_to_A) <= 2 * a_A:
            q_B += 1

    return {
        "x_A": x_A_points,
        "y_A": y_A_points,
        "x_B": x_B_points,
        "y_B": y_B_points,
        "AonB": q_A,
        "BonA": q_B,
    }
