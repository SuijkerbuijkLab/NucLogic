def generate_points(x_center, y_center, angle_deg, a_radius, b_radius, num_points=359):
    """
    Generates points along an ellipse perimeter.

    Args:
        x_center: X-coordinate of ellipse center
        y_center: Y-coordinate of ellipse center
        angle_deg: Rotation angle in degrees (counterclockwise)
        a_radius: Semi-major axis length
        b_radius: Semi-minor axis length
        num_points: Number of points to generate (default: 359)

    Returns:
        x_points: X-coordinates of points on the ellipse
        y_points: Y-coordinates of points on the ellipse
    """
    import numpy as np

    # Convert angle to radians
    angle_rad = (angle_deg * np.pi) / 180

    # Generate angles around the ellipse
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Generate points on an axis-aligned ellipse
    x_aligned = a_radius * np.cos(theta)
    y_aligned = b_radius * np.sin(theta)

    # Create rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Rotate points
    x_rotated = x_aligned * cos_angle - y_aligned * sin_angle
    y_rotated = x_aligned * sin_angle + y_aligned * cos_angle

    # Translate to center
    x_points = x_rotated + x_center
    y_points = y_rotated + y_center

    return x_points, y_points
