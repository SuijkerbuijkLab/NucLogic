def fit_3D_line(x, y, z=None, points_to_fit=None):
    """
    Fits a 3D line to X, Y, Z coordinates using linear regression.

    Args:
        x: Array of X-coordinates
        y: Array of Y-coordinates
        z: Array of Z-coordinates (default: sequential 1, 2, 3, ...)
        points_to_fit: Number of points to use for fitting (default: all points)

    Returns:
        Dictionary containing:
        - xpred: Predicted X values along the fitted line
        - ypred: Predicted Y values along the fitted line
        - xcenter: X-coordinate of line center
        - ycenter: Y-coordinate of line center
        - zcenter: Z-coordinate of line center
        - xref: Predicted X at next Z-position (extrapolation)
        - yref: Predicted Y at next Z-position (extrapolation)
        - betaX: Regression coefficients for X [intercept, slope]
        - betaY: Regression coefficients for Y [intercept, slope]
        - vx: X-component of direction vector
        - vy: Y-component of direction vector
        - vz: Z-component of direction vector
        - vmodule: Magnitude of direction vector
    """
    import numpy as np
    from OSCAR_stitching.linear_regression import linear_regression

    # Convert inputs to numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Default Z values: sequential from 1 to length(x)
    if z is None:
        z = np.arange(1.0, len(x) + 1.0)
    else:
        z = np.array(z, dtype=float)

    # Default: fit all points
    if points_to_fit is None:
        points_to_fit = len(x)

    # Create design matrix: [ones, z values]
    # This allows fitting: x = beta0 + beta1 * z
    z_matrix = np.column_stack([np.ones(len(z)), z])

    # Perform linear regression for X and Y separately
    regX = linear_regression(x, z_matrix, points_to_fit=points_to_fit)
    regY = linear_regression(y, z_matrix, points_to_fit=points_to_fit)

    # Get predicted values
    xpred = regX["ypredicted"]
    ypred = regY["ypredicted"]

    # Calculate center coordinates
    xcenter = (xpred[0] + xpred[-1]) / 2
    ycenter = (ypred[0] + ypred[-1]) / 2
    zcenter = (z[0] + z[-1]) / 2

    # Get regression coefficients
    betaX = regX["beta"]
    betaY = regY["beta"]

    # Calculate reference point (extrapolation to next Z)
    # xref = beta0 + beta1 * (z_last + 1)
    xref = betaX[1] * (z[-1] + 1) + betaX[0]
    yref = betaY[1] * (z[-1] + 1) + betaY[0]

    # Calculate direction vector components
    vx = xpred[-1] - xpred[0]
    vy = ypred[-1] - ypred[0]
    vz = z[-1] - z[0] + 1

    # Calculate vector magnitude
    vmodule = np.sqrt(vx**2 + vy**2 + vz**2)

    return {
        "xpred": xpred,
        "ypred": ypred,
        "xcenter": xcenter,
        "ycenter": ycenter,
        "zcenter": zcenter,
        "xref": xref,
        "yref": yref,
        "betaX": betaX,
        "betaY": betaY,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "vmodule": vmodule,
    }
