def linear_regression(
    y,
    x_matrix=None,
    points_to_fit=None,
):
    """
    Performs linear regression: y = X * beta

    Args:
        y: Dependent variable (1D array)
        x_matrix: Design matrix (2D array with intercept column)
        points_to_fit: Number of points to use for fitting

    Returns:
        Dictionary with:
        - beta: Regression coefficients [intercept, slope]
        - ypredicted: Predicted values
        - rsquared: R-squared value
    """
    import numpy as np

    y = np.array(y, dtype=float)
    if x_matrix is None:
        x_matrix = np.column_stack([np.ones(len(y)), np.arange(1, len(y) + 1)])
    else:
        x_matrix = np.array(x_matrix, dtype=float)

    if points_to_fit is None:
        points_to_fit = len(y)

    # Use subset for fitting if specified
    if points_to_fit != len(y):
        y_fit = y[:points_to_fit]
        x_fit = x_matrix[:points_to_fit, :]
    else:
        y_fit = y
        x_fit = x_matrix

    # Calculate regression coefficients: beta = (X'X)^-1 * X'y
    beta = np.linalg.solve(x_fit.T @ x_fit, x_fit.T @ y_fit)

    # Predict for all points (not just fitting subset)
    ypredicted = x_matrix @ beta

    # Calculate R-squared
    ss_res = np.sum((y - ypredicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    rsquared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {"beta": beta, "ypredicted": ypredicted, "y": y, "rsquared": rsquared}
