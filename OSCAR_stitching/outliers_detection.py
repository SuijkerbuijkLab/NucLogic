def outliers_detection(data_vector, slice_vector, med_size_min):
    """
    Detects outliers in a data vector using linear regression and confidence intervals.

    Args:
        data_vector: The data to analyze for outliers
        slice_vector: Indices corresponding to data points
        med_size_min: Number of initial points to use for regression baseline

    Returns:
        Tuple of:
        - H: Vector marking outliers (negative) vs inliers (positive)
        - y1: Filtered data excluding outliers
        - cutPoints: Starting indices of outlier regions
        - returnPoints: Ending indices of outlier regions
    """
    import numpy as np
    from OSCAR_stitching.linear_regression import linear_regression

    # Subset for initial regression
    a = data_vector[:med_size_min]

    # Check if regression is possible
    z_test = np.column_stack([np.ones(len(a)), np.arange(1, len(a) + 1)])

    if np.linalg.det(z_test.T @ z_test) != 0:
        # Fit linear regression to establish baseline
        reg = linear_regression(data_vector, points_to_fit=med_size_min)

        y_predicted = reg["ypredicted"]
        y = reg["y"]

        # Use interquartile range as robust estimate of spread
        sigma = abs(
            np.percentile(y[:med_size_min], 25) - np.percentile(y[:med_size_min], 75)
        )

        # Calculate confidence intervals
        sup_CI = y_predicted + sigma
        down_CI = y_predicted - sigma
    else:
        # Fallback: use median and IQR
        y = data_vector
        sigma = abs(
            np.percentile(y[:med_size_min], 25) - np.percentile(y[:med_size_min], 75)
        )

        y_predicted = np.ones(len(y)) * np.median(y[:med_size_min])
        sup_CI = y_predicted + sigma
        down_CI = y_predicted - sigma

    # Mark outliers
    H = slice_vector.copy().astype(float)
    y1 = []

    for point in range(len(y)):
        b = point  # Index in predicted values

        # Check if point is outside confidence interval
        if y[point] > sup_CI[b] or y[point] < down_CI[b]:
            H[point] = -H[point]  # Mark as outlier (negative)
        else:
            y1.append(data_vector[point])

    # Find contiguous outlier regions
    cutPoints = []
    returnPoints = []

    i = 0
    while i < len(H):
        if H[i] < 0:  # Start of outlier region
            cutPoints.append(i)

            # Find end of outlier region
            j = i
            while j < len(H) - 1 and H[j + 1] < 0:
                j += 1

            returnPoints.append(j)
            i = j + 1
        else:
            i += 1

    return H, np.array(y1), np.array(cutPoints), np.array(returnPoints)
