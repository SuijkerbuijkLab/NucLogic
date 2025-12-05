def terminator_returns(dist_vector, angle_vector, n_points):
    """
    Determines termination criteria based on distance and angle outlier analysis.

    This function detects outliers in trajectory distance and angle data, then
    identifies regions where the object may have been incorrectly merged with another.

    Args:
        dist_vector: Vector of distances from fitted line (typically dist[1:] to skip first)
        angle_vector: Vector of angle cosines between trajectory segments
        n_points: Threshold value (typically med_z) for dividing "in" vs "out" regions

    Returns:
        Dictionary containing:
        - 'Hdist': Binary vector marking distance outliers (1) vs inliers (0)
        - 'Hvect': Binary vector marking angle outliers (1) vs inliers (0)
        - 'Htotal': Combined outlier metric (0, 1, or 2)
        - 'numbIn': Number of outliers in first n_points slices
        - 'numbOut': Number of outliers after n_points slices
        - 'cutOut': Indices where outlier regions start
        - 'lengthOut': Lengths of outlier regions
    """
    import numpy as np
    from OSCAR_stitching.outliers_detection import outliers_detection

    # Perform outlier detection on distance data
    HD, yD, cutPointsD, returnPointsD = outliers_detection(
        dist_vector, np.arange(1, len(dist_vector) + 1), n_points
    )

    # Perform outlier detection on angle data
    HV, yV, cutPointsV, returnPointsV = outliers_detection(
        angle_vector, np.arange(1, len(angle_vector) + 1), n_points
    )

    # Convert outlier markers to binary (1 = outlier, 0 = inlier)
    # Original formula: ((H / abs(H)) - 1) / (-2)
    # When H > 0 (inlier): (1 - 1) / (-2) = 0
    # When H < 0 (outlier): (-1 - 1) / (-2) = 1
    HD = ((HD / np.abs(HD)) - 1) / (-2)
    HD = HD.astype(int)

    HV = ((HV / np.abs(HV)) - 1) / (-2)
    HV = HV.astype(int)

    # Combined outlier metric: HT = 0 (no outlier), 1 (one type), or 2 (both types)
    HT = HD + HV

    # Count outliers in different regions
    cin = 0  # Outliers in first n_points slices
    cout = 0  # Outliers after n_points slices
    cutPointsf = []  # Starting indices of outlier regions

    for i in range(len(HT)):
        if HT[i] > 1:  # Both distance AND angle are outliers
            if i < n_points:
                cin += 1
            else:
                cout += 1
                cutPointsf.append(float(i))

    # Calculate lengths of outlier regions
    if len(cutPointsf) > 1:
        lengthSlicesf = np.diff(cutPointsf) - 1
    else:
        lengthSlicesf = np.array([])

    return {
        "Hdist": HD,
        "Hvect": HV,
        "Htotal": HT,
        "numbIn": cin,
        "numbOut": cout,
        "cutOut": np.array(cutPointsf),
        "lengthOut": lengthSlicesf,
    }
