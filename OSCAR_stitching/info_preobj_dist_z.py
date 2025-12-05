def info_preobj_dist_z(temp_elongated, index1=0, index2=None):
    """
    Calculates trajectory quality metrics for an elongated object.

    Args:
        temp_elongated: DataFrame containing elongated object data
        index1: Starting index for the data subset (default: 0)
        index2: Ending index for the data subset (default: None = all rows)

    Returns:
        Dictionary containing:
        - 'dist': Vector of Euclidean distances from each centroid to the fitted 3D line
        - 'ang': Vector of angle cosines between consecutive trajectory segments
        - 'pearson': Pearson correlation coefficient between distances and angles
    """
    import numpy as np
    from OSCAR_stitching.fit_3D_line import fit_3D_line
    from OSCAR_stitching.sort_points_ascending import sort_points_ascending
    from OSCAR_stitching.get_covariance import get_covariance

    # Set default index2 to end of DataFrame
    if index2 is None:
        index2 = len(temp_elongated)

    # Extract subset of data for line fitting
    if index1 < 0 or index2 > len(temp_elongated):
        # Use all data if indices are out of bounds
        x_subset = temp_elongated["x"].values
        y_subset = temp_elongated["y"].values
        z_subset = temp_elongated["z"].values
        n_points = len(x_subset)
    else:
        # Use specified subset
        x_subset = temp_elongated.iloc[index1:index2]["x"].values
        y_subset = temp_elongated.iloc[index1:index2]["y"].values
        z_subset = temp_elongated.iloc[index1:index2]["z"].values
        n_points = len(x_subset)

    # Check if line fitting is possible
    z_test = np.column_stack(
        [np.ones(len(x_subset)), np.arange(1.0, len(x_subset) + 1.0)]
    )

    if np.linalg.det(z_test.T @ z_test) != 0:
        # Fit 3D line to the subset
        line = fit_3D_line(x=x_subset, y=y_subset, z=z_subset)

        cx = line["xcenter"]
        cy = line["ycenter"]

        # Get regression coefficients
        beta_x = line["betaX"]
        beta_y = line["betaY"]

        # Extract slopes (direction of the line)
        slope_x = beta_x[1]
        slope_y = beta_y[1]

        # Predict X and Y for ALL points in temp_elongated (not just subset)
        z_matrix_all = np.column_stack(
            [np.ones(len(temp_elongated)), temp_elongated["z"].values]
        )

        x_predicted = z_matrix_all @ beta_x
        y_predicted = z_matrix_all @ beta_y
    else:
        # Fallback: use median position
        cx = np.median(x_subset)
        cy = np.median(y_subset)
        slope_x = 0
        slope_y = 0
        # No predictions possible
        x_predicted = None
        y_predicted = None

    # Initialize output vectors
    dist_v = []
    ang_v = []
    pearson_correlation = 0

    if len(temp_elongated) > 1:
        # ===== Calculate distances from fitted line =====
        try:
            if x_predicted is not None and y_predicted is not None:
                # Euclidean distance from each centroid to the fitted line
                dist_v = np.sqrt(
                    (x_predicted - temp_elongated["x"].values) ** 2
                    + (y_predicted - temp_elongated["y"].values) ** 2
                )
            else:
                raise ValueError("No predicted line")
        except:
            print("No predicted line - using distances from median center")
            # Fallback: distances from median center
            sorted_indices, sorted_distances, dist_v = sort_points_ascending(
                cx, cy, temp_elongated["x"].values, temp_elongated["y"].values
            )

        # ===== Calculate angles between consecutive segments =====
        # Reference vector: direction of the fitted line in 3D
        z_span = temp_elongated.iloc[-1]["z"] - temp_elongated.iloc[0]["z"]
        ref_vector = np.array([slope_x, slope_y, z_span])

        for i in range(len(temp_elongated) - 1):
            # Vector from current point to next point
            dx = temp_elongated.iloc[i + 1]["x"] - temp_elongated.iloc[i]["x"]
            dy = temp_elongated.iloc[i + 1]["y"] - temp_elongated.iloc[i]["y"]
            test_vector = np.array([dx, dy, 1])

            # Cosine of angle between vectors
            dot_product = np.dot(test_vector, ref_vector)
            magnitude_product = np.linalg.norm(test_vector) * np.linalg.norm(ref_vector)

            if magnitude_product != 0:
                angle_cosine = dot_product / magnitude_product
            else:
                angle_cosine = 0

            ang_v.append(angle_cosine)

        # ===== Calculate Pearson correlation =====
        if len(dist_v) > 1:
            # Use distances starting from index 1 (skip first point)
            # and all angles (which has one less element than distances)
            dist_subset = dist_v[1:]  # Skip first distance
            ang_subset = ang_v  # All angles

            if len(dist_subset) == len(ang_subset) and len(dist_subset) > 1:
                pearson_correlation = get_covariance(dist_subset, ang_subset)
            else:
                pearson_correlation = 0
        else:
            pearson_correlation = 0

    return {
        "dist": np.array(dist_v) if isinstance(dist_v, list) else dist_v,
        "ang": np.array(ang_v),
        "pearson": pearson_correlation,
    }
