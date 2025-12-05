def ellipses_connector(data2d, slice_idx, temp_elongated, obj_idx, min_z, k=4):
    import numpy as np
    from OSCAR_stitching.fit_3D_line import fit_3D_line
    from OSCAR_stitching.overlapped_ellipses import overlapped_ellipses
    from OSCAR_stitching.sort_points_ascending import sort_points_ascending

    next_step = 0
    n_points = int(round(min_z))
    next_ellipse_idx = None

    # If the current cell is already longer than min_z, we will still only use min_z amount of slices
    # to predict the next positions
    if len(temp_elongated) >= n_points:
        xS = temp_elongated.iloc[:n_points]["x"].values
        yS = temp_elongated.iloc[:n_points]["y"].values
        zS = temp_elongated.iloc[:n_points]["z"].values
        angref = temp_elongated.iloc[:n_points]["orientation"].median()
        aref = temp_elongated.iloc[:n_points]["majorAxisLength"].median()
        bref = temp_elongated.iloc[:n_points]["minorAxisLength"].median()
    else:
        # Use all available slices if we don't have enough yet
        xS = temp_elongated["x"].values
        yS = temp_elongated["y"].values
        zS = temp_elongated["z"].values
        angref = temp_elongated["orientation"].median()
        aref = temp_elongated["majorAxisLength"].median()
        bref = temp_elongated["minorAxisLength"].median()

    # Create design matrix for regression
    z_test = np.column_stack([np.ones(len(xS)), np.arange(1, len(xS) + 1)])

    # Check if regression is possible, this will fail if we have to little slices yet
    if np.linalg.det(z_test.T @ z_test) != 0:
        # Fit 3D line and predict next position
        line = fit_3D_line(x=xS, y=yS, z=zS)  # test
        predicted_x = line["xref"]
        predicted_y = line["yref"]
    else:
        # Fallback: use median position
        predicted_x = np.median(xS)
        predicted_y = np.median(yS)

    # Extract properties of the last ellipse in the elongated object
    a_current = temp_elongated.iloc[-1]["majorAxisLength"] / 2
    b_current = temp_elongated.iloc[-1]["minorAxisLength"] / 2
    angle_current = temp_elongated.iloc[-1]["orientation"]  # Flip sign
    x_current = temp_elongated.iloc[-1]["x"]
    y_current = temp_elongated.iloc[-1]["y"]

    # Find nearest ellipses in the next slice
    next_slice_x = data2d[slice_idx + 1]["x"].values
    next_slice_y = data2d[slice_idx + 1]["y"].values

    sorted_indices, sorted_distances, distances = sort_points_ascending(
        x_current, y_current, next_slice_x, next_slice_y
    )

    # Now we will use a KNN approach to find the best matching ellipses,
    overlapping = []
    if len(sorted_indices) < k:  # If not enough neigbours in next slice, reduce k
        k = len(sorted_indices)
    for i in range(k):
        # Extract properties of candidate ellipse in next slice
        candidate_idx = sorted_indices[i]
        a_candidate = data2d[slice_idx + 1].iloc[candidate_idx]["majorAxisLength"] / 2
        b_candidate = data2d[slice_idx + 1].iloc[candidate_idx]["minorAxisLength"] / 2
        angle_candidate = data2d[slice_idx + 1].iloc[candidate_idx]["orientation"]
        x_candidate = data2d[slice_idx + 1].iloc[candidate_idx]["x"]
        y_candidate = data2d[slice_idx + 1].iloc[candidate_idx]["y"]

        # Check overlap between current ellipse and candidate
        overlap_info = overlapped_ellipses(
            a_current,
            b_current,
            x_current,
            y_current,
            angle_current,
            a_candidate,
            b_candidate,
            x_candidate,
            y_candidate,
            angle_candidate,
        )

        overlap_score = overlap_info["AonB"] + overlap_info["BonA"]

        # If overlap exists, add to selection
        if overlap_score > 0:
            overlapping.append(candidate_idx)
        else:
            # No overlap → stop checking further ellipses
            break

    if len(overlapping) > 0:
        sorted_indexes, sorted_distances_knn, distances_knn = sort_points_ascending(
            predicted_x,
            predicted_y,
            data2d[slice_idx + 1].iloc[overlapping]["x"].values,
            data2d[slice_idx + 1].iloc[overlapping]["y"].values,
        )
        next_step = 1
        next_ellipse_idx = overlapping[sorted_indexes[0]]

    # Return the connection information
    return {
        "next_ellipse_idx": next_ellipse_idx,  # None or int
        "continue_elongation": next_step,  # 0 or 1
        "predicted_x": predicted_x,
        "predicted_y": predicted_y,
    }
