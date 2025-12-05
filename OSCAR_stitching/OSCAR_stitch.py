import tifffile
import numpy as np
import pandas as pd
from dataclasses import dataclass

import os


def data_2D_chunking(image):
    from skimage.measure import regionprops
    import pandas as pd

    data2d = []
    empty_slice_counter = 0

    for z in range(image.shape[0]):
        props = regionprops(image[z])

        if len(props) == 0:
            # Empty slice - add empty DataFrame with correct columns
            empty_df = pd.DataFrame(
                columns=[
                    "label",
                    "x",
                    "y",
                    "z",
                    "area",
                    "orientation",
                    "assignmentTag",
                    "majorAxisLength",
                    "minorAxisLength",
                    "indexinslice",
                    "z_index_without_empty",
                ]
            )
            data2d.append(empty_df)
            empty_slice_counter += 1
            continue

        slice_data = []
        for idx, prop in enumerate(props):
            slice_data.append(
                {
                    "label": prop.label,
                    "x": int(prop.centroid[1]),
                    "y": int(prop.centroid[0]),
                    "z": z,  # This is indexInZ in Julia
                    "area": prop.area,
                    "orientation": prop.orientation,
                    "assignmentTag": 0,
                    "majorAxisLength": prop.major_axis_length,
                    "minorAxisLength": prop.minor_axis_length,
                    "indexinslice": idx,  # This is indexData in Julia
                    "z_index_without_empty": z - empty_slice_counter,
                }
            )
        data2d.append(pd.DataFrame(slice_data))

    return data2d


def sort_points_ascending_map_to_z(x_mask, y_mask, all_x, all_y, all_z):
    # Calculate Euclidean distance from each point to reference
    distances = np.sqrt((all_x - x_mask) ** 2 + (all_y - y_mask) ** 2)

    # Find where Z changes (boundary between current and previous slice)
    # This creates indices that track position within each Z-slice
    first_z_change = np.where(all_z != all_z[0])[0]

    if len(first_z_change) > 0:
        first_idx = first_z_change[0]  # Index where Z changes

        # Create indices for each slice:
        # First slice: [0, 1, 2, ..., first_idx-1]
        # Second slice: [0, 1, 2, ..., remaining_length-1]
        indices_in_each_z = np.concatenate(
            [
                np.arange(first_idx),  # Indices for first Z-slice
                np.arange(len(all_z) - first_idx),  # Indices for second Z-slice
            ]
        )
    else:
        # All points are in the same Z-slice
        indices_in_each_z = np.arange(len(all_z))

    # Sort points by distance (ascending order)
    indexes_sorted = np.argsort(distances)

    # Apply sorting to other arrays to maintain correspondence
    indices_in_each_z = indices_in_each_z[indexes_sorted]
    sorted_distances = distances[indexes_sorted]  # Sorted distances
    z_sorted = all_z[indexes_sorted].astype(int)  # Z-slices after sorting

    return indexes_sorted, sorted_distances, distances, z_sorted, indices_in_each_z


def sort_points_ascending(xc, yc, all_x, all_y):
    # Calculate Euclidean distance from each point to reference
    distances = np.sqrt((all_x - xc) ** 2 + (all_y - yc) ** 2)

    # Sort distances and get sorted indices
    indexes_sorted = np.argsort(distances)
    distances_sorted = distances[indexes_sorted]

    return indexes_sorted, distances_sorted, distances


def checkIntit(data2d, slice_idx, obj_idx):
    mean_diameter = 0
    disappear_init = np.array([[0, 0, 0]])

    x = data2d[slice_idx].iloc[obj_idx]["x"]  # x position of mask
    y = data2d[slice_idx].iloc[obj_idx]["y"]  # y postion of mask

    if slice_idx != 0:  # if the data2d is not the first slice
        all_x_previous_slice_and_this_slice = np.concatenate(
            [data2d[slice_idx]["x"].values, data2d[slice_idx - 1]["x"].values]
        )
        all_y_previous_slice_and_this_slice = np.concatenate(
            [data2d[slice_idx]["y"].values, data2d[slice_idx - 1]["y"].values]
        )
        all_z_previous_slice_and_this_slice = np.concatenate(
            [
                data2d[slice_idx]["z_index_without_empty"].values,
                data2d[slice_idx - 1]["z_index_without_empty"].values,
            ]
        )

        # Calculate the distances from this mask to all masks in the current and previous slice
        indexes_sorted, sorted_distances, distances, z_sorted, indices_in_each_z = (
            sort_points_ascending_map_to_z(
                x,
                y,
                all_x_previous_slice_and_this_slice,
                all_y_previous_slice_and_this_slice,
                all_z_previous_slice_and_this_slice,
            )
        )
        # We only want to consider masks that are assigned (assignmentTag == 1) and non-zero distance (ie. not itself)
        indexes_sorted = [
            indexes_sorted[k]
            for k in range(len(indexes_sorted))
            if data2d[z_sorted[k]].iloc[indices_in_each_z[k]]["assignmentTag"] == 1
            and sorted_distances[k] != 0
        ]
        distances_filtered = distances[indexes_sorted]

        if len(distances_filtered) > 0 and np.any(distances_filtered < mean_diameter):
            # Mark current object as merged
            data2d[slice_idx].at[obj_idx, "assignmentTag"] = 1

            # Record merger coordinates
            disappear_init = np.array(
                [
                    [
                        data2d[slice_idx].iloc[obj_idx]["x"],
                        data2d[slice_idx].iloc[obj_idx]["y"],
                        data2d[slice_idx].iloc[obj_idx]["z"],
                    ]
                ]
            )

    else:  # First slice (slice_idx == 0)
        # Only check within current slice
        all_x_this_slice = data2d[slice_idx]["x"].values
        all_y_this_slice = data2d[slice_idx]["y"].values

        # Sort by distance to current object
        indexes_sorted, distances_sorted, distances = sort_points_ascending(
            x, y, all_x_this_slice, all_y_this_slice
        )

        # Filter: only keep already-assigned objects (tag == 1) and exclude self (distance != 0)
        valid_indices = [
            k
            for k in range(len(indexes_sorted))
            if (
                data2d[slice_idx].iloc[indexes_sorted[k]]["assignmentTag"] == 1
                and distances_sorted[k] != 0
            )
        ]

        if len(valid_indices) > 0:
            indexes_sorted = indexes_sorted[valid_indices]
            distances_filtered = distances[indexes_sorted]

            # Check if any nearby assigned objects exist
            if len(distances_filtered) > 0 and np.any(
                distances_filtered < mean_diameter
            ):
                # Increment tag (allows tracking multiple mergers)
                data2d[slice_idx].at[obj_idx, "assignmentTag"] += 1

                # Record merger coordinates
                disappear_init = np.array(
                    [
                        [
                            data2d[slice_idx].iloc[obj_idx]["x"],
                            data2d[slice_idx].iloc[obj_idx]["y"],
                            data2d[slice_idx].iloc[obj_idx]["z"],
                        ]
                    ]
                )

    return disappear_init


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


def fit_3d_line(x, y, z=None, points_to_fit=None):
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


def ellipses_connector(data2d, slice_idx, temp_elongated, obj_idx, min_z, k=4):
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
        line = fit_3d_line(x=xS, y=yS, z=zS)
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
        line = fit_3d_line(x=x_subset, y=y_subset, z=z_subset)

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


def get_covariance(var1, var2):
    """
    Calculates the Pearson correlation coefficient between two vectors.

    Args:
        var1: First vector of data
        var2: Second vector of data

    Returns:
        Pearson correlation coefficient
    """
    # Convert to numpy arrays if needed
    var1 = np.array(var1)
    var2 = np.array(var2)

    # Check if standard deviations are zero
    std1 = np.std(var1, ddof=1)  # Use ddof=1 for sample std
    std2 = np.std(var2, ddof=1)

    if std1 * std2 == 0:
        pearson_coefficient = 1.0
    else:
        # Calculate covariance and normalize by standard deviations
        covariance = np.cov(var1, var2, ddof=1)[0, 1]  # ddof=1 for sample covariance
        pearson_coefficient = covariance / (std1 * std2)

    return pearson_coefficient


def pre_obj_elongation(slice_idx, obj_idx, data2d, max_z, min_z, med_z):
    import pandas as pd

    # Create deep copy to avoid modifying original data2d
    data2d = [df.copy() for df in data2d]

    slice_data = data2d[slice_idx]

    volumes = []
    pearson_ind = 0
    volume_elipse = (
        slice_data.iloc[obj_idx]["majorAxisLength"]
        * slice_data.iloc[obj_idx]["minorAxisLength"]
        * np.pi
    )
    temp_elongated = slice_data.iloc[
        [obj_idx]
    ].copy()  # starts out as just the elipse in 1 slice

    # Store the starting slice index
    starting_slice = slice_idx
    current_slice = slice_idx  # Track current position

    # Loop until max_z slices or end of data2d
    # FIX: Check against starting_slice + max_z and len(data2d) - 1
    while (
        current_slice < starting_slice + max_z - 1 and current_slice < len(data2d) - 1
    ):
        ellipse_info = ellipses_connector(
            data2d, current_slice, temp_elongated, obj_idx, min_z
        )
        continue_elongation = ellipse_info["continue_elongation"]
        next_ellipse_idx = 0  # Initialize

        if continue_elongation == 1:
            # Connection found - extend the elongated object
            next_ellipse_idx = ellipse_info["next_ellipse_idx"]

            # Update for next iteration
            obj_idx = next_ellipse_idx  # Update obj_idx for next connector call
            current_slice += 1  # Move to next slice

            # Append connected ellipse to elongated object
            new_row = data2d[current_slice].iloc[[obj_idx]].copy()
            temp_elongated = pd.concat([temp_elongated, new_row], ignore_index=True)

            # Calculate cumulative volume (sum of all ellipse areas)
            cumulative_volume = (
                temp_elongated["majorAxisLength"]
                * temp_elongated["minorAxisLength"]
                * np.pi
            ).sum()
            volumes.append(cumulative_volume)
        else:
            # No connection found - stop elongation
            break

    info = info_preobj_dist_z(temp_elongated, index1=0, index2=int(round(med_z)))
    distances_from_line = info["dist"]
    angles_between_segments = info["ang"]
    pearson = info["pearson"]

    # Return everything
    return {
        "temp_elongated": temp_elongated,
        "volumes": volumes,
        "distances_from_line": distances_from_line,
        "angles_between_segments": angles_between_segments,
        "pearson": pearson,
    }


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


@dataclass
class Objects3D:
    """
    Container for validated 3D objects and metadata.

    Attributes:
        preok: List of DataFrames, each representing a validated 3D object
        infoDF: DataFrame with summary statistics (original length, cut point, coherence)
        inis: Total number of initiator objects processed
        noInits: Array of merged object coordinates
    """

    preok: list  # List of pd.DataFrame
    infoDF: pd.DataFrame
    inis: int
    noInits: np.ndarray


def object_splitter_3D(image, min_z=4, med_z=6, max_z=9, nuclei_channel=0):
    from tqdm import tqdm

    data2d = data_2D_chunking(image)

    # Storage containers for final 3D objects and their metrics
    validated_objects_3d = []  # List of DataFrames, each representing a 3D object
    cut_points_f = []  # List of integers: where each object was truncated
    coherence = []  # List of floats: Pearson correlation for each object
    object_lengths = []  # List of integers: number of slices per object

    # Track which objects have been assigned
    # assigned = [np.zeros(len(props), dtype=bool) for props in data2d.groupby("z")]

    # Global counters
    border_effect_low = 1
    initiators = 0
    merged_objects = np.zeros((1, 3))  # Objects that merged/disappeared
    counter_cin = 0
    count_pearson = 0
    counter_truncations = 0

    print("Starting 3D object stitching...")
    # Main loop to process each slice
    for slice_idx in tqdm(range(len(data2d) - border_effect_low)):
        current_slice = data2d[slice_idx]  # Get DataFrame for this slice
        num_objects = len(current_slice)  # Number of rows in this DataFrame

        for obj_idx in range(num_objects):  # Loop through each object in the slice
            mask = current_slice.iloc[obj_idx]
            # This function checks if the object merges or disappears in the slice below
            disappear_init = checkIntit(data2d, slice_idx, obj_idx)
            # If merger detected, record it
            if np.any(disappear_init != 0):
                merged_objects = np.vstack([merged_objects, disappear_init])

            # Go over any object that is not stitched yet
            if current_slice.iloc[obj_idx]["assignmentTag"] == 0:
                initiators += 1

                # Perform 3D elongation
                pre_obj = pre_obj_elongation(
                    slice_idx, obj_idx, data2d, max_z, min_z, med_z
                )

                temp_elongated = pre_obj["temp_elongated"]  # Elongated object data2d
                pre_vol = pre_obj["volumes"]  # Volume progression
                dist_v = pre_obj["distances_from_line"]  # Distances from fitted line
                ang_v = pre_obj["angles_between_segments"]  # Angles between segments
                pearson = pre_obj["pearson"]  # Coherence score

                # Initialize truncation point (full length by default)
                cut_point = len(temp_elongated)
                factor = pearson

                # Check if the elongated object meets minimum length
                if len(temp_elongated) >= min_z:
                    # If longer than median length, evaluate for possible truncation
                    if len(temp_elongated) > med_z:
                        # Detect potential merge points
                        terminator_info = terminator_returns(
                            dist_v[1:],  # Skip first distance
                            ang_v,
                            int(round(med_z)),
                        )

                        cin = terminator_info["numbIn"]
                        cout = terminator_info["numbOut"]

                        # Normalize by segment length
                        cin = cin / med_z
                        cout = cout / (len(temp_elongated) - med_z + 1)

                        cut_positions = terminator_info["cutOut"]
                        outlier_lengths = terminator_info["lengthOut"]

                        # Check if later part is more erratic than beginning
                        if cin < cout and cout > 0:
                            counter_cin += 1

                            # Get first outlier position
                            n_point_h = int(cut_positions[0] + 1)

                            # Calculate Pearson BEFORE cut point
                            if n_point_h - 2 < 0:
                                pearson_in = 0
                            else:
                                pearson_in = get_covariance(
                                    dist_v[1 : n_point_h - 1],
                                    ang_v[0 : n_point_h - 2],
                                )

                            # Calculate Pearson AFTER cut point
                            if n_point_h - 1 < 0:
                                pearson_out = 0
                            else:
                                pearson_out = get_covariance(
                                    dist_v[n_point_h - 1 :],
                                    ang_v[n_point_h - 2 :],
                                )

                            # Truncate if trajectory before cut is better
                            if pearson_in > pearson:
                                count_pearson += 1
                                cut_point = int(n_point_h)
                                counter_truncations += 1

                    # ===== STORE OBJECT METADATA =====
                    cut_points_f.append(cut_point)
                    object_lengths.append(len(temp_elongated))
                    coherence.append(pearson)

                    # ===== MARK ELLIPSES AS ASSIGNED =====
                    for zz in range(cut_point):
                        # Use the actual column names from temp_elongated
                        slice_z = int(temp_elongated.iloc[zz]["z"])  # This is indexInZ
                        data_idx = int(
                            temp_elongated.iloc[zz]["indexinslice"]
                        )  # This is indexData

                        # Increment assignment tag in original data
                        data2d[slice_z].at[data_idx, "assignmentTag"] += 1

                        # Update tag in elongated object
                        temp_elongated.at[zz, "assignmentTag"] = data2d[slice_z].iloc[
                            data_idx
                        ]["assignmentTag"]

                    # ===== STORE FINAL OBJECT (truncated if necessary) =====
                    final_object = temp_elongated.iloc[:cut_point].copy()
                    validated_objects_3d.append(final_object)

    # Create summary DataFrame
    summary_df = pd.DataFrame(
        {
            "original_length": object_lengths,
            "cut_point": cut_points_f,
            "pearson": coherence,
        }
    )

    print(f"\n=== Summary ===")
    print(f"Initiators found: {initiators}")
    print(f"Valid 3D objects: {len(validated_objects_3d)}")
    print(f"Objects truncated: {counter_truncations}")
    print(f"Cin/Cout triggers: {counter_cin}")
    print(f"Pearson-based truncations: {count_pearson}")

    # ===== RETURN OBJECTS3D DATACLASS =====
    return (
        Objects3D(
            preok=validated_objects_3d,
            infoDF=summary_df,
            inis=initiators,
            noInits=merged_objects,
        ),
        counter_cin,
        count_pearson,
    )


# ...existing code...


def draw_3d_ellipses_from_summary(summary_df, dims, label_ellipses=True):
    """
    Draws 3D ellipses as 2D cross-sections using information from a summary DataFrame.
    """
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


def create_summary_from_objects(objects_3d):
    """
    Creates a summary DataFrame with geometric properties for each 3D object.

    Args:
        objects_3d: Objects3D dataclass containing validated 3D objects

    Returns:
        DataFrame with columns needed for visualization:
        - Xcenter, Ycenter, Zcenter: Center coordinates
        - A, B: Major and minor axis lengths
        - Vz: Object length in Z
        - AngleXY: Orientation angle in XY plane
        - Vx, Vy: Direction vector components
        - C: Magnitude of direction vector
        - Label: Object ID
    """
    summary_data = []

    for obj_id, obj_df in enumerate(objects_3d.preok, start=1):
        if len(obj_df) == 0:
            continue

        # Calculate center coordinates (mean position)
        x_center = obj_df["x"].mean()
        y_center = obj_df["y"].mean()
        z_center = obj_df["z"].mean()

        # Calculate mean axis lengths
        a = obj_df["majorAxisLength"].mean()
        b = obj_df["minorAxisLength"].mean()

        # Object length in Z
        v_z = len(obj_df)

        # Calculate mean orientation angle (convert to degrees)
        angle_xy = np.degrees(obj_df["orientation"].mean())

        # Calculate direction vector
        if len(obj_df) > 1:
            # Fit 3D line to get direction
            line = fit_3d_line(
                x=obj_df["x"].values, y=obj_df["y"].values, z=obj_df["z"].values
            )
            vx = line["vx"]
            vy = line["vy"]
            v_module = line["vmodule"]
        else:
            # Single slice object
            vx = 0.0
            vy = 0.0
            v_module = 1.0

        summary_data.append(
            {
                "Label": obj_id,
                "Xcenter": x_center,
                "Ycenter": y_center,
                "Zcenter": z_center,
                "A": a,
                "B": b,
                "Vz": v_z,
                "AngleXY": angle_xy,
                "Vx": vx,
                "Vy": vy,
                "C": v_module,
            }
        )

    return pd.DataFrame(summary_data)


# Update the test code at the bottom:
# Test code
image = tifffile.imread(
    r"C:\Users\6331823\Local SSD\Data_Elise\EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24\segmented\Frame-10_masks.tif"
)

print(f"Input image shape: {image.shape}")

object3D, _, _ = object_splitter_3D(image, min_z=4, med_z=6, max_z=9)

# Create summary DataFrame from the validated objects
summary_df = create_summary_from_objects(object3D)
print(f"\nSummary DataFrame:\n{summary_df.head()}")

# Draw the 3D ellipses
stitched = draw_3d_ellipses_from_summary(
    summary_df=summary_df,  # Use the new summary
    dims=image.shape,
    label_ellipses=True,
)

import napari

print(f"Stitched image shape: {stitched.shape}")
print(f"Number of stitched objects: {len(np.unique(stitched)) - 1}")

tifffile.imwrite(
    r"C:\Users\6331823\Local SSD\Data_Elise\EM_Exp007_Imaging_20x_2025-11-21_Maria_Elise_14.47.50_CZC34578YN_F24\\Frame-10_stitched_OSCAR.tif",
    stitched.astype(np.uint16),
)

viewer = napari.Viewer()
viewer.add_labels(stitched, name="stitched", scale=(2.5, 0.64, 0.64))
viewer.add_labels(image, name="original", scale=(2.5, 0.64, 0.64))
napari.run()
