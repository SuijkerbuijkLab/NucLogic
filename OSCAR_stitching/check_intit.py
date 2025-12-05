def check_intit(data2d, slice_idx, obj_idx):
    import numpy as np
    from OSCAR_stitching.sort_points_ascending import sort_points_ascending
    from OSCAR_stitching.sort_points_ascending_map_to_z import (
        sort_points_ascending_map_to_z,
    )

    mean_diameter = 0
    disappear_init = np.array([[0, 0, 0]])

    x = data2d[slice_idx].iloc[obj_idx]["x"]  # x position of mask
    y = data2d[slice_idx].iloc[obj_idx]["y"]  # y postion of mask

    if slice_idx != 0:  # if the data2d is not the first slice
        all_x_previous_slice_and_this_slice = np.concatenate(
            [data2d[slice_idx]["x"].values, data2d[slice_idx - 1]["x"].values]
        ).astype(
            np.float64
        )  # Convert to float!
        all_y_previous_slice_and_this_slice = np.concatenate(
            [data2d[slice_idx]["y"].values, data2d[slice_idx - 1]["y"].values]
        ).astype(
            np.float64
        )  # Convert to float!
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
        # Filter valid indices safely
        # Filter valid indices safely
        valid_mask = []
        for k in range(len(indexes_sorted)):
            z_idx = z_sorted[k]
            row_idx = indices_in_each_z[k]

            # Check if z_idx is a valid index in data2d (which is a list)
            if (
                0 <= z_idx < len(data2d)  # Check if z_idx is valid list index
                and row_idx < len(data2d[z_idx])
                and data2d[z_idx].iloc[row_idx]["assignmentTag"] == 1
                and sorted_distances[k] != 0
            ):
                valid_mask.append(k)

        # Filter indexes_sorted and sorted_distances using valid_mask
        indexes_sorted = [indexes_sorted[k] for k in valid_mask]
        sorted_distances_filtered = [sorted_distances[k] for k in valid_mask]
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
