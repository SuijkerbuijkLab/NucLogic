def sort_points_ascending_map_to_z(x_mask, y_mask, all_x, all_y, all_z):
    import numpy as np

    # Ensure inputs are proper numpy types
    x_mask = np.float64(x_mask)  # Convert to scalar numpy float
    y_mask = np.float64(y_mask)  # Convert to scalar numpy float
    all_x = np.asarray(all_x, dtype=np.float64)
    all_y = np.asarray(all_y, dtype=np.float64)
    all_z = np.asarray(all_z, dtype=np.int32)

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
