def sort_points_ascending(xc, yc, all_x, all_y):
    import numpy as np

    # Calculate Euclidean distance from each point to reference
    distances = np.sqrt((all_x - xc) ** 2 + (all_y - yc) ** 2)

    # Sort distances and get sorted indices
    indexes_sorted = np.argsort(distances)
    distances_sorted = distances[indexes_sorted]

    return indexes_sorted, distances_sorted, distances
