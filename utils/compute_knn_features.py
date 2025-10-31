# Function to calculate the k-nearest neighbors for all cells in an organoid.


import sklearn.neighbors


def compute_knn_features(
    data, k=5, position_columns=["x", "y", "z"], get_phenotype_score=True
):
    # Get an array of the coordinates of every cell
    coordinates = data[position_columns].to_numpy()

    # Fit the k-nearest neighbors model and compute the k-nearest neighbors for each cell
    neighbours = sklearn.neighbors.NearestNeighbors(
        n_neighbors=k + 1, algorithm="kd_tree"
    ).fit(coordinates)
    distances, indices = neighbours.kneighbors(coordinates)
    knn_distances = distances[:, 1:]  # Exclude the first column (distance to itself)
    knn_indices = indices[:, 1:]  # Exclude the first column (index of itself)

    # Calculate the mean distance of every cell to its 5 nearest neighbours, which can measure density
    mean_knn_distance = [distance.mean() for distance in knn_distances]

    # Add the mean distance and the indices of the k-nearest neighbors to the dataframe
    data["mean_knn_distance"] = mean_knn_distance
    data[f"{k}_knn_neighbors"] = list(knn_indices)

    # This cell id is the same as the label-1, this is what you can compare the knn_neighbours column with
    data["cell_id"] = range(len(data))

    if not get_phenotype_score:
        return data

    # Here we calculate a phenotype similarity score
    # This means that each cell gets a score of 0 to 5 based on how many of its nearest neigbours have the same phenotype
    # A score of 5 means that all 5 nearest neighbours have the same phenotype,
    phenotype_scores = []
    for i, nn_indices_row in enumerate(data[f"{k}_knn_neighbors"]):
        score = calculate_phenotype_similarity_score(
            i, nn_indices_row, data["phenotype"]
        )
        phenotype_scores.append(score)

    # Add score to dataframe
    data["phenotype_similarity_score"] = phenotype_scores

    return data


def calculate_phenotype_similarity_score(row_idx, nn_indices_row, phenotype_series):
    # Get the phenotype of the current cell
    current_phenotype = phenotype_series.iloc[row_idx]

    # Get phenotypes of the nearest neighbors
    neighbor_phenotypes = phenotype_series.iloc[nn_indices_row]

    # Count how many neighbors have the same phenotype
    same_phenotype_count = (neighbor_phenotypes == current_phenotype).sum()

    return same_phenotype_count
