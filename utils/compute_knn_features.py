# Function to calculate the k-nearest neighbors for all cells in an organoid.


import ast
import numpy as np
import sklearn.neighbors


def compute_knn_features(
    data,
    k=5,
    position_columns=["x", "y", "z"],
    get_phenotype_score=True,
    label_column="label",
    distance_column="mean_knn_distance",
    neighbors_column=None,
    add_cell_id=True,
):
    data = data.copy()

    try:
        k = int(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be a positive integer.") from exc
    if k < 1:
        raise ValueError("k must be a positive integer.")

    if neighbors_column is None:
        neighbors_column = f"{k}_knn_neighbors"

    n_cells = len(data)

    # Prefer explicit labels (segmentation IDs) if available; otherwise use row index labels.
    if label_column in data.columns:
        label_values = data[label_column].to_numpy()
    else:
        label_values = data.index.to_numpy()

    if n_cells == 0:
        data[distance_column] = []
        data[neighbors_column] = []
        if add_cell_id:
            data["cell_id"] = []
        if get_phenotype_score:
            data["phenotype_similarity_score"] = []
        return data

    if n_cells <= 1:
        data[distance_column] = np.nan
        data[neighbors_column] = [[] for _ in range(n_cells)]
        if add_cell_id:
            data["cell_id"] = range(len(data))
        if get_phenotype_score:
            data["phenotype_similarity_score"] = np.nan
        return data

    # Get an array of the coordinates of every cell
    coordinates = data[position_columns].to_numpy()

    # Fit the k-nearest neighbors model and compute the k-nearest neighbors for each cell
    effective_k = min(k, n_cells - 1)
    neighbours = sklearn.neighbors.NearestNeighbors(
        n_neighbors=effective_k + 1, algorithm="kd_tree"
    ).fit(coordinates)
    distances, indices = neighbours.kneighbors(coordinates)
    knn_distances = distances[:, 1:]  # Exclude the first column (distance to itself)
    knn_indices = indices[:, 1:]  # Exclude the first column (index of itself)
    knn_neighbor_labels = [label_values[row_indices].tolist() for row_indices in knn_indices]

    # Calculate the mean distance of every cell to its 5 nearest neighbours, which can measure density
    mean_knn_distance = [distance.mean() for distance in knn_distances]

    # Add the mean distance and the indices of the k-nearest neighbors to the dataframe
    data[distance_column] = mean_knn_distance
    data[neighbors_column] = knn_neighbor_labels

    # This cell id is the same as the label-1, this is what you can compare the knn_neighbours column with
    if add_cell_id:
        data["cell_id"] = range(len(data))

    if not get_phenotype_score:
        return data

    if "phenotype" not in data.columns:
        data["phenotype_similarity_score"] = np.nan
        return data

    if label_column in data.columns:
        phenotype_lookup = data.drop_duplicates(subset=[label_column]).set_index(
            label_column
        )["phenotype"]
        current_keys = data[label_column].to_numpy()
    else:
        phenotype_lookup = data["phenotype"]
        current_keys = data.index.to_numpy()

    # Here we calculate a phenotype similarity score
    # This means that each cell gets a score of 0 to 5 based on how many of its nearest neigbours have the same phenotype
    # A score of 5 means that all 5 nearest neighbours have the same phenotype,
    phenotype_scores = []
    for current_key, nn_labels_row in zip(current_keys, data[neighbors_column]):
        score = calculate_phenotype_similarity_score(
            current_key, nn_labels_row, phenotype_lookup
        )
        phenotype_scores.append(score)

    # Add score to dataframe
    data["phenotype_similarity_score"] = phenotype_scores

    return data


def calculate_phenotype_similarity_score(current_key, nn_labels_row, phenotype_lookup):
    if nn_labels_row is None or len(nn_labels_row) == 0:
        return np.nan

    # Get the phenotype of the current cell
    current_phenotype = phenotype_lookup.get(current_key)
    if current_phenotype is None:
        return np.nan

    # Get phenotypes of the nearest neighbors
    neighbor_phenotypes = phenotype_lookup.reindex(nn_labels_row)

    # Count how many neighbors have the same phenotype
    same_phenotype_count = (neighbor_phenotypes == current_phenotype).sum()

    return same_phenotype_count


def _normalize_neighbor_labels(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return []
    return []


def compute_phenotype_similarity_from_neighbors(
    data,
    neighbors_column,
    phenotype_column="phenotype",
    label_column="label",
    output_column="phenotype_similarity_score",
):
    data = data.copy()

    if len(data) == 0:
        data[output_column] = []
        return data

    if neighbors_column not in data.columns:
        raise ValueError(
            f"Missing neighbors column '{neighbors_column}'. Please calculate KNN first."
        )
    if phenotype_column not in data.columns:
        raise ValueError(
            f"Missing phenotype column '{phenotype_column}'. Please calculate phenotypes first."
        )

    if label_column in data.columns:
        phenotype_lookup = data.drop_duplicates(subset=[label_column]).set_index(
            label_column
        )[phenotype_column]
        current_keys = data[label_column].to_numpy()
    else:
        phenotype_lookup = data[phenotype_column]
        current_keys = data.index.to_numpy()

    normalized_neighbors = data[neighbors_column].apply(_normalize_neighbor_labels)

    scores = []
    for current_key, neighbor_labels in zip(current_keys, normalized_neighbors):
        score = calculate_phenotype_similarity_score(
            current_key, neighbor_labels, phenotype_lookup
        )
        scores.append(score)

    data[output_column] = scores
    return data
