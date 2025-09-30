# Function to calculate the k-nearest neighbors for all cells in an organoid.


import numpy as np
import sklearn.neighbors


def knn(data, k=5):
    print("Calculating KNN...")


def compute_knn_features(points, k=5):
    if len(points) == 0:
        return [], [], []

    setX = points["cx"] / points["cx"].max()
    setY = points["cy"] / points["cy"].max()
    setZ = points["cz"] / points["cz"].max()

    coords = np.stack([setX, setY, setZ], axis=1)

    tree = sklearn.neighbors.KDTree(coords)
    k_use = min(k, len(coords))
    distances, indices = tree.query(coords, k=k_use)

    distAv = []
    pWT = []
    pC = []

    for i in range(len(indices)):
        neighbor_indices = indices[i]
        neighbor_phenotypes = points.iloc[neighbor_indices]["phenotype"]

        count_WT = sum(p.startswith("WT") for p in neighbor_phenotypes)
        count_C = sum(p.startswith("C") for p in neighbor_phenotypes)
        total = len(neighbor_phenotypes)

        distAv.append(np.median(distances[i]))
        pWT.append(count_WT)
        pC.append(count_C)

    return distAv, pWT, pC
