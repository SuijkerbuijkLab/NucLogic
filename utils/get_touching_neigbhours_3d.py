import numpy as np
from collections import defaultdict


def get_touching_neighbors_3d(labels_zyx, include_diagonals=False):
    """
    labels_zyx: 3D integer label volume (0 = background)
    returns dict: label -> sorted list of touching labels
    """
    pairs = touching_pairs_3d(labels_zyx, include_diagonals=include_diagonals)
    return neighbors_dict_from_pairs(pairs)


def touching_pairs_3d(labels_zyx, include_diagonals=False):
    """
    labels_zyx: 3D integer label volume (0 = background)
    returns: (N,2) array of touching label pairs (sorted, unique)
    """
    L = np.asarray(labels_zyx)
    assert L.ndim == 3, "Expected ZYX volume"

    # 6-connectivity shifts (face touching)
    shifts = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Optional 26-connectivity (includes edge/corner touching)
    if include_diagonals:
        shifts = []
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    if (dz, dy, dx) == (0, 0, 0):
                        continue
                    shifts.append((dz, dy, dx))

    all_pairs = []

    for dz, dy, dx in shifts:
        a = L[
            0 : L.shape[0] - dz if dz else L.shape[0],
            0 : L.shape[1] - dy if dy else L.shape[1],
            0 : L.shape[2] - dx if dx else L.shape[2],
        ]
        b = L[dz:, dy:, dx:]

        # touching if both labeled and different label IDs
        m = (a > 0) & (b > 0) & (a != b)
        if not np.any(m):
            continue

        p = np.stack([a[m], b[m]], axis=1)
        p.sort(axis=1)  # canonical (small,big)
        all_pairs.append(p)

    if not all_pairs:
        return np.empty((0, 2), dtype=L.dtype)

    pairs = np.concatenate(all_pairs, axis=0)
    pairs = np.unique(pairs, axis=0)
    return pairs


def neighbors_dict_from_pairs(pairs):
    """
    pairs: (N,2) array
    returns dict: label -> sorted list of touching labels
    """
    nbrs = defaultdict(set)
    for u, v in pairs:
        nbrs[int(u)].add(int(v))
        nbrs[int(v)].add(int(u))
    return {k: sorted(v) for k, v in nbrs.items()}
