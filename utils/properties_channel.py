import numpy as np
import pandas as pd


def properties_channel(mask, image, type=None):
    labels = mask.ravel().astype(np.intp)
    values = image.ravel().astype(np.float64)

    max_label = int(labels.max()) if labels.size > 0 else 0
    col = f"{type.lower()}_mean_intensity" if type else "mean_intensity"
    if max_label == 0:
        return pd.DataFrame(columns=["label", col])

    sums = np.bincount(labels, weights=values, minlength=max_label + 1)
    counts = np.bincount(labels, minlength=max_label + 1)

    # Skip label 0 (background) and any gaps with no voxels
    valid = np.where((counts > 0) & (np.arange(max_label + 1) > 0))[0]
    means = sums[valid] / counts[valid]

    return pd.DataFrame({"label": valid.astype(int), col: means})
