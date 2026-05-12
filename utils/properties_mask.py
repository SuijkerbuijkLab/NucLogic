import numpy as np
import pandas as pd


def properties_mask(image):
    image = np.asarray(image)
    z, y, x = np.nonzero(image)
    if z.size == 0:
        return pd.DataFrame(columns=["label", "z", "y", "x", "bounding_box", "volume"])

    labels = image[z, y, x]
    order = np.argsort(labels, kind="stable")
    labels_s = labels[order]
    z_s, y_s, x_s = z[order], y[order], x[order]

    unique_labels, idx, counts = np.unique(labels_s, return_index=True, return_counts=True)

    z_mean = np.add.reduceat(z_s.astype(np.float64), idx) / counts
    y_mean = np.add.reduceat(y_s.astype(np.float64), idx) / counts
    x_mean = np.add.reduceat(x_s.astype(np.float64), idx) / counts

    z_min = np.minimum.reduceat(z_s, idx)
    y_min = np.minimum.reduceat(y_s, idx)
    x_min = np.minimum.reduceat(x_s, idx)
    z_max = np.maximum.reduceat(z_s, idx)
    y_max = np.maximum.reduceat(y_s, idx)
    x_max = np.maximum.reduceat(x_s, idx)

    bboxes = list(zip(
        z_min.tolist(), y_min.tolist(), x_min.tolist(),
        (z_max + 1).tolist(), (y_max + 1).tolist(), (x_max + 1).tolist(),
    ))

    return pd.DataFrame({
        "label": unique_labels.astype(int),
        "z": z_mean,
        "y": y_mean,
        "x": x_mean,
        "bounding_box": bboxes,
        "volume": counts.astype(int),
    })
