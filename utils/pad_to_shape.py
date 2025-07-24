import numpy as np


def pad_to_shape(array, target_shape):
    pad_width = []
    for dim, target in zip(array.shape, target_shape):
        total_pad = target - dim
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    return np.pad(array, pad_width, mode="constant", constant_values=0)
