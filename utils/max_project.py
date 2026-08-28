import os

import numpy as np
from alive_progress import alive_bar

from utils.load_image import as_numpy, load_image
from utils.save_as_tiff import save_as_tiff


def max_project(file, name="nuclei"):
    movie, voxel_size, time_interval, _ = load_image(file)
    T, C, _, Y, X = movie.shape

    max_data = np.zeros((T, C, Y, X), dtype=movie.dtype)

    # One (T, C) Z-stack per read, so a movie far larger than RAM still projects.
    with alive_bar(T * C, title="Generating Max Z projection") as bar:
        for t in range(T):
            for c in range(C):
                max_data[t, c] = np.max(as_numpy(movie[t, c]), axis=0)
                bar()

    print(f"Saving max projection of shape {max_data.shape}")

    proj_XY_name = os.path.join(os.path.dirname(file), f"{name}_projXY.tif")
    # Z is projected away, so the file carries no Z calibration.
    save_as_tiff(
        proj_XY_name,
        max_data,
        "TCYX",
        (1.0, voxel_size[1], voxel_size[2]),
        time_interval,
    )

    return max_data
