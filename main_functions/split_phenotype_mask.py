import os
import pandas as pd
import numpy as np

from utils.load_image import load_image
from utils.save_as_tiff import save_as_tiff


def split_phenotype_mask(input_directory, phenotype_1, phenotype_2):
    properties = pd.read_csv(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_properties.tsv"
        ),
        sep="\t",
    )
    phenotype_column_name = f"phenotype_{phenotype_1}_vs_{phenotype_2}"

    if phenotype_column_name not in properties.columns:
        print(
            f"Did not find correct phenotype column ({phenotype_column_name}) in properties file. Please run the calculate_phenotypes function first to generate the phenotype column."
        )
        return

    # The segmented movie carries the voxel size and time interval; read them here
    # so the phenotype masks written below keep the same metadata.
    loaded, voxel_size, time_interval, _ = load_image(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_segmented.tif"
        ),
        lazy=False,
    )
    segmentation = loaded[:, 0]  # T,C,Z,Y,X with C=1 -> T,Z,Y,X

    print(segmentation.shape)

    T, Z, Y, X = segmentation.shape
    # Output: T, Z, 2, Y, X  — channel 0 = phenotype_1 labels, channel 1 = phenotype_2 labels
    new_segmentation = np.zeros((T, Z, 2, Y, X), dtype=segmentation.dtype)

    for timepoint in range(T):
        time_properties = properties[properties["timepoint"] == timepoint]
        time_mask = segmentation[timepoint]  # Z, Y, X

        labels_1 = time_properties[
            time_properties[phenotype_column_name] == phenotype_1
        ]["label"].values
        labels_2 = time_properties[
            time_properties[phenotype_column_name] == phenotype_2
        ]["label"].values

        mask_phenotype_1 = np.isin(time_mask, labels_1)
        mask_phenotype_2 = np.isin(time_mask, labels_2)

        # Preserve original label values in each channel
        new_segmentation[timepoint, :, 0, :, :] = np.where(
            mask_phenotype_1, time_mask, 0
        )
        new_segmentation[timepoint, :, 1, :, :] = np.where(
            mask_phenotype_2, time_mask, 0
        )

    phenotype_1_mask = new_segmentation[:, :, 0, :, :]
    phenotype_2_mask = new_segmentation[:, :, 1, :, :]

    name = os.path.basename(input_directory)
    save_as_tiff(
        os.path.join(input_directory, f"{name}_{phenotype_1}_mask.tif"),
        phenotype_1_mask,
        "TZYX",
        voxel_size,
        time_interval,
    )
    save_as_tiff(
        os.path.join(input_directory, f"{name}_{phenotype_2}_mask.tif"),
        phenotype_2_mask,
        "TZYX",
        voxel_size,
        time_interval,
    )
