import os
import tifffile
import pandas as pd
import numpy as np


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

    segmentation = tifffile.imread(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_segmented.tif"
        )
    )

    while segmentation.ndim < 5:
        segmentation = np.expand_dims(segmentation, axis=0)
    # segmentation is TZCYX with C=1; squeeze out the channel dim for easy indexing
    # shape: (T, Z, 1, Y, X) -> work on (T, Z, Y, X)
    seg = segmentation[:, :, 0, :, :]  # T, Z, Y, X

    T, Z, Y, X = seg.shape
    # Output: T, Z, 2, Y, X  — channel 0 = phenotype_1 labels, channel 1 = phenotype_2 labels
    new_segmentation = np.zeros((T, Z, 2, Y, X), dtype=segmentation.dtype)

    for timepoint in range(T):
        time_properties = properties[properties["timepoint"] == timepoint]
        time_mask = seg[timepoint]  # Z, Y, X

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

    name = os.path.basename(input_directory)
    tifffile.imwrite(
        os.path.join(
            input_directory,
            f"{name}_split_phenotype_mask_{phenotype_1}_vs_{phenotype_2}.tif",
        ),
        new_segmentation,
        bigtiff=True,
        metadata={
            "unit": "um",
            "axes": "TZCYX",
        },
        compression="zlib",
        compressionargs={"level": 8},
    )
