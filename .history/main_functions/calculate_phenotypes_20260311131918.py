from utils import calculate_cutoff
import os
import pandas as pd
import numpy as np


def calculate_phenotypes(
    input_directory,
    phenotype_1,
    phenotype_2,
    cutoff_method,
    raw_or_background_subtracted,
):
    properties_file = f"{os.path.basename(input_directory)}_properties.tsv"
    if properties_file not in os.listdir(input_directory):
        print(
            f"No properties file found in {input_directory}. Please run the segmentation function first to generate the properties file."
        )
        print(os.listdir(input_directory))
        return

    if "raw" in raw_or_background_subtracted.lower():
        phenotype_1_column = f"{phenotype_1}_raw"
        phenotype_2_column = f"{phenotype_2}_raw"
    elif "background_subtracted" in raw_or_background_subtracted.lower():
        phenotype_1_column = f"{phenotype_1}_background_subtracted"
        phenotype_2_column = f"{phenotype_2}_background_subtracted"

    properties = pd.read_csv(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_properties.tsv"
        ),
        sep="\t",
    )

    ratio_col = f"ratio_{phenotype_1_column.lower()}_{phenotype_2_column.lower()}"
    log_ratio_col = f"log_{ratio_col}"

    properties[ratio_col] = properties.apply(
        lambda row: (row[phenotype_1_column.lower()] + 1e-6)
        / (row[phenotype_2_column.lower()] + 1e-6),
        axis=1,
    )
    properties[log_ratio_col] = np.log10(properties[ratio_col] + 1e-6)

    if "automatic" in cutoff_method.lower():
        cutoff = calculate_cutoff(properties, log_ratio_col)

        count_phenotype_1 = np.sum(properties[log_ratio_col] > cutoff)
        count_phenotype_2 = np.sum(properties[log_ratio_col] <= cutoff)
        total = count_phenotype_1 + count_phenotype_2

        # if more than 97% of the cells are of one phenotype, assign all cells to that phenotype to avoid misclassification due to noise
        if count_phenotype_1 / total > 0.97:
            properties["phenotype"] = phenotype_1
        elif count_phenotype_2 / total > 0.97:
            properties["phenotype"] = phenotype_2
        else:
            properties["phenotype"] = properties[log_ratio_col].apply(
                lambda x: f"{phenotype_1}" if x >= cutoff else f"{phenotype_2}"
            )
    elif "fixed" in cutoff_method.lower():
        cutoff = 0
        properties["phenotype"] = properties[log_ratio_col].apply(
            lambda x: f"{phenotype_1}" if x >= cutoff else f"{phenotype_2}"
        )

    properties.to_csv(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_properties.tsv"
        ),
        index=False,
        sep="\t",
    )
