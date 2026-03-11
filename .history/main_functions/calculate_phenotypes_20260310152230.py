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
    if not "properties" in os.listdir(input_directory):
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
    elif "fixed" in cutoff_method.lower():
        cutoff = 0

    properties["phenotype"] = properties[log_ratio_col].apply(
        lambda x: f"{phenotype_1}" if x >= cutoff else f"{phenotype_2}"
    )
