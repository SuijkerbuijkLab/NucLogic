from utils.calculate_cutoff import calculate_cutoff
import os
import pandas as pd
import numpy as np


def phenotype_column_name(phenotype_1, region_1, phenotype_2, region_2):
    """Name of the phenotype call column. Non-nuclei regions are part of the
    name so runs measured in different regions do not overwrite each other."""

    def tag(channel, region):
        return channel if region == "nuclei" else f"{channel}_{region}"

    return f"phenotype_{tag(phenotype_1, region_1)}_vs_{tag(phenotype_2, region_2)}".lower()


def _intensity_column(columns, channel, region, suffix):
    """Region-tagged intensity column, falling back to the untagged name used
    by properties files written before regions were part of the name."""
    tagged = f"{channel}_{region}_{suffix}_mean_intensity".lower()
    if tagged in columns:
        return tagged
    legacy = f"{channel}_{suffix}_mean_intensity".lower()
    if region == "nuclei" and legacy in columns:
        return legacy
    return None


def calculate_phenotypes(
    input_directory,
    phenotype_1,
    phenotype_2,
    cutoff_method,
    custom_cutoff,
    raw_or_background_subtracted,
    phenotype_1_region="nuclei",
    phenotype_2_region="nuclei",
):
    phenotype_column = phenotype_column_name(
        phenotype_1, phenotype_1_region, phenotype_2, phenotype_2_region
    )
    properties_file = f"{os.path.basename(input_directory)}_properties.tsv"
    if properties_file not in os.listdir(input_directory):
        print(
            f"No properties file found in {input_directory}. Please run the segmentation function first to generate the properties file."
        )
        print(os.listdir(input_directory))
        return

    suffix = (
        "raw"
        if "raw" in raw_or_background_subtracted.lower()
        else "background_subtracted"
    )

    properties = pd.read_csv(
        os.path.join(input_directory, properties_file),
        sep="\t",
    )
    properties.columns = [col.lower() for col in properties.columns]

    column_1 = _intensity_column(
        properties.columns, phenotype_1, phenotype_1_region, suffix
    )
    column_2 = _intensity_column(
        properties.columns, phenotype_2, phenotype_2_region, suffix
    )
    missing = [
        f"{channel} in {region}"
        for channel, region, column in (
            (phenotype_1, phenotype_1_region, column_1),
            (phenotype_2, phenotype_2_region, column_2),
        )
        if column is None
    ]
    if missing:
        print(
            f"No {suffix} mean intensity found for {' and '.join(missing)}. "
            "Re-run segmentation or statistics with that region measured."
        )
        return

    ratio_col = f"ratio_{column_1}_{column_2}"
    log_ratio_col = f"log_{ratio_col}"
    properties[ratio_col] = (properties[column_1] + 1e-6) / (
        properties[column_2] + 1e-6
    )
    properties[log_ratio_col] = np.log10(properties[ratio_col] + 1e-6)

    log_ratio = properties[log_ratio_col]
    # Cells with no voxels in the chosen region (an empty cytoplasm shell) have no
    # intensity; leave them unassigned instead of silently calling them phenotype 2.
    valid = log_ratio.notna()
    if not valid.all():
        print(
            f"{int((~valid).sum())} cells have no intensity in the selected region; leaving their phenotype empty."
        )

    if "automatic" in cutoff_method.lower():
        cutoff = calculate_cutoff(properties, log_ratio_col)
        total = int(valid.sum())
        above = int((log_ratio[valid] > cutoff).sum())
        # If more than 97% of the cells fall on one side, assign them all to that
        # phenotype to avoid misclassification due to noise.
        if total and above / total > 0.97:
            cutoff = -np.inf
        elif total and (total - above) / total > 0.97:
            cutoff = np.inf
    elif "fixed" in cutoff_method.lower():
        cutoff = 0
    elif "custom cutoff" in cutoff_method.lower():
        cutoff = float(custom_cutoff)
    else:
        raise ValueError(f"Unsupported cutoff method: {cutoff_method}")

    properties[phenotype_column] = np.where(
        valid, np.where(log_ratio >= cutoff, phenotype_1, phenotype_2), None
    )

    properties.to_csv(
        os.path.join(input_directory, properties_file),
        index=False,
        sep="\t",
    )
