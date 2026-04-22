import os
import pandas as pd

from utils.compute_knn_features import compute_phenotype_similarity_from_neighbors


def add_phenotype_similarity(
    input_directory,
    neighbors_column,
    output_column,
    phenotype_column=None,
    label_column="label",
):
    properties_file = f"{os.path.basename(input_directory)}_properties.tsv"
    properties_path = os.path.join(input_directory, properties_file)

    if not os.path.exists(properties_path):
        print(
            f"No properties file found in {input_directory}. Please run segmentation first."
        )
        return

    if not neighbors_column:
        print(
            "No neighbors column provided. Skipping phenotype similarity calculation."
        )
        return

    properties = pd.read_csv(properties_path, sep="\t")

    if neighbors_column not in properties.columns:
        print(
            f"No KNN neighbor column '{neighbors_column}' found. Skipping phenotype similarity calculation."
        )
        return

    if phenotype_column is None:
        phenotype_columns = [
            col for col in properties.columns if col.startswith("phenotype_")
        ]
        if not phenotype_columns:
            print(
                "No phenotype column found. Run phenotype calling first or disable phenotype similarity."
            )
            return
        phenotype_column = sorted(phenotype_columns)[-1]

    if phenotype_column not in properties.columns:
        print(
            f"Phenotype column '{phenotype_column}' not found. Skipping phenotype similarity calculation."
        )
        return

    if "timepoint" in properties.columns:
        updated = []
        for _, frame_df in properties.groupby("timepoint", sort=True):
            updated.append(
                compute_phenotype_similarity_from_neighbors(
                    frame_df,
                    neighbors_column=neighbors_column,
                    phenotype_column=phenotype_column,
                    label_column=label_column,
                    output_column=output_column,
                )
            )
        properties = pd.concat(updated, ignore_index=True)
    else:
        properties = compute_phenotype_similarity_from_neighbors(
            properties,
            neighbors_column=neighbors_column,
            phenotype_column=phenotype_column,
            label_column=label_column,
            output_column=output_column,
        )

    properties.to_csv(properties_path, sep="\t", index=False)
    print(
        f"Added phenotype similarity score column '{output_column}' using neighbors in '{neighbors_column}'."
    )
