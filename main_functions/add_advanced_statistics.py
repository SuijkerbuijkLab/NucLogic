import os
import pandas as pd
import numpy as np
import tifffile
from utils.get_extra_mask_properties import get_extra_mask_properties
from utils.compute_shape_descriptors import (
    compute_shape_descriptors,
    DERIVED_SHAPE_DESCRIPTORS,
)
from utils.find_input_file import find_input_file
from utils.expand_mask import expand_mask_3d
from utils.offset_image import offset_image
from utils.compute_knn_features import compute_knn_features
from utils.get_touching_neigbhours_3d import get_touching_neighbors_3d
from utils.load_image import as_numpy, load_image
from utils.save_as_tiff import save_as_tiff
from utils.voxel_size import resolve_voxel_size


def add_advanced_statistics(
    input_directory,
    extra_props,
    channel_names,
    do_crop_sample=False,
    measure_intensity_in="Nuclei",
    cytoplasm_size=5,
    save_measurement_mask=False,
    user_voxel_size=None,
    calculate_neighbour_statistics=False,
    use_knn_neighbours=False,
    knn_list=None,
    use_touching_neighbours_3d=False,
    touching_dilation_um=None,
):
    extra_props = extra_props or []

    should_run_knn = (
        calculate_neighbour_statistics and use_knn_neighbours and bool(knn_list)
    )
    should_run_touching = (
        calculate_neighbour_statistics
        and use_touching_neighbours_3d
        and touching_dilation_um is not None
    )

    if not extra_props and not should_run_knn and not should_run_touching:
        print(
            "No extra properties selected and neighbour statistics are disabled. Nothing to add."
        )
        return

    def _format_param_token(value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return format(numeric_value, "g").replace(".", "p")

    def _is_intensity_property(prop_name):
        return (
            prop_name.startswith("intensity_")
            or prop_name.startswith("centroid_weighted")
            or prop_name.startswith("moments_weighted")
            or prop_name == "image_intensity"
        )


    intensity_props = [p for p in extra_props if _is_intensity_property(p)]
    non_intensity_props = [p for p in extra_props if not _is_intensity_property(p)]
    # Derived, voxel-size-scaled descriptors are computed separately from the
    # generic getattr-based regionprops path (they are not regionprops attrs).
    derived_shape_props = [
        p for p in non_intensity_props if p in DERIVED_SHAPE_DESCRIPTORS
    ]
    shape_props = [
        p for p in non_intensity_props if p not in DERIVED_SHAPE_DESCRIPTORS
    ]

    properties_file = f"{os.path.basename(input_directory)}_properties.tsv"
    if properties_file not in os.listdir(input_directory):
        print(
            f"No properties file found in {input_directory}. Please run the segmentation function first to generate the properties file."
        )
        return

    props = pd.read_csv(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_properties.tsv"
        ),
        sep="\t",
    )

    segmented_file = f"{os.path.basename(input_directory)}_segmented.tif"
    if segmented_file not in os.listdir(input_directory):
        print(
            f"No segmented file found in {input_directory}. Please run the segmentation function first to generate the segmented file."
        )
        return

    segmented_movie = tifffile.imread(os.path.join(input_directory, segmented_file))

    # Expected segmentation shape: T, Z, Y, X (or Z, Y, X for fixed samples)
    if segmented_movie.ndim == 3:
        segmented_movie = segmented_movie[np.newaxis, :, :, :]
    elif segmented_movie.ndim == 5 and segmented_movie.shape[2] == 1:
        # Support legacy TZCYX segmentation with C=1
        segmented_movie = segmented_movie[:, :, 0, :, :]
    elif segmented_movie.ndim != 4:
        raise ValueError(
            f"Unexpected segmented image shape {segmented_movie.shape}; expected TZYX or ZYX."
        )

    if do_crop_sample:
        input_file = [
            os.path.join(input_directory, f)
            for f in os.listdir(input_directory)
            if f.endswith(("_cropped.ims", "_cropped.tif"))
        ]
        input_file = input_file[0] if input_file else None
        name = (
            os.path.basename(input_file).split("_cropped.")[0] if input_file else None
        )
        if input_file is None:
            print(
                f"No cropped file found in {input_directory}. Please run crop_sample first or set do_crop_sample to False."
            )
            return
    else:
        input_file = find_input_file(input_directory)
        name = os.path.basename(input_file).split(".")[0]

    # Load source movie with expected shape T, C, Z, Y, X. Lazy: timepoints are
    # materialised one at a time in the loop below.
    loaded_movie, voxel_size, time_interval, metadata_missing = load_image(input_file)

    voxel_size = resolve_voxel_size(voxel_size, user_voxel_size, metadata_missing)

    measure_region = measure_intensity_in.lower().strip()
    if measure_region == "nuclei":
        mask_to_use = segmented_movie
        region_tag = "nuclei"
    elif measure_region == "whole cell":
        mask_to_use = expand_mask_3d(
            segmented_movie,
            dilation_size_um=float(cytoplasm_size),
            voxel_size=voxel_size,
        )
        region_tag = "whole_cell"
    elif measure_region == "cytoplasm":
        whole_cell_mask = expand_mask_3d(
            segmented_movie,
            dilation_size_um=float(cytoplasm_size),
            voxel_size=voxel_size,
        )
        # Keep expanded cell labels only outside the nucleus to preserve per-cell labeling.
        mask_to_use = np.where(segmented_movie == 0, whole_cell_mask, 0).astype(
            whole_cell_mask.dtype
        )
        region_tag = "cytoplasm"
    else:
        raise ValueError(
            "measure_intensity_in must be one of: 'Nuclei', 'Cytoplasm', 'Whole cell'."
        )

    if save_measurement_mask and measure_region != "nuclei":
        sample_name = os.path.basename(input_directory)
        save_as_tiff(
            os.path.join(input_directory, f"{sample_name}_segmented_{region_tag}.tif"),
            mask_to_use,
            "TZYX",
            voxel_size,
            time_interval,
        )

    touching_expanded_movie = None
    touching_prefix = None
    if should_run_touching:
        touching_token = _format_param_token(touching_dilation_um)
        touching_prefix = f"touching_neighbour_{touching_token}um"
        touching_expanded_movie = expand_mask_3d(
            segmented_movie,
            dilation_size_um=float(touching_dilation_um),
            voxel_size=voxel_size,
        )

        sample_name = os.path.basename(input_directory)
        touching_mask_path = os.path.join(
            input_directory,
            f"{sample_name}_segmented_{touching_prefix}.tif",
        )
        save_as_tiff(
            touching_mask_path,
            touching_expanded_movie,
            "TZYX",
            voxel_size,
            time_interval,
        )
        print(f"Saved touching-neighbour expansion mask to {touching_mask_path}")

    n_timepoints = min(
        len(props["timepoint"].unique()),
        mask_to_use.shape[0],
        loaded_movie.shape[0],
    )
    if n_timepoints == 0:
        print("No timepoints found in properties table.")
        return

    updated_timepoint_tables = []

    for timepoint in sorted(props["timepoint"].unique()):
        timepoint_int = int(timepoint)
        if timepoint >= n_timepoints:
            continue

        time_props = props[props["timepoint"] == timepoint].copy()
        mask_3d = mask_to_use[timepoint_int]
        # Read this timepoint once; everything below works on real numpy.
        frame = as_numpy(loaded_movie[timepoint_int])  # C, Z, Y, X

        def _merge_overwrite(base_df, new_df, key="label"):
            # Recompute columns should overwrite previous values instead of creating _x/_y duplicates.
            overlap = [
                col for col in new_df.columns if col != key and col in base_df.columns
            ]
            if overlap:
                base_df = base_df.drop(columns=overlap)
            return base_df.merge(new_df, on=key, how="left")

        # Add non-intensity/shape props once per timepoint.
        if shape_props:
            shape_df = get_extra_mask_properties(
                mask_3d, extra_props=shape_props, voxel_size=voxel_size
            )
            time_props = _merge_overwrite(time_props, shape_df, key="label")

        # Add derived, voxel-size-scaled shape descriptors once per timepoint.
        if derived_shape_props:
            descriptor_df = compute_shape_descriptors(
                mask_3d, voxel_size, derived_shape_props
            )
            time_props = _merge_overwrite(time_props, descriptor_df, key="label")

        # Add intensity-dependent props for each channel as both raw and background-subtracted.
        if intensity_props:
            for ch_idx, channel_name in enumerate(channel_names):
                if ch_idx >= frame.shape[0]:
                    continue

                raw_intensity_df = get_extra_mask_properties(
                    mask_3d,
                    intensity_image=frame[ch_idx],
                    extra_props=intensity_props,
                    channel_name=f"{channel_name}_{region_tag}_raw",
                    voxel_size=voxel_size,
                )
                time_props = _merge_overwrite(time_props, raw_intensity_df, key="label")

                offset_ch = offset_image(frame[ch_idx], "median")
                bgsub_intensity_df = get_extra_mask_properties(
                    mask_3d,
                    intensity_image=offset_ch,
                    extra_props=intensity_props,
                    channel_name=f"{channel_name}_{region_tag}_background_subtracted",
                    voxel_size=voxel_size,
                )
                time_props = _merge_overwrite(
                    time_props, bgsub_intensity_df, key="label"
                )

        if should_run_knn:
            for knn in knn_list:
                time_props = compute_knn_features(
                    time_props,
                    k=knn,
                    position_columns=["z", "y", "x"],
                    get_phenotype_score=False,
                    label_column="label",
                    distance_column=f"mean_distance_{knn}_KNN",
                    neighbors_column=f"neighbours_{knn}_KNN",
                    add_cell_id=False,
                )

        if should_run_touching:
            touching_mask_3d = touching_expanded_movie[timepoint_int]
            touching_neighbors = get_touching_neighbors_3d(touching_mask_3d)

            label_values = time_props["label"].tolist()
            label_to_xyz = {
                int(row.label): np.array([row.z, row.y, row.x], dtype=float)
                for row in time_props.itertuples(index=False)
            }

            touching_rows = []
            for raw_label in label_values:
                label = int(raw_label)
                neighbors = touching_neighbors.get(label, [])
                mean_distance = np.nan
                if neighbors and label in label_to_xyz:
                    base_xyz = label_to_xyz[label]
                    neighbor_distances = []
                    for neighbor in neighbors:
                        if neighbor not in label_to_xyz:
                            continue
                        distance = np.linalg.norm(base_xyz - label_to_xyz[neighbor])
                        neighbor_distances.append(distance)
                    if neighbor_distances:
                        mean_distance = float(np.mean(neighbor_distances))

                touching_rows.append(
                    {
                        "label": label,
                        f"{touching_prefix}_neighbours": neighbors,
                        f"{touching_prefix}_count": len(neighbors),
                        f"mean_distance_{touching_prefix}": mean_distance,
                    }
                )

            touching_df = pd.DataFrame(touching_rows)
            time_props = _merge_overwrite(time_props, touching_df, key="label")

        updated_timepoint_tables.append(time_props)

    if not updated_timepoint_tables:
        print("No matching timepoints to update.")
        return

    updated_props = pd.concat(updated_timepoint_tables, ignore_index=True)
    updated_props.to_csv(
        os.path.join(
            input_directory, f"{os.path.basename(input_directory)}_properties.tsv"
        ),
        sep="\t",
        index=False,
    )

    print(
        f"Added advanced statistics for {len(updated_timepoint_tables)} timepoints to {properties_file}."
    )
