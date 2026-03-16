import os
import pandas as pd
import numpy as np
import tifffile
from utils.get_extra_mask_properties import get_extra_mask_properties
from utils.find_input_file import find_input_file
from imaris_ims_file_reader import ims


def add_advanced_statistics(
    input_directory, extra_props, channel_names, do_crop_sample=False
):
    extra_props = extra_props or []

    if not extra_props:
        print("No extra properties selected. Nothing to add.")
        return

    def _is_intensity_property(prop_name):
        return (
            prop_name.startswith("intensity_")
            or prop_name.startswith("centroid_weighted")
            or prop_name.startswith("moments_weighted")
            or prop_name == "image_intensity"
        )

    intensity_props = [p for p in extra_props if _is_intensity_property(p)]
    shape_props = [p for p in extra_props if not _is_intensity_property(p)]

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

    # Load source movie with expected shape T, C, Z, Y, X
    if input_file.endswith(".ims"):
        loaded_movie = ims(input_file)  # TCZXY
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
    elif input_file.endswith(".tif") or input_file.endswith(".tiff"):
        loaded_movie = tifffile.imread(input_file)
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
        loaded_movie = np.transpose(
            loaded_movie, (0, 2, 1, 3, 4)
        )  # from T,Z,C,Y,X to T,C,Z,Y,X
    else:
        raise ValueError(
            "Unsupported file format in sample directory. Please provide a .tif, .tiff, or .ims file."
        )

    n_timepoints = min(
        len(props["timepoint"].unique()),
        segmented_movie.shape[0],
        loaded_movie.shape[0],
    )
    if n_timepoints == 0:
        print("No timepoints found in properties table.")
        return

    updated_timepoint_tables = []

    for timepoint in sorted(props["timepoint"].unique()):
        if timepoint >= n_timepoints:
            continue

        time_props = props[props["timepoint"] == timepoint].copy()
        mask_3d = segmented_movie[timepoint]
        frame = loaded_movie[timepoint]  # C, Z, Y, X

        # Add non-intensity/shape props once per timepoint.
        if shape_props:
            shape_df = get_extra_mask_properties(mask_3d, extra_props=shape_props)
            time_props = time_props.merge(shape_df, on="label", how="left")

        # Add intensity-dependent props for each channel with channel prefix.
        if intensity_props:
            for ch_idx, channel_name in enumerate(channel_names):
                if ch_idx >= frame.shape[0]:
                    continue
                intensity_df = get_extra_mask_properties(
                    mask_3d,
                    intensity_image=frame[ch_idx],
                    extra_props=intensity_props,
                    channel_name=channel_name,
                )
                time_props = time_props.merge(intensity_df, on="label", how="left")

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
