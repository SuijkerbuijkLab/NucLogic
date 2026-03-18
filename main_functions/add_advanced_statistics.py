import os
import pandas as pd
import numpy as np
import tifffile
from utils.get_extra_mask_properties import get_extra_mask_properties
from utils.find_input_file import find_input_file
from utils.expand_mask import expand_mask
from utils.get_time_interval import get_time_interval
from imaris_ims_file_reader import ims


def add_advanced_statistics(
    input_directory,
    extra_props,
    channel_names,
    do_crop_sample=False,
    measure_intensity_in="Nuclei",
    cytoplasm_size=5,
    save_measurement_mask=False,
    user_voxel_size=(1.0, 1.0, 1.0),
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

    def _to_voxel_tuple(voxel_like):
        try:
            return (
                float(voxel_like[0]),
                float(voxel_like[1]),
                float(voxel_like[2]),
            )
        except (TypeError, ValueError, IndexError):
            return (1.0, 1.0, 1.0)

    def _resolve_voxel_size(measured_voxel, metadata_missing):
        measured = _to_voxel_tuple(measured_voxel)
        user_voxel = _to_voxel_tuple(user_voxel_size)
        user_is_default = np.allclose(user_voxel, (1.0, 1.0, 1.0))
        measured_is_default = np.allclose(measured, (1.0, 1.0, 1.0))
        if not user_is_default and (metadata_missing or measured_is_default):
            print(f"Using user-provided voxel size override: {user_voxel}")
            return user_voxel
        return measured

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

    measure_region = measure_intensity_in.lower().strip()
    if measure_region == "nuclei":
        mask_to_use = segmented_movie
        region_tag = "nuclei"
    elif measure_region == "whole cell":
        mask_to_use = expand_mask(segmented_movie, dilation_size=cytoplasm_size)
        region_tag = "whole_cell"
    elif measure_region == "cytoplasm":
        whole_cell_mask = expand_mask(segmented_movie, dilation_size=cytoplasm_size)
        # Keep expanded cell labels only outside the nucleus to preserve per-cell labeling.
        mask_to_use = np.where(segmented_movie == 0, whole_cell_mask, 0).astype(
            whole_cell_mask.dtype
        )
        region_tag = "cytoplasm"
    else:
        raise ValueError(
            "measure_intensity_in must be one of: 'Nuclei', 'Cytoplasm', 'Whole cell'."
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
    metadata_missing = False
    if input_file.endswith(".ims"):
        time_interval = get_time_interval(input_file)
        loaded_movie = ims(input_file)  # TCZXY
        voxel_size = loaded_movie.resolution
        if voxel_size is None:
            metadata_missing = True
            voxel_size = (1.0, 1.0, 1.0)
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
    elif input_file.endswith(".tif") or input_file.endswith(".tiff"):
        with tifffile.TiffFile(input_file) as tif:
            if tif.is_ome:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(tif.ome_metadata)
                ns = root.tag.split("}")[0].lstrip("{")
                pixels = root.find(f".//{{{ns}}}Pixels")
                z_size = pixels.get("PhysicalSizeZ") if pixels is not None else None
                y_size = pixels.get("PhysicalSizeY") if pixels is not None else None
                x_size = pixels.get("PhysicalSizeX") if pixels is not None else None
                metadata_missing = pixels is None or any(
                    value is None for value in (z_size, y_size, x_size)
                )
                voxel_size = (
                    float(z_size or 1.0),
                    float(y_size or 1.0),
                    float(x_size or 1.0),
                )
                time_interval = float(pixels.get("TimeIncrement", 1.0))
            else:
                metadata_missing = True
                voxel_size = (1.0, 1.0, 1.0)
                time_interval = 1
            loaded_movie = tif.asarray()
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
        loaded_movie = np.transpose(
            loaded_movie, (0, 2, 1, 3, 4)
        )  # from T,Z,C,Y,X to T,C,Z,Y,X
    else:
        raise ValueError(
            "Unsupported file format in sample directory. Please provide a .tif, .tiff, or .ims file."
        )

    voxel_size = _resolve_voxel_size(voxel_size, metadata_missing)

    if save_measurement_mask and measure_region != "nuclei":
        sample_name = os.path.basename(input_directory)
        tifffile.imwrite(
            os.path.join(input_directory, f"{sample_name}_segmented_{region_tag}.tif"),
            mask_to_use,
            bigtiff=True,
            imagej=True,
            resolution=((1 / voxel_size[0]) * 25400, (1 / voxel_size[1]) * 25400),
            metadata={
                "unit": "um",
                "axes": "TZYX",
                "PhysicalSizeX": voxel_size[2],
                "PhysicalSizeXUnit": "um",
                "PhysicalSizeY": voxel_size[1],
                "PhysicalSizeYUnit": "um",
                "PhysicalSizeZ": voxel_size[0],
                "PhysicalSizeZUnit": "um",
                "spacing": voxel_size[0],
                "TimeIncrement": 1 * time_interval,
                "TimeIncrementUnit": "h",
            },
            compression="zlib",
            compressionargs={"level": 8},
        )

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
        if timepoint >= n_timepoints:
            continue

        time_props = props[props["timepoint"] == timepoint].copy()
        mask_3d = mask_to_use[timepoint]
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
                    channel_name=f"{channel_name}_{region_tag}",
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
