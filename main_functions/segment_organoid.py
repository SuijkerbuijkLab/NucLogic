import tifffile

# from cellpose.utils import stitch3D
import numpy as np
import pandas as pd
from alive_progress import alive_it
from imaris_ims_file_reader.ims import ims


import os
import re
import sys
from utils.segment import segment
from utils.stitch_3d import stitch_3d
from utils.properties_mask import properties_mask
from utils.get_time_interval import get_time_interval
from utils.find_input_file import find_input_file
from utils.max_project import max_project
from utils.compensate_voxel_size import compensate_voxel_size
from utils.properties_channel import properties_channel
from utils.offset_image import offset_image
from utils.get_extra_mask_properties import get_extra_mask_properties
from utils.expand_mask import expand_mask


def extract_frame_number(filename):
    match = re.search(r"Frame-(\d+)", filename)
    return int(match.group(1)) if match else -1


def segment_organoid(
    input_directory,  # Folder containing the original tiff or ims file
    cell_model,  # Model that is used to segment cells
    channel_types,
    channel_names,  # Names of the different channels
    breaking_threshold=2.5,  # Threshold for breaking cells in the stitch_3d function
    do_crop_sample=False,
    save_frames=True,
    save_segmentation=True,
    extra_props=None,  # List of extra properties to extract from the masks in addition to the default properties (label, z, y, x, bounding_box, volume)
    measure_intensity_in="Nuclei",
    cytoplasm_size=5,
    save_measurement_mask=False,
):
    nuclei_channels = []
    extra_props = extra_props or []

    def _is_intensity_property(prop_name):
        return (
            prop_name.startswith("intensity_")
            or prop_name.startswith("centroid_weighted")
            or prop_name.startswith("moments_weighted")
            or prop_name == "image_intensity"
        )

    intensity_props = [p for p in extra_props if _is_intensity_property(p)]
    shape_props = [p for p in extra_props if not _is_intensity_property(p)]
    measure_region = measure_intensity_in.lower().strip()

    for i, channel_type in enumerate(channel_types):
        if "nuclei" in channel_type.lower():
            nuclei_channels.append(i)

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

    # Get metadata of the time interval of the movie
    if input_file.endswith(".ims"):
        time_interval = get_time_interval(input_file)
        loaded_movie = ims(input_file)  # TCZXY
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
        voxel_size = loaded_movie.resolution
    elif input_file.endswith(".tif") or input_file.endswith(".tiff"):
        with tifffile.TiffFile(input_file) as tif:
            if tif.is_ome:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(tif.ome_metadata)
                ns = root.tag.split("}")[0].lstrip("{")
                pixels = root.find(f".//{{{ns}}}Pixels")
                voxel_size = (
                    float(pixels.get("PhysicalSizeZ", 1.0)),
                    float(pixels.get("PhysicalSizeY", 1.0)),
                    float(pixels.get("PhysicalSizeX", 1.0)),
                )
                time_interval = float(pixels.get("TimeIncrement", 1.0))
            else:
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

    segmented_movie = []
    measurement_mask_movie = []
    properties = []
    print(f"Loaded movie shape (T, C, Z, Y, X): {loaded_movie.shape}")
    # print(f"full movie shape {loaded_movie.shape}")
    for timepoint in alive_it(
        range(loaded_movie.shape[0]), title="Segmenting organoid"
    ):
        frame = loaded_movie[timepoint]
        if len(nuclei_channels) == 2:
            frame_nuclei = frame[nuclei_channels]
            frame_nuclei = np.max(frame_nuclei, axis=0)
            frame_nuclei_1 = frame[nuclei_channels[0]]
            frame_nuclei_2 = frame[nuclei_channels[1]]
        else:
            frame_nuclei = frame[
                nuclei_channels[0]
            ]  # (Z, Y, X) — avoid extra leading dim
        # print(f"frame shape {frame.shape}")
        # print(f"Segmenting timepoint {timepoint} with shape {frame_nuclei.shape}...")

        # This function will segment every slice in the frame individually using the cell model, and then links them back into a 3D array
        # stdout silenced to stop printing random stuff
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        segmented_stack = segment(frame_nuclei, cell_model)
        # print(segmented_stack.shape, np.unique(segmented_stack))
        # tifffile.imwrite(
        #     os.path.join(input_directory, f"segmented_stack_timepoint_{timepoint}.tif"),
        #     segmented_stack,
        #     imagej=True,
        #     metadata={"axes": "ZYX"},
        #     compression="zlib",
        #     compressionargs={"level": 8},
        # )

        # This function will stitch the 3D segmentation stack into an actual 3D image where cells are linked through the Z.
        # In this way we actually identify full cell nuclei, instead of single masks per slice
        if len(nuclei_channels) == 2:
            segmented_stack_stitched, organoid = stitch_3d(
                segmented_stack,
                image1=frame_nuclei_1,
                image_type_1="nuclei_1",
                image2=frame_nuclei_2,
                image_type_2="nuclei_2",
                breaking_threshold=breaking_threshold,
            )
        else:
            segmented_stack_stitched, organoid = stitch_3d(
                segmented_stack,
                image1=frame_nuclei,
                image_type_1="nuclei",
                breaking_threshold=breaking_threshold,
            )
        sys.stdout = old_stdout  # reset old stdout

        print(
            f"Found {len(np.unique(segmented_stack_stitched)) - 1} cells at timepoint {timepoint}"
        )

        segmented_movie.append(segmented_stack_stitched)

        if save_frames:
            frames_dir = os.path.join(input_directory, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            frame_czyx = loaded_movie[timepoint]  # CZYX
            # CZYX -> TCZYX -> TZCYX for tifffile
            frame_tzcyx = np.transpose(
                np.expand_dims(frame_czyx, axis=0), (0, 2, 1, 3, 4)
            )
            tifffile.imwrite(
                os.path.join(frames_dir, f"Frame-{timepoint}.tif"),
                frame_tzcyx,
                bigtiff=True,
                resolution=(
                    (1 / voxel_size[0]) * 25400,
                    (1 / voxel_size[1]) * 25400,
                ),
                metadata={
                    "unit": "um",
                    "axes": "TZCYX",
                    "PhysicalSizeX": voxel_size[2],
                    "PhysicalSizeXUnit": "um",
                    "PhysicalSizeY": voxel_size[1],
                    "PhysicalSizeYUnit": "um",
                    "PhysicalSizeZ": voxel_size[0],
                    "PhysicalSizeZUnit": "um",
                    "spacing": voxel_size[0],
                },
                compression="zlib",
                compressionargs={"level": 8},
            )

        if save_segmentation:
            seg_dir = os.path.join(input_directory, "segmentation")
            os.makedirs(seg_dir, exist_ok=True)
            # segmented_stack_stitched is ZYX; add T and C dims -> TZCYX
            seg_tzcyx = segmented_stack_stitched[np.newaxis, :, np.newaxis, :, :]
            tifffile.imwrite(
                os.path.join(seg_dir, f"Frame-{timepoint}_segmented.tif"),
                seg_tzcyx,
                bigtiff=True,
                resolution=(
                    (1 / voxel_size[0]) * 25400,
                    (1 / voxel_size[1]) * 25400,
                ),
                metadata={
                    "unit": "um",
                    "axes": "TZCYX",
                    "PhysicalSizeX": voxel_size[2],
                    "PhysicalSizeXUnit": "um",
                    "PhysicalSizeY": voxel_size[1],
                    "PhysicalSizeYUnit": "um",
                    "PhysicalSizeZ": voxel_size[0],
                    "PhysicalSizeZUnit": "um",
                    "spacing": voxel_size[0],
                },
                compression="zlib",
                compressionargs={"level": 8},
            )

        # Get properties of the masked nuclei, such as volume and location of every cell
        props = properties_mask(segmented_stack_stitched)
        # Compensate for voxel size to get real world xyz distance values instead of pixel values
        props = compensate_voxel_size(props, voxel_size)

        if measure_region == "nuclei":
            mask_for_intensity = segmented_stack_stitched
            region_tag = "nuclei"
        elif measure_region == "whole cell":
            mask_for_intensity = expand_mask(
                segmented_stack_stitched, dilation_size=cytoplasm_size
            )
            region_tag = "whole_cell"
        elif measure_region == "cytoplasm":
            whole_cell_mask = expand_mask(
                segmented_stack_stitched, dilation_size=cytoplasm_size
            )
            mask_for_intensity = np.where(
                segmented_stack_stitched == 0, whole_cell_mask, 0
            ).astype(whole_cell_mask.dtype)
            region_tag = "cytoplasm"
        else:
            raise ValueError(
                "measure_intensity_in must be one of: 'Nuclei', 'Cytoplasm', 'Whole cell'."
            )

        if save_measurement_mask and measure_region != "nuclei":
            measurement_mask_movie.append(mask_for_intensity)

        channel_dfs = {}  # Store channel dataframes during for loop
        for i, channel in enumerate(channel_names):
            df_ch = properties_channel(
                mask_for_intensity, frame[i], f"{channel.lower()}_raw"
            )
            offset_ch = offset_image(frame[i], "median")
            df_ch_offset = properties_channel(
                mask_for_intensity,
                offset_ch,
                f"{channel.lower()}_background_subtracted",
            )
            channel_dfs[channel] = pd.merge(df_ch, df_ch_offset, on="label")

        if shape_props:
            extra_shape_df = get_extra_mask_properties(
                segmented_stack_stitched, extra_props=shape_props
            )
            props = props.merge(extra_shape_df, on="label", how="left")

        if intensity_props:
            for ch_idx, channel in enumerate(channel_names):
                if ch_idx >= frame.shape[0]:
                    continue
                extra_intensity_df = get_extra_mask_properties(
                    mask_for_intensity,
                    intensity_image=frame[ch_idx],
                    extra_props=intensity_props,
                    channel_name=f"{channel}_{region_tag}",
                )
                props = props.merge(extra_intensity_df, on="label", how="left")

        # Combine all channel dataframes into one dataframe for the timepoint
        df_timepoint = props.copy()
        for channel in channel_names:
            df_timepoint = df_timepoint.merge(
                channel_dfs[channel], on="label", how="left"
            )
        df_timepoint["timepoint"] = timepoint
        df_timepoint["time"] = timepoint * time_interval
        properties.append(df_timepoint)

    segmented_movie = np.stack(segmented_movie, axis=0)
    print("Segmented movie shape (T, Z, Y, X):", segmented_movie.shape)

    tifffile.imwrite(
        os.path.join(input_directory, f"{name}_segmented.tif"),
        segmented_movie,
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

    if save_measurement_mask and measure_region != "nuclei" and measurement_mask_movie:
        measurement_mask_movie = np.stack(measurement_mask_movie, axis=0)
        tifffile.imwrite(
            os.path.join(input_directory, f"{name}_segmented_{region_tag}.tif"),
            measurement_mask_movie,
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

    # Convert the data from all frames to a pandas data frame
    properties = pd.concat(properties, ignore_index=True)
    properties.insert(0, "sample", name)
    properties.to_csv(
        os.path.join(input_directory, f"{name}_properties.tsv"), index=False, sep="\t"
    )
