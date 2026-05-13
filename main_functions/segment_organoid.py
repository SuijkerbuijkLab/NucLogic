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
from utils.compensate_voxel_size import compensate_voxel_size
from utils.properties_channel import properties_channel
from utils.offset_image import offset_image
from utils.expand_mask import expand_mask_3d
from utils.tiff_metadata import load_tiff_movie_and_metadata


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
    measure_intensity_in="Nuclei",
    cytoplasm_size=5,
    save_measurement_mask=False,
    user_voxel_size=(1.0, 1.0, 1.0),
):
    nuclei_channels = []

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
    metadata_missing = False
    if input_file.endswith(".ims"):
        time_interval = get_time_interval(input_file)
        loaded_movie = ims(input_file)  # TCZXY
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
        voxel_size = loaded_movie.resolution
        if voxel_size is None:
            metadata_missing = True
            voxel_size = (1.0, 1.0, 1.0)
    elif input_file.endswith(".tif") or input_file.endswith(".tiff"):
        loaded_movie, voxel_size, time_interval, metadata_missing = (
            load_tiff_movie_and_metadata(input_file)
        )

    else:
        raise ValueError(
            "Unsupported file format in sample directory. Please provide a .tif, .tiff, or .ims file."
        )

    voxel_size = _resolve_voxel_size(voxel_size, metadata_missing)

    segmented_movie = []
    measurement_mask_movie = []
    properties = []
    print(f"Loaded movie shape (T, C, Z, Y, X): {loaded_movie.shape}")
    # print(f"full movie shape {loaded_movie.shape}")
    for timepoint in alive_it(
        range(loaded_movie.shape[0]), title="Segmenting organoid"
    ):
        frame = loaded_movie[timepoint]
        if frame.ndim != 4:  # If no channel dimension, add one
            frame = np.expand_dims(frame, axis=0)

        if len(nuclei_channels) == 2:
            frame_nuclei = frame[nuclei_channels]
            frame_nuclei = np.max(frame_nuclei, axis=0)
            frame_nuclei_1 = frame[nuclei_channels[0]]
            frame_nuclei_2 = frame[nuclei_channels[1]]
        else:
            frame_nuclei = frame[
                nuclei_channels[0]
            ]  # (Z, Y, X) — avoid extra leading dim
        print(f"frame shape {frame.shape}")
        print(f"frame nuclei shape {frame_nuclei.shape}")
        # print(f"Segmenting timepoint {timepoint} with shape {frame_nuclei.shape}...")

        # This function will segment every slice in the frame individually using the cell model, and then links them back into a 3D array
        # stdout silenced to stop printing random stuff
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        segmented_stack = segment(frame_nuclei, cell_model)

        sys.stdout = old_stdout  # reset old stdout
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
            segmented_stack_stitched = stitch_3d(
                segmented_stack,
                image1=frame_nuclei_1,
                image_type_1="nuclei_1",
                image2=frame_nuclei_2,
                image_type_2="nuclei_2",
                breaking_threshold=breaking_threshold,
            )
        else:
            segmented_stack_stitched = stitch_3d(
                segmented_stack,
                image1=frame_nuclei,
                image_type_1="nuclei",
                breaking_threshold=breaking_threshold,
            )

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
                    1 / voxel_size[2],
                    1 / voxel_size[1],
                ),
                metadata={
                    "unit": "um",
                    "axes": "TZCYX",
                    "spacing": voxel_size[0],
                    "finterval": time_interval,
                    "tunit": "h",
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
                    1 / voxel_size[2],
                    1 / voxel_size[1],
                ),
                metadata={
                    "unit": "um",
                    "axes": "TZCYX",
                    "spacing": voxel_size[0],
                    "finterval": time_interval,
                    "tunit": "h",
                },
                compression="zlib",
                compressionargs={"level": 8},
            )

        # Get properties of the masked nuclei, such as volume and location of every cell
        print(f"Measuring mask properties for timepoint {timepoint}...")
        props = properties_mask(segmented_stack_stitched)
        # Compensate for voxel size to get real world xyz distance values instead of pixel values
        props = compensate_voxel_size(props, voxel_size)

        if measure_region == "nuclei":
            mask_for_intensity = segmented_stack_stitched
            region_tag = "nuclei"
        elif measure_region == "whole cell":
            print(
                f"Expanding nuclei masks to whole cell masks with cytoplasm size {cytoplasm_size} um for timepoint {timepoint}..."
            )
            mask_for_intensity = expand_mask_3d(
                segmented_stack_stitched, dilation_size_um=cytoplasm_size
            )
            region_tag = "whole_cell"
        elif measure_region == "cytoplasm":
            print(
                f"Expanding nuclei masks to cytoplasm masks with cytoplasm size {cytoplasm_size} um for timepoint {timepoint}..."
            )
            whole_cell_mask = expand_mask_3d(
                segmented_stack_stitched, dilation_size_um=cytoplasm_size
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
            print(
                f"Measuring properties for channel {channel} at timepoint {timepoint}..."
            )
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
        resolution=(1 / voxel_size[2], 1 / voxel_size[1]),
        metadata={
            "unit": "um",
            "axes": "TZYX",
            "spacing": voxel_size[0],
            "finterval": time_interval,
            "tunit": "h",
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
            resolution=(1 / voxel_size[2], 1 / voxel_size[1]),
            metadata={
                "unit": "um",
                "axes": "TZYX",
                "spacing": voxel_size[0],
                "finterval": time_interval,
                "tunit": "h",
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
