import tifffile

# from cellpose.utils import stitch3D
import numpy as np
import pandas as pd
from alive_progress import alive_it
from utils.load_image import as_numpy, load_image
from utils.save_as_tiff import save_as_tiff
from utils.voxel_size import resolve_voxel_size


import os
import re
import sys
from utils.segment import segment
from utils.stitch_3d import stitch_3d
from utils.properties_mask import properties_mask
from utils.find_input_file import find_input_file
from utils.compensate_voxel_size import compensate_voxel_size
from utils.properties_channel import properties_channel
from utils.offset_image import offset_image


def extract_frame_number(filename):
    match = re.search(r"Frame-(\d+)", filename)
    return int(match.group(1)) if match else -1


def segment_organoid(
    input_directory,  # Folder containing the original tiff or ims file
    cell_model,  # Model that is used to segment cells
    channel_types,
    channel_names,  # Names of the different channels
    breaking_threshold=2.5,
    size_2d_filter_multiplier=15,
    do_crop_sample=False,
    save_frames=True,
    save_segmentation=True,
    user_voxel_size=None,
):
    nuclei_channels = []

    for i, channel_type in enumerate(channel_types):
        if "nuclei" in channel_type.lower():
            nuclei_channels.append(i)

    if do_crop_sample:
        from utils.cropped_files import find_cropped_files

        input_file = find_cropped_files(input_directory)
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

    # Lazy: timepoints are materialised one at a time inside the loop below, so
    # a movie far larger than RAM can be segmented.
    loaded_movie, voxel_size, time_interval, metadata_missing = load_image(input_file)

    voxel_size = resolve_voxel_size(voxel_size, user_voxel_size, metadata_missing)

    segmented_movie = []
    properties = []
    print(f"Loaded movie shape (T, C, Z, Y, X): {loaded_movie.shape}")
    # print(f"full movie shape {loaded_movie.shape}")
    for timepoint in alive_it(
        range(loaded_movie.shape[0]), title="Segmenting organoid"
    ):
        # Read this timepoint once; everything below works on real numpy.
        frame = as_numpy(loaded_movie[timepoint])
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
                size_2d_filter_multiplier=size_2d_filter_multiplier,
            )
        else:
            segmented_stack_stitched = stitch_3d(
                segmented_stack,
                image1=frame_nuclei,
                image_type_1="nuclei",
                breaking_threshold=breaking_threshold,
                size_2d_filter_multiplier=size_2d_filter_multiplier,
            )

        print(
            f"Found {len(np.unique(segmented_stack_stitched)) - 1} cells at timepoint {timepoint}"
        )

        segmented_movie.append(segmented_stack_stitched)

        if save_frames:
            frames_dir = os.path.join(input_directory, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            # Reuse the frame already read above instead of hitting disk again.
            # CZYX -> TCZYX -> TZCYX for tifffile
            frame_tzcyx = np.transpose(np.expand_dims(frame, axis=0), (0, 2, 1, 3, 4))
            save_as_tiff(
                os.path.join(frames_dir, f"Frame-{timepoint}.tif"),
                frame_tzcyx,
                "TZCYX",
                voxel_size,
                time_interval,
            )

        if save_segmentation:
            seg_dir = os.path.join(input_directory, "segmentation")
            os.makedirs(seg_dir, exist_ok=True)
            # segmented_stack_stitched is ZYX; add T and C dims -> TZCYX
            seg_tzcyx = segmented_stack_stitched[np.newaxis, :, np.newaxis, :, :]
            save_as_tiff(
                os.path.join(seg_dir, f"Frame-{timepoint}_segmented.tif"),
                seg_tzcyx,
                "TZCYX",
                voxel_size,
                time_interval,
            )

        # Get properties of the masked nuclei, such as volume and location of every cell
        print(f"Measuring mask properties for timepoint {timepoint}...")
        props = properties_mask(segmented_stack_stitched)
        # Compensate for voxel size to get real world xyz distance values instead of pixel values
        props = compensate_voxel_size(props, voxel_size)

        channel_dfs = {}  # Store channel dataframes during for loop
        for i, channel in enumerate(channel_names):
            print(
                f"Measuring properties for channel {channel} at timepoint {timepoint}..."
            )
            df_ch = properties_channel(
                segmented_stack_stitched, frame[i], f"{channel.lower()}_nuclei_raw"
            )
            offset_ch = offset_image(frame[i], "median")
            df_ch_offset = properties_channel(
                segmented_stack_stitched,
                offset_ch,
                f"{channel.lower()}_nuclei_background_subtracted",
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

    save_as_tiff(
        os.path.join(input_directory, f"{name}_segmented.tif"),
        segmented_movie,
        "TZYX",
        voxel_size,
        time_interval,
    )

    # Convert the data from all frames to a pandas data frame
    properties = pd.concat(properties, ignore_index=True)
    properties.insert(0, "sample", name)
    properties.to_csv(
        os.path.join(input_directory, f"{name}_properties.tsv"), index=False, sep="\t"
    )
