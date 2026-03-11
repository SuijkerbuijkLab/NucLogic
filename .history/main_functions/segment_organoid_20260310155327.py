import tifffile

# from cellpose.utils import stitch3D
import numpy as np
import pandas as pd
from alive_progress import alive_it
from imaris_ims_file_reader.ims import ims


import os
import re
import sys

import utils


def extract_frame_number(filename):
    match = re.search(r"Frame-(\d+)", filename)
    return int(match.group(1)) if match else -1


def segment_organoid(
    input_directory,  # Folder containing the original tiff or ims file
    cell_model,  # Model that is used to segment cells
    channel_types,
    channel_names,  # Names of the different channels
    breaking_threshold=2.5,  # Threshold for breaking cells in the stitch_3d function
):
    nuclei_channels = []
    for i, channel_type in enumerate(channel_types):
        if "nuclei" in channel_type.lower():
            nuclei_channels.append(i)

    input_file = utils.find_input_file(input_directory)

    name = os.path.basename(input_file).split(".")[0]

    # Get metadata of the time interval of the movie
    if input_file.endswith(".ims"):
        time_interval = utils.get_time_interval(input_file)
        loaded_movie = ims(input_file)  # TCZXY
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
        voxel_size = loaded_movie.resolution
    elif input_file.endswith(".tif") or input_file.endswith(".tiff"):
        loaded_movie = tifffile.imread(input_file)  # TZCXY
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
        loaded_movie = np.transpose(loaded_movie, (0, 2, 1, 3, 4))  # convert to TCZXY
        time_interval = 1
        voxel_size = (1.0, 1.0, 1.0)
    else:
        raise ValueError(
            "Unsupported file format in sample directory. Please provide a .tif, .tiff, or .ims file."
        )

    segmented_movie = []
    properties = []
    print(f"full movie shape {loaded_movie.shape}")
    for timepoint in alive_it(
        range(loaded_movie.shape[0]), title="Segmenting organoid"
    ):
        frame = loaded_movie[timepoint]
        if len(nuclei_channels) == 2:
            frame_nuclei = frame[nuclei_channels]
            frame_nuclei = np.max(frame_nuclei, axis=1)
            frame_nuclei_1 = frame[nuclei_channels[0]]
            frame_nuclei_2 = frame[nuclei_channels[1]]
        else:
            frame_nuclei = frame[nuclei_channels]
        print(f"frame shape {frame.shape}")

        # This function will segment every slice in the frame individually using the cell model, and then links them back into a 3D array
        # stdout silenced to stop printing random stuff
        # old_stdout = sys.stdout  # backup current stdout
        # sys.stdout = open(os.devnull, "w")
        segmented_stack = utils.segment(frame_nuclei, cell_model)
        print(segmented_stack.shape, np.unique(segmented_stack))

        # This function will stitch the 3D segmentation stack into an actual 3D image where cells are linked through the Z.
        # In this way we actually identify full cell nuclei, instead of single masks per slice
        if len(nuclei_channels) == 2:
            segmented_stack_stitched, organoid = utils.stitch_3d(
                segmented_stack,
                image1=frame_nuclei_1,
                image_type_1="nuclei_1",
                image2=frame_nuclei_2,
                image_type_2="nuclei_2",
                breaking_threshold=breaking_threshold,
            )
        else:
            segmented_stack_stitched, organoid = utils.stitch_3d(
                segmented_stack,
                image1=frame_nuclei,
                image_type_1="nuclei",
                breaking_threshold=breaking_threshold,
            )
        # sys.stdout = old_stdout  # reset old stdout

        segmented_movie.append(segmented_stack_stitched)

        # Get properties of the masked nuclei, such as volume and location of every cell
        props = utils.properties_mask(segmented_stack_stitched)
        # Compensate for voxel size to get real world xyz distance values instead of pixel values
        props = utils.compensate_voxel_size(props, voxel_size)

        channel_dfs = {}  # Store channel dataframes during for loop
        for i, channel in enumerate(channel_names):
            df_ch = utils.properties_channel(
                segmented_stack_stitched, frame[:, i], f"{channel}_raw"
            )
            offset_ch = utils.offset_image(frame[:, i], "median")
            df_ch_offset = utils.properties_channel(
                segmented_stack_stitched, offset_ch, f"{channel}_background_subtracted"
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
    print(segmented_movie.shape)
    segmented_movie = np.expand_dims(
        segmented_movie, axis=2
    )  # add channel dimension back for saving in tiff format as TZCXY
    print(segmented_movie.shape)

    tifffile.imwrite(
        os.path.join(input_directory, f"{name}_segmented.tif"),
        segmented_movie,
        imagej=True,
        resolution=((1 / voxel_size[0]) * 25400, (1 / voxel_size[1]) * 25400),
        metadata={
            "unit": "um",
            "axes": "TZCYX",
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
