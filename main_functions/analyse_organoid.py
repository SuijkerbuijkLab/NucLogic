import tifffile
from scipy.ndimage import zoom
from cellpose.utils import stitch3D
import numpy as np
import pandas as pd
from alive_progress import alive_it
import tables
import itertools
from imaris_ims_file_reader.ims import ims

import os
import re
import sys

import utils


def extract_frame_number(filename):
    match = re.search(r"Frame-(\d+)", filename)
    return int(match.group(1)) if match else -1


# This is the function of the program that can analyse a single organoid
def analyse_organoid(
    input_directory,  # Folder containing the original tiff or ims file
    cell_model,  # Model that is used to segment cells
    organoid_model,  # Model that is used to segment organoids for cropping
    channel_names,  # Names of the different channels
    croped_existing=False,  # Is there already a folder present that contains cropped tiffs, then we can skip cropping
    is_fixed=False,  # Is the data from fixed organoids (single timepoint) or live (multiple timepoints)
):
    # Set some parameters for the rest of the scripped
    output_directory_cropped = os.path.join(str(input_directory), "cropped")

    # Extract channel indices with a dictionary for dynamic access
    channel_indices = {
        ch.lower(): i for i, ch in enumerate(channel_names) if ch.strip()
    }

    # Always process the nuclei channel
    nuclei_channel = channel_indices.get("nuclei", -1)
    if nuclei_channel == -1:
        raise ValueError("Nuclei channel is required.")

    # Find the biggest existing IMS or Tiff file in this folder and use it as the input
    input_file = utils.find_input_file(input_directory=input_directory)

    # Get what the file name is without the extension to use for new file generation
    name = os.path.basename(input_file).split(".")[0]

    # Get metadata of the time interval of the movie
    if input_file.endswith(".ims"):
        # Get the voxel size of the movie
        loaded_movie = ims(input_file)
        voxel_size = loaded_movie.resolution  # (Z, Y, X)

        if not is_fixed:
            with tables.open_file(input_file, "r") as hf:
                time_values = hf.root.DataSetTimes.Time.read()
            timestamps = np.array([row[2] for row in time_values])
            timestamps = (
                timestamps // 3.6e9
            )  # gets data in nanoseconds, calculate to hours
            time_interval = timestamps[1] - timestamps[0]

    else:
        time_interval = 1
        voxel_size = (1.0, 1.0, 1.0)  # Assume isotropic if not provided

    # If there are no existing cropped tiffs, we will create a max XY projection used to crop the organoid
    # The max XY projection is then used in the crop function to crop every frame of the movie in both XY and XZ to generate way smaller files for segmentation
    if not croped_existing:

        if is_fixed:
            proj_XY = utils.max_project(
                input_file,
                input_directory,
                nuclei=nuclei_channel,
                name=name,
                fixed=is_fixed,
            )
            utils.crop_fixed(
                proj_XY=proj_XY,
                input_file=input_file,
                output_directory=output_directory_cropped,
                nuclei=nuclei_channel,
                name=name,
                voxel_size=voxel_size,
            )
        else:
            proj_XY = utils.max_project(
                input_file, input_directory, nuclei=nuclei_channel, name=name
            )
            utils.crop(
                proj_XY=proj_XY,
                input_file=input_file,
                output_directory=output_directory_cropped,
                model=organoid_model,
                nuclei=nuclei_channel,
                name=name,
                voxel_size=voxel_size,
            )

    # Find all the created cropped tiff files, every file is a 1 frame of the movie
    files = sorted(
        [
            f
            for f in os.listdir(output_directory_cropped)
            if f.startswith("Frame") and f.endswith(".tif")
        ]
    )

    files = sorted(files, key=extract_frame_number)

    # Used to create padding around every frame to make sure we can stack frames of different XYZ sizes into a single movie for viewing in FIJI / whatever
    max_dims = [0, 0, 0]
    # Used to keep track of every segmented frame to make into a single movie as stated aboves
    segmented_movie = []
    # Used to keep track of results of every frame
    properties = []

    # Loop over every frame, this alive it makes sure we get a nice printed progress bar in the CMD
    for file in alive_it(files, title="Segmenting frames"):

        # Load each frame
        frame_path = os.path.join(output_directory_cropped, file)
        frame = tifffile.imread(frame_path)

        # Select the nuclei channel for segmentation
        frame_nuclei = frame[:, nuclei_channel, :, :]

        # This function will segment every slice in the frame individually using the cell model, and then links them back into a 3D array
        # stdout silenced to stop printing random stuff
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        segmented_stack = utils.segment(frame_nuclei, cell_model)

        # This function will stitch the 3D segmentation stack into an actual 3D image where cells are linked through the Z.
        # In this way we actually identify full cell nuclei, instead of single masks per slice
        segmented_stack_stitched = stitch3D(segmented_stack)
        sys.stdout = old_stdout  # reset old stdout

        # Save segmentation mask tiffile
        os.makedirs(os.path.join(input_directory, "segmented"), exist_ok=True)
        segmented_tiff_file = os.path.join(
            input_directory, "segmented", f"{file.split('.')[0]}_masks.tif"
        )
        tifffile.imwrite(segmented_tiff_file, segmented_stack)

        # The segmentation mask is saved to generate a full movie later on, the XYZ dimensions of this frame are saved to calculate the padding needed for this movie
        segmented_movie.append(segmented_stack_stitched)
        for i in range(3):
            max_dims[i] = max(max_dims[i], segmented_stack.shape[i])

        # Get properties of the masked nuclei, such as volume and location of every cell
        props = utils.properties_mask(segmented_stack)

        # Compensate for voxel size to get real world xyz distance values instead of pixel values
        props = utils.compensate_voxel_size(props, voxel_size)

        # Get properties of every cell in the WT and CRC channels at the masked nuclei locations
        # We do this on an image where the median value (background) is subtracted from the signal
        channel_dfs = {}  # Store channel dataframes during for loop
        for ch_name, ch_index in channel_indices.items():
            if ch_name == "nuclei":
                continue  # Skip nuclei because we already got data from that via properties_mask
            frame_ch = frame[:, ch_index, :, :]
            # Do a background subtraction on the image
            offset_ch = utils.offset_image(frame_ch, type="median")
            # Get intensity data of this channel at the locations of the nuclei masks
            df_ch = utils.properties_channel(
                segmented_stack, offset_ch, channel_names[ch_index].lower()
            )
            channel_dfs[channel_names[ch_index]] = df_ch

        # Combine the information of nuclei, WT, and CRC channel into a single dataframe
        df_final = props.copy()
        for ch_name, df_ch in channel_dfs.items():
            df_final = df_final.merge(df_ch, on="label")
        df_final.insert(0, "frame", int(re.search(r"\d+", file).group()))

        # Add the ratios and log ratios of all combinations to the data frame
        channels = [ch.lower() for ch in channel_dfs.keys() if ch.lower() != "nuclei"]
        for ch_a, ch_b in itertools.combinations(channels, 2):
            raw_a = f"raw_{ch_a}"
            raw_b = f"raw_{ch_b}"
            ratio_col1 = f"ratio_{ch_a}_{ch_b}"
            log_col1 = f"log_{ratio_col1}"
            ratio_col2 = f"ratio_{ch_b}_{ch_a}"
            log_col2 = f"log_{ratio_col2}"

            df_final[ratio_col1] = df_final.apply(
                lambda row: (row[raw_a] + 1e-6) / (row[raw_b] + 1e-6),
                axis=1,
            )
            df_final[log_col1] = np.log10(df_final[ratio_col1] + 1e-6)
            df_final[ratio_col2] = df_final.apply(
                lambda row: (row[raw_b] + 1e-6) / (row[raw_a] + 1e-6),
                axis=1,
            )
            df_final[log_col2] = np.log10(df_final[ratio_col2] + 1e-6)

        # Save the dataframe that contains all information and phenotypes of this frame to a file named properties
        os.makedirs(os.path.join(input_directory, "properties"), exist_ok=True)
        output_txt_file = os.path.join(
            input_directory, "properties", f"{file.split('.')[0]}_props.csv"
        )
        df_final.to_csv(output_txt_file, index=False)

        # Save the data of this frame to the overview of all frames
        properties.append(df_final)

    # Convert the data from all frames to a pandas data frame
    properties = pd.concat(properties, ignore_index=True)

    # Calculate the cutoff value between WT and CRC cells based on the ratio between their signals
    cutoff = utils.calculate_cutoff(properties, f"log_ratio_wt_crc")

    # Loop again over every frame to calculate the number of WT and CRC cells based on the cutoff
    summary_results = []
    for file in files:
        df = properties[properties["frame"] == int(re.search(r"\d+", file).group())]
        for ch_a, ch_b in itertools.combinations(channels, 2):
            log_col = f"log_ratio_wt_crc"
            crc_count = np.sum(df[log_col] < cutoff)
            wt_count = np.sum(df[log_col] >= cutoff)

        total = crc_count + wt_count

        old_df = pd.read_csv(
            os.path.join(
                input_directory, "properties", f"{file.split('.')[0]}_props.csv"
            )
        )
        old_df["phenotype"] = np.where(old_df[log_col] < cutoff, "crc", "wt")
        old_df = utils.compute_knn_features(old_df, k=5)

        old_df.to_csv(
            os.path.join(
                input_directory, "properties", f"{file.split('.')[0]}_props.csv"
            ),
            index=False,
        )

        results = {
            "organoid": name,
            "file": file,
            "wt_count": wt_count,
            "crc_count": crc_count,
            "%wt": wt_count / total * 100,
            "%crc": crc_count / total * 100,
        }
        summary_results.append(pd.DataFrame([results]))

    if not is_fixed:
        summary_results = pd.concat(summary_results, ignore_index=True)
        summary_results["relative_wt"] = (
            summary_results["wt_count"] / summary_results["wt_count"][0]
        )
        summary_results["relative_crc"] = (
            summary_results["crc_count"] / summary_results["crc_count"][0]
        )

        # Save summary of the whole movie as a csv
        summary_txt = os.path.join(input_directory, "summary_results_organoid.csv")
        summary_results.to_csv(summary_txt, index=False)

        # Generate plots and save these as a report
        report = os.path.join(input_directory, "result_report.pdf")
        utils.generate_report(
            summary_results, time_interval=time_interval, output_path=report
        )

    # Pad all segmentation mask frames of the movie
    segmented_movie_padded = [
        utils.pad_to_shape(stack, max_dims) for stack in segmented_movie
    ]

    # Stack all segmentation mask frames of the movie across time
    segmented_movie_array = np.stack(segmented_movie_padded, axis=0)
    # Add a singleton channel axis → shape becomes TZCYX
    segmented_movie_array = np.expand_dims(segmented_movie_array, axis=2)

    # Save this segmentation movie
    segmented_movie_tiff = os.path.join(input_directory, "segmented_movie.tif")
    tifffile.imwrite(
        segmented_movie_tiff,
        segmented_movie_array,
        imagej=True,
        resolution=((1 / 0.65) * 25400, (1 / 0.65) * 25400),
        metadata={
            "unit": "um",
            "axes": "TZCYX",
            "TimeIncrement": 1,
            "TimeIncrementUnit": "h",
        },
    )
