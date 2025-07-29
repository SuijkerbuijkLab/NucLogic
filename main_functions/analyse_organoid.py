import tifffile
from scipy.ndimage import zoom
from cellpose.utils import stitch3D
import numpy as np
import pandas as pd
from alive_progress import alive_it

import os
import re

import utils


# This is the function of the program that can analyse a single organoid
def analyse_organoid(
    input_directory,  # Folder containing the original tiff or ims file
    cell_model,  # Model that is used to segment cells
    organoid_model,  # Model that is used to segment organoids for cropping
    channel_names,  # Names of the different channels
    croped_existing=False,  # Is there already a folder present that contains cropped tiffs, then we can skip cropping
):
    # Set some parameters for the rest of the scripped
    output_directory_cropped = os.path.join(str(input_directory), "cropped")
    nuclei_channel = next(
        (i for i, ch in enumerate(channel_names) if ch.lower() == "nuclei"), -1
    )
    WT_channel = next(
        (i for i, ch in enumerate(channel_names) if ch.lower() == "wt"), -1
    )
    CRC_channel = next(
        (i for i, ch in enumerate(channel_names) if ch.lower() == "crc"), -1
    )

    # Find the biggest existing IMS or Tiff file in this folder and use it as the input
    input_file = utils.find_input_file(input_directory=input_directory)

    # Get what the file name is without the extension to use for new file generation
    name = os.path.basename(input_file).split(".")[0]

    # If there are no existing cropped tiffs, we will create a max XY projection used to crop the organoid
    # The max XY projection is then used in the crop function to crop every frame of the movie in both XY and XZ to generate way smaller files for segmentation
    if not croped_existing:
        utils.max_project(input_file, input_directory, nuclei=nuclei_channel)
        utils.crop(
            XY_path=os.path.join(input_directory, f"{name}_projXY.tif"),
            input_file=input_file,
            model=organoid_model,
            output_directory=output_directory_cropped,
            nuclei=nuclei_channel,
        )

    # Find all the created cropped tiff files, every file is a 1 frame of the movie
    files = sorted(
        [
            f
            for f in os.listdir(output_directory_cropped)
            if f.startswith("Frame") and f.endswith(".tif")
        ]
    )

    # Used to generate a summary file for every organoid movie
    summary_results = []
    # Used to create padding around every frame to make sure we can stack frames of different XYZ sizes into a single movie for viewing in FIJI / whatever
    max_dims = [0, 0, 0]
    # Used to keep track of every segmented frame to make into a single movie as stated aboves
    segmented_movie = []

    # Loop over every frame, this alive it makes sure we get a nice printed progress bar in the CMD
    for file in alive_it(files, title="Segmenting frames"):

        # Load each frame
        frame_path = os.path.join(output_directory_cropped, file)
        frame = tifffile.imread(frame_path)

        # Select the nuclei channel for segmentation
        frame_nuclei = frame[nuclei_channel, :, :, :]

        # This function will segment every slice in the frame individually using the cell model, and then links them back into a 3D array
        segmented_stack = utils.segment(frame_nuclei, cell_model)

        # This function will stitch the 3D segmentation stack into an actual 3D image where cells are linked through the Z.
        # In this way we actually identify full cell nuclei, instead of single masks per slice
        segmented_stack_stitched = stitch3D(segmented_stack)

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

        # Get properties of every cell in the WT and CRC channels at the masked nuclei locations
        frame_WT = frame[WT_channel, :, :, :]
        df_wt = utils.properties_channel(segmented_stack, frame_WT, "WT")

        frame_CRC = frame[CRC_channel, :, :, :]
        df_crc = utils.properties_channel(segmented_stack, frame_CRC, "CRC")

        # Combine the information of nuclei, WT, and CRC channel into a single dataframe
        df_final = props.merge(df_wt, on="label").merge(df_crc, on="label")

        # On the combined dataframe, we run the phenotype function to calculate for every cell whether it is a WT or CRC cell
        (
            df_final,
            count_phenotype_c1,
            count_phenotype_c2,
            count_phenotype_c1and2,
            count_phenotype_empty,
        ) = utils.phenotype(df_final)

        # Save the dataframe that contains all information and phenotypes of this frame to a file named properties
        os.makedirs(os.path.join(input_directory, "properties"), exist_ok=True)
        output_txt_file = os.path.join(
            input_directory, "properties", f"{file.split('.')[0]}_props.csv"
        )
        df_final.to_csv(output_txt_file, index=False)

        # Append data from this frame to the summary we will save of all frames in the end
        summary_results.append(
            {
                "file": file,
                "phenotype_count_c1": count_phenotype_c1,
                "phenotype_count_c2": count_phenotype_c2,
                "phenotype_count_c1and2": count_phenotype_c1and2,
                "phenotype_count_empty": count_phenotype_empty,
            }
        )

    # Save summary of the whole movie as a csv
    summary_results = pd.DataFrame(summary_results)
    summary_txt = os.path.join(input_directory, "summary_results.csv")
    summary_results.to_csv(summary_txt, index=False)

    # Pad all segmentation mask frames of the movie
    segmented_movie_padded = [
        utils.pad_to_shape(stack, max_dims) for stack in segmented_movie
    ]

    # Stack all segmentation mask frames of the movie across time
    segmented_movie_array = np.stack(segmented_movie_padded, axis=0)

    # Save this segmentation movie
    segmented_movie_tiff = os.path.join(input_directory, "segmented_movie.tif")
    tifffile.imwrite(segmented_movie_tiff, segmented_movie_array)
