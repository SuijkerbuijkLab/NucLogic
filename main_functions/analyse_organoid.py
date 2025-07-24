"""
The part of the program you need to run
"""

import tifffile
from scipy.ndimage import zoom
from cellpose.utils import stitch3D
import numpy as np
import pandas as pd
from alive_progress import alive_it

import os
import re

import utils


def analyse_organoid(
    input_directory,
    cell_model,
    organoid_model,
    channel_names,
    croped_existing=False,
):
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

    input_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".ims") or f.endswith(".tif")
    ]

    if not input_files:
        print(f"Warning: No IMS file found in {input_directory}. Skipping this folder.")
        return

    input_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    input_file = input_files[0]
    name = os.path.basename(input_file).split(".")[0]

    if not croped_existing:
        utils.max_project(input_file, input_directory, nuclei=nuclei_channel)
        utils.crop(
            os.path.join(input_directory, f"{name}_projXY.tif"),
            organoid_model,
            output_directory_cropped,
            organoid_model,
        )

    files = sorted(
        [
            f
            for f in os.listdir(output_directory_cropped)
            if f.startswith("Channel-ref") and f.endswith(".tif")
        ]
    )

    summary_results = []
    max_dims = [0, 0, 0]
    segmented_movie = []

    for file in alive_it(files, title="Segmenting frames"):
        # Load each frame
        frame_path = os.path.join(output_directory_cropped, file)
        frame = tifffile.imread(frame_path)
        frame_nuclei = frame[nuclei_channel, :, :, :]

        segmented_stack = utils.segment(
            frame_nuclei, cell_model
        )  # Segment per Z using the model
        segmented_stack = stitch3D(segmented_stack)  # Connect masks across Z

        # Save segmented tiffile
        os.makedirs(os.path.join(input_directory, "segmented"), exist_ok=True)
        segmented_tiff_file = os.path.join(
            input_directory, "segmented", f"{file.split('.')[0]}_masks.tif"
        )
        tifffile.imwrite(segmented_tiff_file, segmented_stack)
        segmented_movie.append(segmented_stack)
        for i in range(3):
            max_dims[i] = max(max_dims[i], segmented_stack.shape[i])

        # Get properties of the masked nuclei
        props = utils.properties_mask(segmented_stack)

        # Get properties of the WT and CRC channels at the masked nuclei locations
        frame_WT = frame[WT_channel, :, :, :]
        df_wt = utils.properties_channel(segmented_stack, frame_WT, "WT")

        frame_CRC = frame[CRC_channel, :, :, :]
        df_crc = utils.properties_channel(segmented_stack, frame_CRC, "CRC")

        df_final = props.merge(df_wt, on="label").merge(df_crc, on="label")

        (
            df_final,
            count_phenotype_c1,
            count_phenotype_c2,
            count_phenotype_c1and2,
            count_phenotype_empty,
        ) = utils.phenotype(df_final)

        os.makedirs(os.path.join(input_directory, "properties"), exist_ok=True)
        output_txt_file = os.path.join(
            input_directory, "properties", f"{file.split('.')[0]}_props.txt"
        )
        df_final.to_csv(output_txt_file, sep="\t", index=False)

        # Append data from this frame to the summary
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
    summary_txt = os.path.join(input_directory, "summary_results.txt")
    summary_results.to_csv(summary_txt, sep="\t", index=False)

    # Pad all frames
    segmented_movie_padded = [
        utils.pad_to_shape(stack, max_dims) for stack in segmented_movie
    ]

    # Stack across time
    segmented_movie_array = np.stack(segmented_movie_padded, axis=0)

    # Save it
    segmented_movie_tiff = os.path.join(input_directory, "segmented_movie.tif")
    tifffile.imwrite(segmented_movie_tiff, segmented_movie_array)
