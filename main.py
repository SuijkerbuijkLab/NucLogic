"""
The part of the program you need to run
"""

import tifffile
from scipy.ndimage import zoom
from cellpose.utils import stitch3D
import numpy as np
import pandas as pd

import os
import re

import utils

# Add directory where all organoid data is stored
input_directory = r"C:\Users\6331823\Downloads\crop_test"
rescale_factor = (1, 0.5, 0.5)  # Rescale factor for the images for cellpose processing

output_directory = os.path.join(str(input_directory), "cropped")

XY_path = [
    os.path.join(input_directory, f)
    for f in os.listdir(input_directory)
    if "cropped" in f and f.endswith(".tif")
]
utils.crop_tiff_stack(input_directory, output_directory, XY_path)

files = sorted(
    [
        f
        for f in os.listdir(output_directory)
        if f.startswith("Channel-ref") and f.endswith(".tif")
    ]
)

# model = utils.load_model(
#     custom_model=True,
#     model_path=r"Z:\users\6331823\Mario Pipeline\codes\models\organoids_3D.pkl",
# )
model = utils.load_model()

summary_results = []

for file in files:
    # Load each frame
    frame_path = os.path.join(output_directory, file)
    print(f"Processing {file}...")
    frame = tifffile.imread(frame_path)
    frame = utils.single_channel(frame)  # Ensure single channel data

    original_shape = frame.shape  # Save original shape for later rescaling
    frame = zoom(frame, zoom=rescale_factor, order=1)  # Rescale imagefor speed

    frame = utils.threshold(frame)  # Apply mean thresholding

    segmented_stack = utils.segment(frame, model)  # Segment per Z using the model
    segmented_stack = stitch3D(segmented_stack)  # Connect masks across Z

    # Rescale the segmented stack to the original shape
    zoom_factors = np.array(original_shape) / np.array(segmented_stack.shape)
    segmented_stack = zoom(segmented_stack, zoom=zoom_factors, order=0).astype(
        np.uint16
    )

    # Save segmented tiffile
    os.makedirs(os.path.join(input_directory, "segmented"), exist_ok=True)
    segmented_tiff_file = os.path.join(
        input_directory, "segmented", f"{file.split('.')[0]}_masks.tif"
    )
    tifffile.imwrite(segmented_tiff_file, segmented_stack)

    # Get properties of the masked nuclei
    props = utils.properties_mask(segmented_stack)

    # Get properties of the WT and CRC channels at the masked nuclei locations
    frameNumber = re.search(r"frame-(\d+)", file).group(1)
    wt_ch = os.path.join(
        input_directory, "cropped", f"Channel-WT-frame-{frameNumber}.tif"
    )
    df_wt = utils.properties_channel(segmented_stack, wt_ch, "WT")

    crc_ch = os.path.join(
        input_directory, "cropped", f"Channel-CRC-frame-{frameNumber}.tif"
    )
    df_crc = utils.properties_channel(segmented_stack, crc_ch, "CRC")

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
    print(f"Quantified phenotype data saved to: {output_txt_file}")

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
summary_csv = os.path.join(input_directory, "summary_results.csv")
summary_results.to_csv(summary_csv, sep="\t", index=False)
