import tifffile
import numpy as np
from alive_progress import alive_bar

import os
import re

from .threshold import threshold


# Function to crop TIFF stack based on max projections made in FIJI
def crop_tiff_stack(input_dir, output_dir, projXY_path, crop=True):
    try:
        # Load projections
        projXY = tifffile.imread(projXY_path)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Regex to find frame info
        pattern = re.compile(r"frame-(\d+)\.tif")

        # Find available frames
        frame_numbers = set()
        for file in os.listdir(input_dir):
            match = pattern.search(file)
            if match:
                frame_numbers.add(int(match.group(1)))

        if not frame_numbers:
            raise ValueError("No frames found in the input directory.")

        frames = sorted(frame_numbers)

        with alive_bar(len(frames), title="Cropping frames") as bar:
            for t in frames:
                ref_path = os.path.join(input_dir, f"Channel-ref-frame-{t}.tif")
                crc_path = os.path.join(input_dir, f"Channel-CRC-frame-{t}.tif")
                wt_path = os.path.join(input_dir, f"Channel-WT-frame-{t}.tif")

                if os.path.exists(ref_path):
                    # print(f"Processing frame {t}")

                    if crop:
                        frame = tifffile.imread(ref_path)
                        # print(f"Original shape: {frame.shape}")

                        # Determine the number of channels
                        if frame.ndim == 4:
                            num_channels = frame.shape[1]
                        else:
                            num_channels = 1
                            frame = frame[:, np.newaxis, :, :]

                        # Initialize cropping limits
                        slice_min, slice_max = 0, frame.shape[0]
                        row_min, row_max = 0, frame.shape[2]
                        col_min, col_max = 0, frame.shape[3]

                        # Create masks based on projections
                        maskXY = projXY[t] > 0

                        # Find coordinates of non-zero pixels
                        coordsXY = np.where(maskXY)

                        # Adjust cropping limits based on XY projection
                        if coordsXY[0].size > 0:
                            row_min, row_max = (
                                np.min(coordsXY[0]),
                                np.max(coordsXY[0]) + 1,
                            )
                            col_min, col_max = (
                                np.min(coordsXY[1]),
                                np.max(coordsXY[1]) + 1,
                            )

                        ref = frame.copy()[
                            slice_min:slice_max, :, row_min:row_max, col_min:col_max
                        ]

                        ref = threshold(
                            ref
                        )  # Apply thresholding to the reference channel
                        tt_ref = threshold(ref)
                        thrs_intensity = np.mean(tt_ref)

                        z_list = []
                        for z in range(slice_max):
                            plane_check = tt_ref[z, :, :]
                            if np.mean(plane_check) > thrs_intensity:
                                z_list.append(z)

                        slice_min, slice_max = min(z_list), max(z_list) + 1

                        # Process each channel
                        for channel_path, channel_name in zip(
                            [ref_path, crc_path, wt_path], ["ref", "CRC", "WT"]
                        ):
                            if os.path.exists(channel_path):
                                frame = tifffile.imread(channel_path)
                                if frame.ndim == 3:
                                    frame = frame[:, np.newaxis, :, :]

                                cropped = frame[
                                    slice_min:slice_max,
                                    :,
                                    row_min:row_max,
                                    col_min:col_max,
                                ]
                                # print(f"Cropped shape ({channel_name}): {cropped.shape}")

                                # Ensure the correct shape: (T, Z, C, Y, X, S)
                                cropped = np.expand_dims(
                                    cropped, axis=0
                                )  # Add T dimension
                                cropped = np.expand_dims(
                                    cropped, axis=-1
                                )  # Add S dimension

                                output_path = os.path.join(
                                    output_dir, f"Channel-{channel_name}-frame-{t}.tif"
                                )
                                tifffile.imwrite(
                                    output_path,
                                    cropped,
                                    imagej=True,
                                    metadata={"axes": "TZCYXS"},
                                )
                            else:
                                print(f"Warning: {channel_path} not found.")
                else:
                    print(f"Frame {t} not found: {ref_path}")
                bar()

        print(f"Frames cropped and saved to {output_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
