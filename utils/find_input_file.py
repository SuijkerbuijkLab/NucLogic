# Function to find the biggest existing tiff or ims file in a input folder

import os


def find_input_file(input_directory):
    # Find the existing IMS or Tiff files in this folder
    input_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".ims") or f.endswith(".tif")
    ]
    # If no IMS or Tiff file is found, print a warning
    if not input_files:
        print(f"Warning: No IMS file found in {input_directory}. Skipping this folder.")
        return

    # From the found IMS and Tiff files, select the bigest file to use as the input file, as this is probably the main file you want to analyse
    input_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    input_file = input_files[0]

    return input_file
