# Function to find the biggest existing image file in a input folder

import os

from utils.load_image import is_supported


def find_input_file(input_directory, types=None, exclude=()):
    # Every format load_image can read, unless the caller restricts it further
    input_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if (any(f.lower().endswith(e) for e in types) if types else is_supported(f))
        and not f.endswith(exclude)
    ]
    # If no readable image is found, print a warning
    if not input_files:
        print(
            f"Warning: No readable image found in {input_directory}. "
            f"Skipping this folder."
        )
        return

    # From the found IMS and Tiff files, select the bigest file to use as the input file, as this is probably the main file you want to analyse
    input_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    input_file = input_files[0]

    return input_file
