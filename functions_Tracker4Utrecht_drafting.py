import subprocess
import sys

# List of required packages
required_packages = [
    "numpy", "tifffile", "pandas", "scipy", "ipywidgets","IPython.display"
]

# Install any missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import os
import re
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import tifffile  # To read TIFF metadata

# Function to validate and process the directory
def process_directory():
    pathI = input("Please enter the directory path: ").strip()
    pathI = Path(pathI)

    if not pathI.exists() or not pathI.is_dir():
        raise FileNotFoundError(f"The specified path does not exist or is not a directory: {pathI}")

    files = [f.name for f in pathI.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in the directory: {pathI}")

    files.insert(0, "None")
    return pathI, files

# Function to handle metadata and file checks
def process_metadata(pathI):
    metadata_path = os.path.join(pathI, "archivo_metadata.txt")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"The metadata file does not exist: {metadata_path}")
    return metadata_path


def create_widgets(files, pathI):
    user_data = {}  # Dictionary to store user data

    projXY_selector = widgets.Dropdown(
        options=files,
        description='projXY:',
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    projXZ_selector = widgets.Dropdown(
        options=files,
        description='projXZ:',
        layout=widgets.Layout(width='50%'),
        style={'description_width': 'initial'}
    )

    crop_check = widgets.Checkbox(
        value=False,
        description='Apply crop?',
        style={'description_width': 'initial'}
    )

    output_name = widgets.Text(
        value='output',
        placeholder='Enter output file name',
        description='Output Name:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    submit_button = widgets.Button(description="Submit")
    output = widgets.Output()

    # Callback function
    def select_files_callback(b):
        projXY_file = projXY_selector.value
        projXZ_file = projXZ_selector.value
        projXY_path, projXZ_path = None, None

        with output:
            output.clear_output()
            if projXY_file != "None":
                projXY_path = pathI / projXY_file
                if os.path.exists(projXY_path):
                    print(f"Selected projXY file: {projXY_path}")
                    # Get dimensions without opening the image
                    with tifffile.TiffFile(projXY_path) as tif:
                        projXY_dims = tif.pages[0].shape  # (height, width)
                    print(f"projXY dimensions: {projXY_dims}")
                else:
                    print(f"Warning: {projXY_path} does not exist.")

            if projXZ_file != "None":
                projXZ_path = pathI / projXZ_file
                if os.path.exists(projXZ_path):
                    print(f"Selected projXZ file: {projXZ_path}")
                    # Get dimensions without opening the image
                    with tifffile.TiffFile(projXZ_path) as tif:
                        projXZ_dims = tif.pages[0].shape  # (height, width)
                    print(f"projXZ dimensions: {projXZ_dims}")
                else:
                    print(f"Warning: {projXZ_path} does not exist.")

            user_data.update({
                'projXY_path': projXY_path,
                'projXZ_path': projXZ_path,
                'crop': crop_check.value,
                'output_name': output_name.value,
            })

            print("User data updated:", user_data)

    # Connect the callback to the button
    submit_button.on_click(select_files_callback)

    # Show the widgets
    display(projXY_selector, projXZ_selector, crop_check, output_name, submit_button, output)

    # Return the collected data once the button is clicked
    return user_data  # Return the collected data
