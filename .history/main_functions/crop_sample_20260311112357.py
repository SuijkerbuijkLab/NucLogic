import os
import tifffile
from imaris_ims_file_reader import ims
from utils import find_input_file, get_time_interval, max_project, crop_fixed, crop
from help_functions import save_as_ims
import numpy as np


def crop_sample(input_directory, channel_types, organoid_model, save_as=".ims"):

    nuclei_channels = []
    for i, channel_type in enumerate(channel_types):
        if "nuclei" in channel_type.lower():
            nuclei_channels.append(i)
    dual_nuclei = len(nuclei_channels) == 2

    # Find the biggest existing IMS or Tiff file in this folder and use it as the input
    input_file = find_input_file(input_directory=input_directory)

    # Get what the file name is without the extension to use for new file generation
    name = os.path.basename(input_file).split(".")[0]

    # Get metadata of the time interval of the movie
    if input_file.endswith(".ims"):
        time_interval = get_time_interval(input_file)
        loaded_movie = ims(input_file)
        voxel_size = loaded_movie.resolution
    else:
        with tifffile.TiffFile(input_file) as tif:
            if tif.is_ome:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(tif.ome_metadata)
                ns = root.tag.split("}")[0].lstrip("{")
                pixels = root.find(f".//{{{ns}}}Pixels")
                voxel_size = (
                    float(pixels.get("PhysicalSizeZ", 1.0)),
                    float(pixels.get("PhysicalSizeY", 1.0)),
                    float(pixels.get("PhysicalSizeX", 1.0)),
                )
                time_interval = float(pixels.get("TimeIncrement", 1.0))
            else:
                voxel_size = (1.0, 1.0, 1.0)
                time_interval = 1.0
        loaded_movie = tif.asarray()

    is_fixed = loaded_movie.ndim < 5

    # If there are no existing cropped tiffs, we will create a max XY projection used to crop the organoid
    # The max XY projection is then used in the crop function to crop every frame of the movie in both XY and XZ to generate way smaller files for segmentation
    proj_XY = max_project(
        input_file,
        name=name,
        fixed=is_fixed,
    )

    if is_fixed:
        # given as Z C Y X
        cropped_movie = crop_fixed(
            proj_XY=proj_XY,
            input_file=input_file,
            nuclei=nuclei_channels,
            name=name,
            dual_nuclei=dual_nuclei,
        )
        cropped_movie = np.expand_dims(
            cropped_movie, axis=0
        )  # add time dimension as the first dimension

    else:
        cropped_movie = crop(
            proj_XY=proj_XY,
            input_file=input_file,
            model=organoid_model,
            nuclei=nuclei_channels,
            name=name,
            dual_nuclei=dual_nuclei,
        )

    if save_as == ".ims":
        # Input TCZYC!
        cropped_movie = cropped_movie.transpose(0, 2, 1, 3, 4)  # TCZYC
        save_as_ims(
            input_movie=cropped_movie,
            output_filename=os.path.join(
                os.path.dirname(input_file), f"{name}_cropped.ims"
            ),
            voxel_size=voxel_size,
            time_interval=time_interval,
        )
    elif save_as == ".tif":
        # Save this final cropped frame in the output folder as tif
        tifffile.imwrite(
            os.path.join(os.path.dirname(input_file), f"{name}_cropped.tif"),
            cropped_movie,
            bigtiff=True,
            resolution=(
                (1 / voxel_size[1]) * 25400,
                (1 / voxel_size[2]) * 25400,
            ),
            metadata={
                "unit": "um",
                "axes": "TZCYX",
                "PhysicalSizeX": voxel_size[2],
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": voxel_size[1],
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": voxel_size[0],
                "PhysicalSizeZUnit": "µm",
                "spacing": voxel_size[0],
                "TimeIncrement": time_interval,
                "TimeIncrementUnit": "h",
            },
            compression="zlib",
            compressionargs={"level": 8},
        )
