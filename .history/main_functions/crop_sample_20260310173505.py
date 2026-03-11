import os
import tifffile
from imaris_ims_file_reader import ims
from utils import find_input_file, get_time_interval, max_project, crop_fixed, crop

def crop_sample(input_directory, channel_types, organoid_model):

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
        time_interval = 1
        voxel_size = (1.0, 1.0, 1.0)  # Assume isotropic if not provided
        loaded_movie = tifffile.imread(input_file)
    
    is_fixed = loaded_movie.ndim < 5
    
    # If there are no existing cropped tiffs, we will create a max XY projection used to crop the organoid
    # The max XY projection is then used in the crop function to crop every frame of the movie in both XY and XZ to generate way smaller files for segmentation
    proj_XY = max_project(
        input_file,
        name=name,
        fixed=is_fixed,
    )

    if is_fixed:
        crop_fixed(
            proj_XY=proj_XY,
            input_file=input_file,
            nuclei=nuclei_channels,
            name=name,
            voxel_size=voxel_size,
            dual_nuclei=dual_nuclei,
        )

    else:
        crop(
            proj_XY=proj_XY,
            input_file=input_file,
            model=organoid_model,
            nuclei=nuclei_channels,
            name=name,
            voxel_size=voxel_size,
            dual_nuclei=dual_nuclei,
            time_interval=time_interval,
        )
