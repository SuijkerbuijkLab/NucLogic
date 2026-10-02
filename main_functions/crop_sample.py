import os
from utils.cropped_files import CROPPED_SUFFIXES
from utils.find_input_file import find_input_file
from utils.max_project import max_project
from utils.crop_fixed import crop_fixed
from utils.crop import crop
from utils.load_image import ims_acquisition_start, load_image
from utils.save_as_ims import save_as_ims
from utils.save_as_tiff import save_as_tiff
from utils.voxel_size import resolve_voxel_size
import numpy as np


def crop_sample(
    input_directory,
    channel_types,
    organoid_model,
    save_as=".ims",
    manual_fixed=False,
    fixed_only=False,
    user_voxel_size=None,
):

    nuclei_channels = []
    for i, channel_type in enumerate(channel_types):
        if "nuclei" in channel_type.lower():
            nuclei_channels.append(i)

    # Find the biggest existing IMS or Tiff file in this folder and use it as the input.
    # Earlier cropped files are excluded, so re-cropping never crops a cropped file.
    input_file = find_input_file(input_directory=input_directory, exclude=CROPPED_SUFFIXES)

    # Get what the file name is without the extension to use for new file generation
    name = os.path.basename(input_file).split(".")[0]

    # Lazy: only the shape is needed here, the crop functions re-open the file
    # themselves to read pixels.
    loaded_movie, voxel_size, time_interval, metadata_missing = load_image(input_file)

    voxel_size = resolve_voxel_size(voxel_size, user_voxel_size, metadata_missing)

    # Loader normalizes TIFF/IMS to 5D (T,C,Z,Y,X); fixed samples are represented as T=1.
    is_fixed = loaded_movie.shape[0] == 1

    if fixed_only and not is_fixed:
        return False

    # If there are no existing cropped tiffs, we will create a max XY projection used to crop the organoid
    # The max XY projection is then used in the crop function to crop every frame of the movie in both XY and XZ to generate way smaller files for segmentation
    proj_XY = max_project(
        input_file,
        name=name,
    )

    if is_fixed:
        # given as T Z C Y X
        cropped_movie = crop_fixed(
            proj_XY=proj_XY,
            input_file=input_file,
            nuclei=nuclei_channels,
            name=name,
            manual=manual_fixed,
        )

    else:
        cropped_movie = crop(
            proj_XY=proj_XY,
            input_file=input_file,
            model=organoid_model,
            nuclei=nuclei_channels,
            name=name,
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
            start_time=(
                ims_acquisition_start(input_file)
                if input_file.endswith(".ims")
                else None
            ),
        )
    else:
        # Default: OME-TIFF. Anything other than .ims lands here, so an unknown
        # setting writes the portable format rather than silently writing
        # nothing. The name stays "_cropped.tif" because the rest of the
        # pipeline finds cropped files by that suffix.
        save_as_tiff(
            os.path.join(os.path.dirname(input_file), f"{name}_cropped.tif"),
            cropped_movie,
            "TZCYX",
            voxel_size,
            time_interval,
        )

    return True
