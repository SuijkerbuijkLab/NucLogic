import os
from utils.find_input_file import find_input_file
from utils.max_project import max_project
from utils.crop_fixed import crop_fixed
from utils.crop import crop
from utils.load_image import ims_acquisition_start, load_image
from utils.save_as_ims import save_as_ims
from utils.save_as_tiff import save_as_tiff
import numpy as np


def crop_sample(
    input_directory,
    channel_types,
    organoid_model,
    save_as=".ims",
    manual_fixed=False,
    fixed_only=False,
    user_voxel_size=(1.0, 1.0, 1.0),
):
    def _to_voxel_tuple(voxel_like):
        try:
            return (
                float(voxel_like[0]),
                float(voxel_like[1]),
                float(voxel_like[2]),
            )
        except (TypeError, ValueError, IndexError):
            return (1.0, 1.0, 1.0)

    def _should_use_user_voxel(measured_voxel, metadata_missing):
        measured = _to_voxel_tuple(measured_voxel)
        user_voxel = _to_voxel_tuple(user_voxel_size)
        user_is_default = np.allclose(user_voxel, (1.0, 1.0, 1.0))
        measured_is_default = np.allclose(measured, (1.0, 1.0, 1.0))
        if not user_is_default and (metadata_missing or measured_is_default):
            print(f"Using user-provided voxel size override: {user_voxel}")
            return user_voxel
        return measured

    nuclei_channels = []
    for i, channel_type in enumerate(channel_types):
        if "nuclei" in channel_type.lower():
            nuclei_channels.append(i)

    # Find the biggest existing IMS or Tiff file in this folder and use it as the input
    input_file = find_input_file(input_directory=input_directory)

    # Get what the file name is without the extension to use for new file generation
    name = os.path.basename(input_file).split(".")[0]

    # Lazy: only the shape is needed here, the crop functions re-open the file
    # themselves to read pixels.
    loaded_movie, voxel_size, time_interval, metadata_missing = load_image(input_file)

    voxel_size = _should_use_user_voxel(voxel_size, metadata_missing)

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
    elif save_as == ".tif":
        # Save this final cropped frame in the output folder as tif
        save_as_tiff(
            os.path.join(os.path.dirname(input_file), f"{name}_cropped.tif"),
            cropped_movie,
            "TZCYX",
            voxel_size,
            time_interval,
        )

    return True
