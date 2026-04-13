import os
import tifffile
from imaris_ims_file_reader import ims
from utils.find_input_file import find_input_file
from utils.get_time_interval import get_time_interval
from utils.max_project import max_project
from utils.crop_fixed import crop_fixed
from utils.crop import crop
from utils.save_as_ims import save_as_ims
import numpy as np
from utils.tiff_metadata import load_tiff_movie_and_metadata


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

    # Get metadata of the time interval of the movie
    metadata_missing = False
    if input_file.endswith(".ims"):
        time_interval = get_time_interval(input_file)
        loaded_movie = ims(input_file)
        voxel_size = loaded_movie.resolution
        if voxel_size is None:
            metadata_missing = True
            voxel_size = (1.0, 1.0, 1.0)
        while loaded_movie.ndim < 5:
            loaded_movie = np.expand_dims(loaded_movie, axis=0)
    else:
        loaded_movie, voxel_size, time_interval, metadata_missing = (
            load_tiff_movie_and_metadata(input_file)
        )

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
        fixed=is_fixed,
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
        )
    elif save_as == ".tif":
        # Save this final cropped frame in the output folder as tif
        tifffile.imwrite(
            os.path.join(os.path.dirname(input_file), f"{name}_cropped.tif"),
            cropped_movie,
            bigtiff=True,
            imagej=True,
            resolution=(
                1 / voxel_size[2],
                1 / voxel_size[1],
            ),
            metadata={
                "unit": "um",
                "axes": "TZCYX",
                "spacing": voxel_size[0],
                "finterval": time_interval,
                "tunit": "h",
            },
            compression="zlib",
            compressionargs={"level": 8},
        )

    return True
