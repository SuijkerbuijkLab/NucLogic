import os
import traceback
from datetime import datetime

import numpy as np
import tables
import tifffile
from alive_progress import alive_bar
from matplotlib.colors import to_rgba
from PyImarisWriter import PyImarisWriter as PW
from imaris_ims_file_reader.ims import ims

import utils


def crop_organoid(
    input_directory,
    organoid_model,
    channel_names,
    channel_colors,
    crop_existing=False,
):
    # Setting stuff for the output
    output_directory_cropped = os.path.join(str(input_directory), "cropped")
    nuclei_channel = next(
        (i for i, ch in enumerate(channel_names) if ch.lower() == "nuclei"),
    )

    ims_movie = utils.find_input_file(input_directory, types=[".ims"])
    loaded_movie = ims(ims_movie)
    voxel_size = loaded_movie.resolution  # (Z, Y, X)
    timepoints = loaded_movie.TimePoints
    with tables.open_file(ims_movie, "r") as hf:
        time_values = hf.root.DataSetTimes.Time.read()
    timestamps = np.array([row[2] for row in time_values])
    timestamps = timestamps // 1000
    name = os.path.basename(ims_movie)
    name = name.split(".")[0]

    output_filename = os.path.join(
        input_directory,
        f"{name}_cropped.ims",
    )

    if not crop_existing:
        proj_XY = utils.max_project(ims_movie, input_directory, nuclei=nuclei_channel)
        proj_XY_name = os.path.join(os.path.dirname(ims_movie), f"{name}_projXY.tif")
        tifffile.imwrite(
            proj_XY_name,
            proj_XY,
            compression="zlib",
            compressionargs={"level": 8},
        )

        try:
            utils.crop(
                proj_XY=proj_XY,
                input_file=ims_movie,
                output_directory=output_directory_cropped,
                model=organoid_model,
                nuclei=nuclei_channel,
                name=name,
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(
                f"❌ ERROR during cropping of organoid. Possibly too few nuclei marker-positive cells.\n\n{e}\n\nTraceback:\n{tb}"
            )

    max_dims = [0, 0, 0, 0]
    movie = []
    for frame in range(timepoints):
        image = tifffile.imread(
            os.path.join(output_directory_cropped, f"Frame-{frame}.tif")
        )

        movie.append(image)
        for i in range(4):
            max_dims[i] = max(max_dims[i], image.shape[i])

    # Pad all frames
    movie_padded = [utils.pad_to_shape(frame, max_dims) for frame in movie]

    # Stack across time
    movie = np.stack(movie_padded, axis=0)

    # Reorder movie from (T, C, Z, Y, X) to (Z, Y, X, C, T)
    movie_reordered = np.transpose(movie, (2, 3, 4, 1, 0))
    movie_reordered = np.ascontiguousarray(movie_reordered)
    Z, Y, X, C, T = movie_reordered.shape

    # Create image size
    dimension_sequence = PW.DimensionSequence("x", "y", "z", "c", "t")
    image_size = PW.ImageSize(x=X, y=Y, z=Z, c=C, t=T)
    block_size = PW.ImageSize(x=X, y=Y, z=1, c=1, t=1)
    sample_size = PW.ImageSize(x=1, y=1, z=1, c=1, t=1)
    image_extents = PW.ImageExtents(
        0.0, 0.0, 0.0, voxel_size[2] * X, voxel_size[1] * Y, voxel_size[0] * Z
    )

    # Compression
    options = PW.Options()
    options.mCompressionAlgorithmType = PW.eCompressionAlgorithmLZ4
    options.mEnableLogProgress = True

    # Dummy progress callback
    callback_class = utils.dummy_callback()

    # Create converter
    converter = PW.ImageConverter(
        "uint16",
        image_size,
        sample_size,
        dimension_sequence,
        block_size,
        output_filename,
        options,
        "OrganoidSegmenter",
        "v1.0",
        callback_class,
    )

    num_blocks = image_size / block_size
    block_index = PW.ImageSize()
    total_blocks = Z * C * T

    with alive_bar(total_blocks, title="Writing cropped IMS file") as bar:
        for c in range(num_blocks.c):
            block_index.c = c
            for t in range(num_blocks.t):
                block_index.t = t
                for z in range(num_blocks.z):
                    block_index.z = z
                    z_start = z * block_size.z
                    z_end = min(z_start + block_size.z, Z)
                    for y in range(num_blocks.y):
                        block_index.y = y
                        y_start = y * block_size.y
                        y_end = min(y_start + block_size.y, Y)
                        for x in range(num_blocks.x):
                            block_index.x = x
                            x_start = x * block_size.x
                            x_end = min(x_start + block_size.x, X)

                            # Extract real voxel block
                            block = movie_reordered[
                                z_start:z_end, y_start:y_end, x_start:x_end, c, t
                            ]

                            # Reshape to (Z, Y, X, 1, 1) for compatibility
                            block = block[:, :, :, np.newaxis, np.newaxis].astype(
                                np.uint16
                            )

                            if converter.NeedCopyBlock(block_index):
                                converter.CopyBlock(block, block_index)

                            bar()

    parameters = PW.Parameters()
    for i in range(C):
        parameters.set_channel_name(
            i, channel_names[i] if i < len(channel_names) else f"Channel {i}"
        )
    # Time info
    time_infos = [datetime.utcfromtimestamp(t / 1e6) for t in timestamps]

    # Convert to PW.Color using to_rgba
    colors = []
    for name in channel_colors:
        try:
            r, g, b, a = to_rgba(name)
            colors.append(PW.Color(r, g, b, a))
        except ValueError:
            print(f"Color '{name}' is not recognized. Using default white.")
            colors.append(PW.Color(1, 1, 1, 1))

    # --- Apply colors to ColorInfo objects ---
    color_infos = []
    for i in range(C):  # assuming C is the number of channels
        ci = PW.ColorInfo()
        if i < len(channel_colors):
            ci.set_base_color(colors[i])
        else:
            ci.set_base_color(PW.Color(1, 1, 1, 1))  # default white for extras
        color_infos.append(ci)

    # Finalize writing
    converter.Finish(
        image_extents,  # Required physical bounds
        parameters,  # Parameters
        time_infos,  # TimeInfos
        color_infos,  # ColorInfos
        False,  # adjust_color_range
    )
    try:
        converter.Destroy()
    except OSError as e:
        print("Warning: cleanup failed — continuing anyway")

    print(f"✅ Wrote IMS file: {output_filename}")
