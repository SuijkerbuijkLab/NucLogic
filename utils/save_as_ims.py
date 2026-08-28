# Must be imported before PyImarisWriter runs a conversion: once it has, a first
# `import h5py` in the same process fails to load its HDF5 DLL, which breaks
# reading any .ims afterwards.
import h5py  # noqa: F401

from PyImarisWriter import PyImarisWriter as PW
from alive_progress import alive_bar
from matplotlib.colors import to_rgba
from datetime import datetime, timedelta
from utils.dummy_callback import dummy_callback
import numpy as np


def save_as_ims(
    input_movie,
    output_filename,
    voxel_size,
    time_interval,
    channel_colors=["blue", "lime", "magenta", "gray", "yellow"],
    channel_names=["Channel_1", "Channel_2", "Channel_3", "Channel_4", "Channel_5"],
    start_time=None,
):
    """Save a 5D numpy array as an IMS file using PyImarisWriter, Input as TCZYX.

    time_interval is in hours; start_time is the acquisition time of the first
    timepoint, so a cropped copy keeps the timing of the original.
    """

    try:
        interval_hours = float(time_interval)
        if not np.isfinite(interval_hours) or interval_hours <= 0:
            raise ValueError(f"unusable time interval: {time_interval}")
    except (TypeError, ValueError) as e:
        print(f"Non fatal error reading time interval ({e}); assuming 1 hour.")
        interval_hours = 1.0

    if start_time is None:
        start_time = datetime.now()

    T, C, Z, Y, X = input_movie.shape

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
    options.mCompressionAlgorithmType = PW.eCompressionAlgorithmShuffleGzipLevel6
    options.mEnableLogProgress = True

    # Dummy progress callback
    callback_class = dummy_callback()

    # Create converter
    converter = PW.ImageConverter(
        "uint16",
        image_size,
        sample_size,
        dimension_sequence,
        block_size,
        output_filename,
        options,
        "NucLogic",
        "v1.0",
        callback_class,
    )

    num_blocks = image_size / block_size
    block_index = PW.ImageSize()
    total_blocks = Z * C * T

    with alive_bar(total_blocks, title="Writing IMS file") as bar:
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
                            block = input_movie[
                                t,
                                c,
                                z_start:z_end,
                                y_start:y_end,
                                x_start:x_end,
                            ]

                            if converter.NeedCopyBlock(block_index):
                                converter.CopyBlock(block, block_index)

                            bar()

    parameters = PW.Parameters()
    for i in range(C):
        parameters.set_channel_name(
            i, channel_names[i] if i < len(channel_names) else f"Channel {i}"
        )
    # Time info
    time_infos = [start_time + timedelta(hours=i * interval_hours) for i in range(T)]

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
