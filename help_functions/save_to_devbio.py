import os
import tifffile
import shutil
from PyImarisWriter import PyImarisWriter as PW
import tables
from alive_progress import alive_bar
from matplotlib.colors import to_rgba
from datetime import datetime
import imaris_ims_file_reader.ims as ims
import numpy as np
import czifile


class dummy_callback:
    def RecordProgress(self, percent_complete: float, bytes_remaining: int):
        pass  # No-op


def write_ims(
    file,
    output_filename,
    voxel_size=(2.48, 0.642, 0.642),
    timestamps=None,
    channel_names=["DAPI", "AldoB", "mTmG", "Lyz"],
    channel_colors=["blue", "green", "magenta", "gray"],
):
    # if isinstance(file, str):
    #     movie = tifffile.imread(file)  # Z, C, Y, X
    # else:
    #     movie = file  # Z, C, Y, X
    movie = file  # Z, C, Y, X
    movie = np.expand_dims(movie, axis=0)  # add time axis
    # new_movie = np.transpose(movie, (0, 2, 1, 3, 4))  # T, C, Z, Y, X
    # new_movie = np.ascontiguousarray(new_movie)
    new_movie = movie  # T, C, Z, Y, X

    T, C, Z, Y, X = new_movie.shape

    if not voxel_size:
        voxel_size = movie.resolution  # (Z, Y, X)

    if timestamps is None:
        time_infos = [datetime.now()] * T
    elif timestamps == "ims":
        with tables.open_file(file, "r") as hf:
            time_values = hf.root.DataSetTimes.Time.read()
        timestamps = np.array([row[2] for row in time_values])
        timestamps = timestamps // 1000
        time_infos = [datetime.utcfromtimestamp(t / 1e6) for t in timestamps]
    else:  # use the number given x to make an array that is x spaced time points
        base_time = datetime.now()
        time_infos = [
            datetime.fromtimestamp(base_time.timestamp() + i * timestamps * 3600)
            for i in range(T)
        ]

    if not channel_names:
        channel_names = ["magenta", "green", "blue"]
    if not channel_colors:
        channel_colors = ["magenta", "green", "blue"]

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
    # options.mNumberOfThreads = 6
    options.mCompressionAlgorithmType = PW.eCompressionAlgorithmShuffleGzipLevel4
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
                            block = new_movie[
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

    # input_directory = r"C:\Users\6331823\Local SSD\Data_Merel\EXP087"
    # output_directory = r"Z:\shared\Merel en Sebas analyse\Exp087"

    # samples = [
    #     file
    #     for file in os.listdir(input_directory)
    #     if os.path.isdir(os.path.join(input_directory, file))
    # ]

    # for sample in samples:
    #     # only if sample name contains a number higher than 79
    #     # if int("".join(filter(str.isdigit, sample))) <= 79:
    #     #     continue
    #     sample_dir = os.path.join(input_directory, sample)
    #     os.makedirs(os.path.join(output_directory, sample), exist_ok=True)

    #     to_copy = [
    #         f"{sample}_projXY.tif",
    #         f"{sample}_projXY_tracked.tif",
    #         "dapi_stitched.tif",
    #         "aldob_stitched.tif",
    #         "lyz_stitched.tif",
    #     ]

    #     for file_name in to_copy:
    #         shutil.copy(
    #             os.path.join(sample_dir, file_name),
    #             os.path.join(output_directory, sample, file_name),
    #         )

    #     shutil.copy(
    #         os.path.join(sample_dir, "properties", "Frame-0_props.csv"),
    #         os.path.join(output_directory, sample, f"{sample}_properties.csv"),
    #     )

    #     ims_file = os.path.join(output_directory, sample, f"{sample}_cropped.ims")
    #     cropped_file = os.path.join(sample_dir, "cropped", "Frame-0.tif")

    #     write_ims(
    #         cropped_file,
    #         ims_file,
    #         voxel_size=(2.47, 0.304, 0.304),
    #         channel_names=["mTmG", "DAPI", "AldoB", "Lyz"],
    #         channel_colors=["magenta", "blue", "lime", "white"],
    #     )

    # write_ims(
    #     r"C:\Users\6331823\Local SSD\Data_Anna\Exp137\MIX_s48\MIX_s48.tif",
    #     r"C:\Users\6331823\Local SSD\Data_Anna\Exp137\MIX_s48\MIX_s48_gzip9.ims",
    #     voxel_size=(1, 1, 1),
    #     channel_names=["H2B-Cerulean", "Dendra2", "mTmG"],
    #     channel_colors=["cyan", "green", "magenta"],
    #     timestamps=4,
    # )

    # movie = tifffile.imread(
    #     r"C:\Users\6331823\Local SSD\Data_Anna\Exp137\MIX_s48\MIX_s48.tif"
    # )  # Z, C, Y, X
    # print(movie.shape)

    # input_directory = (
    #     r"Z:\users\5595347\Microscope\sd4\AKG_Exp140_TL_mTmGH2B_ExpansionvsIsolation"
    # )

    # table = {
    #     "s1": "WT_Expansion_1",
    #     "s2": "WT_Expansion_5",
    #     "s3": "WT_Expansion_7",
    #     "s4": "WT_Expansion_9",
    #     "s5": "WT_Expansion_12",
    #     "s6": "WT_Expansion_13",
    #     "s7": "WT_Expansion_15",
    #     "s8": "WT_Expansion_18",
    #     "s9": "WT_Expansion_20",
    #     "s10": "WT_Expansion_21",
    #     "s11": "WT_Expansion_22",
    #     "s14": "WT_Expansion_29",
    #     "s17": "MIX_Expansion_3",
    #     "s18": "MIX_Expansion_4",
    #     "s25": "MIX_Expansion_15",
    #     "s26": "MIX_Expansion_16",
    #     "s27": "MIX_Expansion_17",
    #     "s29": "MIX_Expansion_19",
    #     "s36": "MIX_Expansion_27",
    #     "s39": "MIX_Expansion_32",
    #     "s43": "MIX_Expansion_36",
    #     "s44": "MIX_Expansion_37",
    #     "s45": "MIX_Expansion_39",
    #     "s46": "MIX_Expansion_40",
    #     "s47": "MIX_Expansion_41",
    #     "s48": "MIX_Expansion_42",
    #     "s50": "MIX_Expansion_44",
    #     "s51": "MIX_Expansion_46",
    #     "s54": "CRC_Expansion_3",
    #     "s55": "CRC_Expansion_4",
    #     "s56": "CRC_Expansion_5",
    #     "s57": "CRC_Expansion_6",
    #     "s58": "CRC_Expansion_7",
    #     "s59": "CRC_Expansion_8",
    #     "s60": "CRC_Expansion_10",
    #     "s61": "CRC_Expansion_11",
    #     "s62": "CRC_Expansion_16",
    #     "s63": "CRC_Expansion_17",
    #     "s64": "CRC_Expansion_18",
    #     "s65": "CRC_Expansion_19",
    # }

    # for s_key, expansion_name in table.items():
    #     movie = []
    #     for t in range(1, 14):
    #         nuclei = tifffile.imread(
    #             os.path.join(
    #                 input_directory,
    #                 f"AKG_Exp140_TL_ExpansionvsIsolation2_w1CSU-445_{s_key}_t{t}.TIF",
    #             )
    #         )
    #         # Remove the right most 63 pixels and add 63 black pixels to the left
    #         nuclei = nuclei[:, :, 0 : nuclei.shape[2] - 63]
    #         nuclei = np.pad(
    #             nuclei, ((0, 0), (0, 0), (63, 0)), mode="constant", constant_values=0
    #         )

    #         wt = tifffile.imread(
    #             os.path.join(
    #                 input_directory,
    #                 f"AKG_Exp140_TL_ExpansionvsIsolation2_w3CSU-561orange_{s_key}_t{t}.TIF",
    #             )
    #         )
    #         crc = tifffile.imread(
    #             os.path.join(
    #                 input_directory,
    #                 f"AKG_Exp140_TL_ExpansionvsIsolation2_w2CSU-488_{s_key}_t{t}.TIF",
    #             )
    #         )
    #         image = np.stack((nuclei, wt, crc), axis=1)
    #         movie.append(image)
    #         print(t)
    #     movie = np.stack(movie, axis=0)

    # os.makedirs(
    #     rf"C:\Users\6331823\Local SSD\Data_Anna\Exp140\{expansion_name}", exist_ok=True
    # )


input_directory = r"E:\Airyscan\Exp.ML.034_Staining_MLKL_RIP3_Diff_uTs_slide2_251031\Stitched_251106_slide2"
output_dir = os.path.join(os.path.dirname(input_directory), "stitched_imaris")
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_directory):
    # if not file.endswith(".czi"):
    #     continue
    if not (
        file
        == "251031_Exp_ML_034_MLKL_RIP#_Diff_uTs_2025_10_31__16_23_35-Stitching-04.czi"
    ):
        continue
    input_file = os.path.join(input_directory, file)
    image = czifile.imread(input_file)
    print(image.shape)
    # image = tifffile.imread(input_file)
    # image = np.transpose(image, (1, 0, 2, 3))
    image = np.squeeze(image)
    image = image[0, :, :, :, :]
    print(image.shape)

    write_ims(
        image,
        os.path.join(output_dir, os.path.basename(input_file).replace(".czi", ".ims")),
        voxel_size=(2.5, 0.691, 0.691),
        channel_names=["Dendra2", "mTmG", "DAPI", "RIP3"],
        channel_colors=["lime", "magenta", "blue", "white"],
        timestamps=None,
    )
