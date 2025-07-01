# Script to split ims file into list of TIFF files

from imaris_ims_file_reader.ims import ims
import tifffile
from alive_progress import alive_bar

import os


def split_files(file, output_directory, nuclei=2, WT=1, CRC=0):
    os.makedirs(output_directory, exist_ok=True)

    movie = ims(file)  # T,C,Z,Y,X

    with alive_bar(movie.TimePoints, title="Making TIFFs from frames") as bar:
        for frame in range(movie.TimePoints):
            for channel in range(movie.Channels):
                image = movie[frame, channel, :, :, :]
                if channel == CRC:
                    name = "CRC"
                elif channel == WT:
                    name = "WT"
                elif channel == nuclei:
                    name = "Ref"
                else:
                    name = "Channel4"
                tif = os.path.join(
                    output_directory, f"Channel-{name}-frame-{frame}.tif"
                )
                tifffile.imwrite(tif, image)

            bar()
            