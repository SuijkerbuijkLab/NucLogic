import os

# The suffixes crop_sample writes, and how segmentation finds a sample's cropped file.
CROPPED_SUFFIXES = ("_cropped.ims", "_cropped.tif")


def find_cropped_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(CROPPED_SUFFIXES)
    )


def needs_crop(directory, crop_mode):
    """crop_mode: "crop" (reuse existing cropped files), "recrop" or "none"."""
    if crop_mode == "recrop":
        return True
    return crop_mode == "crop" and not find_cropped_files(directory)


def remove_stale_cropped_files(directory, save_as):
    """After cropping, delete cropped files in the other format.

    Re-cropping as .tif next to an old _cropped.ims would otherwise leave two, and
    segmentation would pick whichever the directory listing returns first.
    """
    keep = "_cropped.ims" if save_as == ".ims" else "_cropped.tif"
    for path in find_cropped_files(directory):
        if not path.endswith(keep):
            os.remove(path)
