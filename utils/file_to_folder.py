# Function that will move all files into a folder with the name of that file


def file_to_folder(input_directory, position_counts=None):
    """Move each loose image file into a folder named after it.

    position_counts optionally maps file name to its number of positions, as
    scan_directory already worked them out; passing it avoids opening every
    file a second time. Names it does not list are taken to hold one position,
    which holds because that scan looks at every supported file in the folder.

    Returns a list of (file name, error) for files that could not be moved.
    A file open elsewhere -- most often still loaded in the napari viewer --
    cannot be renamed on Windows, and that must not stop the other samples
    from being created.
    """
    import os
    import shutil

    from utils.load_image import count_positions, is_supported
    from utils.split_positions import stem_of

    failures = []

    for file in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file)

        if os.path.isfile(file_path) and is_supported(file):
            # Multiposition files stay put: moving one into its own folder would
            # make it that sample's input file, and the pipeline would analyse
            # only its first position. utils/split_positions.py handles them.
            if position_counts is not None:
                positions = position_counts.get(file, 1)
            else:
                positions = count_positions(file_path)
            if positions > 1:
                continue

            # Remove the file extension to create the folder name. stem_of
            # knows about compound extensions, so "s.ome.tif" gives "s", not
            # "s.ome".
            folder_name = stem_of(file)
            new_folder_path = os.path.join(input_directory, folder_name)

            # Create the new folder if it doesn't exist
            os.makedirs(new_folder_path, exist_ok=True)

            # Move the file into the new folder
            destination = os.path.join(new_folder_path, file)
            try:
                shutil.move(file_path, destination)
            except OSError as error:
                failures.append((file, error))
                # shutil.move falls back to copy-then-delete when the rename
                # fails, so a locked file can leave a full copy behind with the
                # original still in place. Undo it, then drop the empty folder
                # so it is not listed as a sample.
                if os.path.exists(file_path) and os.path.exists(destination):
                    try:
                        os.remove(destination)
                    except OSError:
                        pass
                try:
                    os.rmdir(new_folder_path)
                except OSError:
                    pass

    return failures


if __name__ == "__main__":
    input_directory = r"C:\Users\6331823\Local SSD\Data_Miriam\20240312"

    file_to_folder(input_directory)
