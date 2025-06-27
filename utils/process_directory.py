from pathlib import Path


# Function to validate and process the directory
def process_directory(input_dir):
    pathI = Path(input_dir)

    if not pathI.exists() or not pathI.is_dir():
        raise FileNotFoundError(
            f"The specified path does not exist or is not a directory: {pathI}"
        )

    files = [f.name for f in pathI.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in the directory: {pathI}")

    files.insert(0, "None")
    return pathI, files
