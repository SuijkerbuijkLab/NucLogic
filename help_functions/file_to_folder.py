# Function that will move all files into a folder with the name of that file

import os
import shutil

input_directory = r"C:\Users\6331823\Local SSD\Data_Anna\crc"

for file in os.listdir(input_directory):
    file_path = os.path.join(input_directory, file)

    if os.path.isfile(file_path):
        # Remove the file extension to create the folder name
        folder_name = os.path.splitext(file)[0]
        new_folder_path = os.path.join(input_directory, folder_name)

        # Create the new folder if it doesn't exist
        os.makedirs(new_folder_path, exist_ok=True)

        # Move the file into the new folder
        shutil.move(file_path, os.path.join(new_folder_path, file))
