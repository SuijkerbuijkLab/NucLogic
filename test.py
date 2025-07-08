import utils

import os

input_directory = r"C:\Users\6331823\Local SSD\TL02"

ims_file = [
    os.path.join(input_directory, f)
    for f in os.listdir(input_directory)
    if f.endswith(".ims")
][0]

movie = os.path.join(input_directory, ims_file)

utils.max_project(movie, input_directory)

