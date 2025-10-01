import json
import os

import napari

with open(r"miscellaneous\napari_paths.json", "r") as f:
    paths = json.load(f)

viewer = napari.Viewer()

for movie in paths:
    viewer.open(movie)

napari.run()
