import napari
import json
import os

with open("napari_paths.json", "r") as f:
    paths = json.load(f)

viewer = napari.Viewer()

for folder in paths:
    tiffs = [f for f in os.listdir(folder) if f.endswith(".tif")]
    for tif in tiffs:
        viewer.open(os.path.join(folder, tif))

napari.run()
