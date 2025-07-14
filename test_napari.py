import napari
import tifffile

# Load the 3D TIFF file
mask = tifffile.imread(r"E:\Users\Sebastian_van_Dijk\TEMP\TL01\segmented\Channel-ref-frame-59_masks.tif")

# Start napari viewer and show the mask
viewer = napari.Viewer()
viewer.add_labels(mask, name="Organoid Mask", scale=(4.96/0.621, 1, 1))

napari.run()