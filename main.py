import os

from analyse_organoid import analyse_organoid
from utils.load_model import load_model
from dialog import dialog

cell_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\TEMP\Train model cell segmentation\models\cell_segmentation_organoid2",
)
    
organoid_model = load_model(
    r"Z:\users\6331823\Mario Pipeline\whole_organoid_segmentation",
    )

organoids = dialog().dirs
for organoid in organoids:
    print(f"Processing organoid: {os.path.basename(organoid)}")
    analyse_organoid(organoid, cell_model, organoid_model, True)
    print(f"Finished processing organoid: {organoid}")