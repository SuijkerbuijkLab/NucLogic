import os

from analyse_organoid import analyse_organoid
from utils.load_model import load_model
from dialog import dialog

cell_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\TEMP\Train model cell segmentation\models\cell_segmentation_organoid2",
)
    
organoid_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\TEMP\Train model whole organoid segmentation\smoothed_XY\models\whole_organoid_segmentation",
    )

organoids = dialog().dirs
for organoid in organoids:
    print(f"Processing organoid: {os.path.basename(organoid)}")
    analyse_organoid(organoid, cell_model, organoid_model)
    print(f"Finished processing organoid: {organoid}")