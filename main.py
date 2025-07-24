from main_functions.dialog import dialog
from main_functions.progressbar import progressbar

ui = dialog()
ui.show()
organoids = ui.dirs

channel_names = [name for name in ui.channel_info]

import os

from main_functions.analyse_organoid import analyse_organoid
from utils.load_model import load_model
from dialog import dialog
import traceback
import shutil

cell_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\Train model cell segmentation\models\cell_segmentation_organoid2",
)

organoid_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\Train model whole organoid segmentation\smoothed_XY\models\whole_organoid_segmentation",
)

def process_next(index):
    if index >= len(organoids):
        progress.close()
        return

    organoid = organoids[index]
    print(f"Processing organoid: {os.path.basename(organoid)}")

    try:
        analyse_organoid(organoid, 
                         cell_model, 
                         organoid_model,
                         channel_names=channel_names,
                         croped_existing=ui.advanced_settings["cropped_exists"])
    except Exception as e:
        print(f"Skipped organoid due to error: {e}")
        traceback.print_exc()

    # if viewer is None:
    #     viewer = napari.Viewer()
    #     napari.run()
    # viewer.add_labels(
    #     tifffile.imread(os.path.join(organoid, "segmented_movie.tif")),
    #     name=os.path.basename(organoid),
    #     scale=(1, 4.94 / 0.621, 1, 1),
    # )

    if ui.advanced_settings["delete_cropped"] and os.path.exists(
        os.path.join(organoid, "cropped")
    ):
        shutil.rmtree(os.path.join(organoid, "cropped"))

    max_project = [f for f in os.listdir(organoid) if f.endswith("projXY.tif")]

    # Delete max projection if it exists and deletion is enabled
    if ui.advanced_settings["delete_max_proj"] and max_project:
        path = os.path.join(organoid, max_project[0])
        if os.path.exists(path):
            os.remove(path)

    print(f"Finished processing organoid: {organoid}")

    progress.update(index + 1)
    progress.root.after(10, lambda: process_next(index + 1))


progress = progressbar(total_tasks=len(organoids))
progress.root.after(10, lambda: process_next(0))
progress.root.mainloop()