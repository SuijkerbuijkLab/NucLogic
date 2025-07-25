from main_functions.dialog import dialog
from main_functions.progressbar import progressbar

# Importing and loading the UI first in the script, to make sure the user sees the UI before all the slow loading of cellpose etc starts
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

# Loading the model that is used to segment cells
cell_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\Train model cell segmentation\models\cell_segmentation_organoid2",
)

# Loading the model that is used to segment the important organoid
organoid_model = load_model(
    r"E:\Users\Sebastian_van_Dijk\Train model whole organoid segmentation\smoothed_XY\models\whole_organoid_segmentation",
)


# This process next function is the main loop, it is encoded in this function to make sure the progress bar of the UI works and updates after every organoid
def process_next(index):
    # Stops the loop once all organoids have been analysed
    if index >= len(organoids):
        progress.close()
        return

    # Set the organoid to analyse in this instance of the loop
    organoid = organoids[index]
    print(f"Processing organoid: {os.path.basename(organoid)}")

    # Running the main function of the script, using a try except block so that the script continues during batch processing if a single organoid fails
    try:
        analyse_organoid(
            organoid,
            cell_model=cell_model,
            organoid_model=organoid_model,
            channel_names=channel_names,
            croped_existing=ui.advanced_settings["cropped_exists"],
        )
    except Exception as e:
        print(f"Skipped organoid due to error: {e}")
        traceback.print_exc()

    # Delete the cropped folder if the user specified this, this will save a lot of space on the disk
    if ui.advanced_settings["delete_cropped"] and os.path.exists(
        os.path.join(organoid, "cropped")
    ):
        shutil.rmtree(os.path.join(organoid, "cropped"))

    # Find the max XY and max XY tracked projections
    max_project = [f for f in os.listdir(organoid) if f.endswith("projXY.tif")]
    max_project_tracked = [
        f for f in os.listdir(organoid) if f.endswith("projXY_tracked.tif")
    ]

    # Delete max and max tracked projection if it exists and deletion is enabled
    if ui.advanced_settings["delete_max_proj"] and max_project:
        path = os.path.join(organoid, max_project[0])
        if os.path.exists(path):
            os.remove(path)
    if ui.advanced_settings["delete_max_proj_tracked"] and max_project_tracked:
        path = os.path.join(organoid, max_project_tracked[0])
        if os.path.exists(path):
            os.remove(path)

    print(f"Finished processing organoid: {organoid}")

    # End the loop by updating the index to go to the next organoid in the list
    progress.update(index + 1)
    progress.root.after(10, lambda: process_next(index + 1))


# Running the function and progressbar
progress = progressbar(total_tasks=len(organoids))
progress.root.after(10, lambda: process_next(0))
progress.root.mainloop()
