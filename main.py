from main_functions.dialog import dialog
from main_functions.progressbar import progressbar

# Importing and loading the UI first in the script, to make sure the user sees the UI before all the slow loading of cellpose etc starts
ui = dialog()
ui.show()
organoids = ui.dirs
channel_names = [name for name in ui.channel_info]
channel_names = [name for name in channel_names if name != ""]

import os
import re

from main_functions.analyse_organoid import analyse_organoid
from utils.load_model import load_model
from sam2.build_sam import build_sam2_video_predictor
import traceback
import shutil
import pandas as pd

# Loading the model that is used to segment cells
cell_model = load_model(
    r"C:\Users\6331823\Local SSD\Train model cell segmentation\models\cell_segmentation_4",
)

# Loading the model that is used to segment the important organoid
# organoid_model = load_model(
#     r"C:\Users\6331823\Local SSD\Train model whole organoid segmentation\smoothed_XY\models\whole_liver_organoid_segmentation",
# )
CHECKPOINT = r"C:\Users\6331823\Downloads\sam2.1_hiera_small.pt"
CONFIG = r"C:\Users\6331823\Downloads\sam2.1_hiera_s.yaml"
organoid_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)


def extract_frame_number(filename):
    match = re.search(r"Frame-(\d+)", filename)
    return int(match.group(1)) if match else -1


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

        # Make an overview results of all segmented organoids
        df = pd.read_csv(os.path.join(organoid, "summary_results_organoid.csv"))
        summary_results.append(df)

        # Make a file that really has all data collected of all organoids
        all_properties = []
        files = [
            f
            for f in os.listdir(os.path.join(organoid, "properties"))
            if f.endswith(".csv")
        ]
        files = sorted(files, key=extract_frame_number)
        for file in files:
            df = pd.read_csv(os.path.join(organoid, "properties", file))
            df.insert(0, "organoid", os.path.basename(organoid))
            all_properties.append(df)
        all_properties = pd.concat(all_properties, ignore_index=True)
        results.append(all_properties)
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


# Dataframe that will contain all summarised data
summary_results = []

# Dataframe that will contain all data
results = []

# Running the function and progressbar
progress = progressbar(total_tasks=len(organoids))
progress.root.after(10, lambda: process_next(0))
progress.root.mainloop()

# Saving the summary results data frame
summary_results = pd.concat(summary_results, ignore_index=True)
summary_results_csv = os.path.join(os.path.dirname(organoids[0]), "summary_results.csv")
summary_results.to_csv(summary_results_csv, index=False)

# Saving the  results data frame
results = pd.concat(results, ignore_index=True)
results_csv = os.path.join(os.path.dirname(organoids[0]), "full_results.csv")
results.to_csv(results_csv, index=False)
