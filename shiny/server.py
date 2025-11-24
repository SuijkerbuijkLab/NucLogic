import os
import re
import shutil
import subprocess
import sys
import tkinter as tk
import traceback
from datetime import datetime
from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd

from shiny import reactive, render, ui

from main_functions import count_cell_types


def summarize_growth(data):
    filtered = data[data["phenotype"].isin(["wt", "crc"])].copy()

    # Count phenotype occurrences per organoid and frame
    counts = (
        filtered.groupby(["organoid", "frame", "phenotype"])
        .size()
        .reset_index(name="count")
    )

    # Pivot so each phenotype is a column
    pivoted = counts.pivot_table(
        index=["organoid", "frame"], columns="phenotype", values="count", fill_value=0
    ).reset_index()

    # Add missing columns if they don't exist
    if "wt" not in pivoted.columns:
        pivoted["wt"] = 0
    if "crc" not in pivoted.columns:
        pivoted["crc"] = 0

    # Merge back the type info
    types = data[["organoid", "type"]].drop_duplicates()
    pivoted = pivoted.merge(types, on="organoid", how="left")

    # Normalize by frame 0 per organoid
    def normalize(group):
        baseline = group[group["frame"] == 0][["wt", "crc"]]
        if baseline.empty:
            group["relative_wt"] = None
            group["relative_crc"] = None
        else:
            base_vals = baseline.iloc[0]
            group["relative_wt"] = (
                group["wt"] / base_vals["wt"] if base_vals["wt"] > 0 else None
            )
            group["relative_crc"] = (
                group["crc"] / base_vals["crc"] if base_vals["crc"] > 0 else None
            )
        return group

    summary = pivoted.groupby("organoid").apply(normalize).reset_index(drop=True)

    # Compute % composition
    summary["%wt"] = summary["wt"] / (summary["wt"] + summary["crc"])
    summary["%crc"] = summary["crc"] / (summary["wt"] + summary["crc"])
    print(summary)
    summary = summary.rename(columns={"frame": "time"})

    return summary


def classify_type(df):
    phenotypes = df["phenotype"].unique()

    # Check if there's only one phenotype and it's "crc" (case insensitive)
    if len(phenotypes) == 1 and phenotypes[0].lower() == "crc":
        return "crc"
    # Check if there's only one phenotype and it's "wt" (case insensitive)
    elif len(phenotypes) == 1 and phenotypes[0].lower() == "wt":
        return "wt"
    elif len(phenotypes) == 2:
        return "mix"
    else:
        return "unknown"


def server(input, output, session):
    selected_path = reactive.Value("")

    @reactive.effect
    @reactive.event(
        input.browse_dirs_segmentation,
        input.browse_dirs_cropper,
        input.browse_dirs_plotting,
        input.browse_dirs_napari,
    )
    def run_launcher():
        print("Button clicked — launching folder picker...")
        try:
            subprocess.run(["python", r"shiny\launcher.py"], check=True)
            with open(r"miscellaneous\selected_path.txt", "r") as f:
                path = f.read().strip()
                selected_path.set(path)
                print(f"✅ Path loaded into app: {path}")
        except Exception as e:
            print(f"❌ Error running launcher: {e}")

    selection_state = reactive.Value(True)

    @reactive.effect
    @reactive.event(
        input.toggle_select_segmentation,
        input.toggle_select_cropper,
        input.toggle_select_plotting,
        input.toggle_select_napari,
    )
    def toggle_selection():
        selection_state.set(not selection_state.get())
        print("Togled selection state:", selection_state.get())

    def organoid_list_core(path, selection_state, input_id):
        _ = selection_state.get()

        if path and os.path.isdir(path):
            try:
                files = os.listdir(path)
                newfiles = [
                    file for file in files if os.path.isdir(os.path.join(path, file))
                ]
                if not newfiles:
                    return ui.markdown("**No valid organoids found.**")

                selected = newfiles if selection_state.get() else []

                return ui.input_selectize(
                    input_id,
                    "Select organoids",
                    multiple=True,
                    choices=newfiles,
                    selected=selected,
                )
            except Exception as e:
                return ui.markdown(f"**Error reading directory:** {e}")
        else:
            return ui.markdown("**Invalid or empty path.**")

    @render.ui
    def organoid_list_plotting():
        return organoid_list_core(
            selected_path.get(), selection_state, "organoid_select_plotting"
        )

    @render.ui
    def organoid_list_cropper():
        return organoid_list_core(
            selected_path.get(), selection_state, "organoid_select_cropper"
        )

    @render.ui
    def organoid_list_segmentation():
        return organoid_list_core(
            selected_path.get(), selection_state, "organoid_select_segmentation"
        )

    @render.ui
    def organoid_list_napari():
        return organoid_list_core(
            selected_path.get(), selection_state, "organoid_select_napari"
        )

    @render.ui
    def cropping_options():
        if input.is_fixed():
            return ui.tags.div(
                ui.input_checkbox(
                    "automatic_cropping",
                    "Fixed organoids are cropped automatically, often fails when organoids touch each other",
                    True,
                ),
                ui.input_checkbox(
                    "semi_automatic_cropping",
                    "Organoids that are predicted to touch will be prompted for manual cropping",
                    False,
                ),
                ui.input_checkbox(
                    "manual_cropping",
                    "All organoids must be manually cropped",
                    False,
                ),
                style="margin-left: 20px; border-left: 2px solid #ccc; padding-left: 10px;",
            )
        else:
            return ui.tags.div()

    @render.ui
    def specific_measurements_options():
        if input.specific_measurements():
            return ui.tags.div(
                ui.input_file(
                    "specific_model_file",
                    "Select a custom model file for specific segmentation",
                ),
                style="margin-left: 20px; border-left: 2px solid #ccc; padding-left: 10px;",
            )
        else:
            return ui.tags.div()

    @render.ui
    def specific_measurements_options():
        if input.specific_measurements():
            return ui.tags.div(
                ui.input_file(
                    "specific_model_file",
                    "Select a custom model file for specific segmentation",
                ),
                ui.input_slider(
                    "ratio_threshold",
                    "Set the ratio threshold for specific segmentation",
                    min=-2,
                    max=2,
                    value=0,
                    step=0.1,
                ),
                ui.input_slider(
                    "filter_specific_size",
                    "Set the filter size for specific segmentation",
                    min=0,
                    max=5,
                    value=0,
                    step=1,
                ),
                style="margin-left: 20px; border-left: 2px solid #ccc; padding-left: 10px;",
            )
        else:
            return ui.tags.div()

    @reactive.effect
    @reactive.event(input.run_segmenter)
    def run_segmentation():
        from sam2.build_sam import build_sam2_video_predictor

        from main_functions.count_cell_types import count_cell_types
        from main_functions.analyse_organoid import analyse_organoid
        from utils.load_model import load_model
        from utils.max_project import max_project
        from utils.find_input_file import find_input_file
        from utils.find_mask_area import find_mask_area
        from utils.crop_fixed import crop_fixed
        import tifffile

        progress_count.set(0)  # Reset progress at start

        def extract_frame_number(filename):
            match = re.search(r"Frame-(\d+)", filename)
            return int(match.group(1)) if match else -1

        # Get channel names
        channel_names = [
            getattr(input, f"channel_name_{i}")().strip() for i in range(5)
        ]

        # Get alternative channel names when different than Nuclei wt crc channels are added
        channel_names_alternative = [
            getattr(input, f"channel_name_alternative_{i}")().strip() for i in range(5)
        ]

        # Make a list of the final channel names incl wt crc Nuclei and different names
        final_channel_names = []
        for i in range(len(channel_names)):
            if channel_names[i] == "Different":
                final_channel_names.append(channel_names_alternative[i])
            elif channel_names[i] == "Empty":
                continue
            else:
                final_channel_names.append(channel_names[i])

        # Load models once
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, "models")

        if input.model_file():
            uploaded_files = input.model_file()
            # Then access the first element from that list
            print(f"Using custom model: {uploaded_files[0]['name']}")
            cell_model = load_model(uploaded_files[0]["datapath"])
        else:
            cell_model = load_model(os.path.join(model_dir, "cell_segmentation_4"))

        if input.specific_measurements():
            uploaded_files = input.specific_model_file()
            print(f"Specific model file input: {uploaded_files}")

            if uploaded_files and len(uploaded_files) > 0:
                print("Loading custom specific model")
                print(f"Using custom specific model: {uploaded_files[0]['name']}")
                specific_model = load_model(uploaded_files[0]["datapath"])

        organoid_model = build_sam2_video_predictor(
            os.path.join(model_dir, "sam2.1_hiera_s.yaml"),
            os.path.join(model_dir, "sam2.1_hiera_small.pt"),
        )

        # Get organoid paths
        base_path = selected_path.get()
        base_path = os.path.normpath(base_path)
        organoid_names = input.organoid_select_segmentation()
        organoids = [os.path.join(base_path, name) for name in organoid_names]

        # Get advanced settings
        settings = {
            "cropped_exists": input.cropped_exists(),
            "delete_cropped": input.delete_cropped(),
            "delete_max_proj": input.delete_max_proj(),
            "delete_max_proj_tracked": input.delete_max_proj_tracked(),
            "is_fixed": False,
            "specific_measurements": False,
            "automatic_cropping": True,
            "manual_cropping": False,
            "semi_automatic_cropping": False,
        }

        if input.is_fixed():
            settings["is_fixed"] = True
            settings["automatic_cropping"] = input.automatic_cropping()
            settings["manual_cropping"] = input.manual_cropping()
            settings["semi_automatic_cropping"] = input.semi_automatic_cropping()
        if input.specific_measurements():
            settings["specific_measurements"] = input.specific_measurements()
            settings["filter_specific_size"] = input.filter_specific_size()
            settings["ratio_threshold"] = input.ratio_threshold()

        if settings["is_fixed"] and not settings["cropped_exists"]:
            for organoid in organoids:
                channel_indices = {
                    ch.lower(): i
                    for i, ch in enumerate(final_channel_names)
                    if ch.strip()
                }
                nuclei_channel = channel_indices.get("nuclei", -1)
                input_file = find_input_file(input_directory=organoid)
                proj_XY = max_project(
                    input_file,
                    nuclei=nuclei_channel,
                    fixed=True,
                    name=os.path.basename(organoid),
                )
                area = find_mask_area(proj_XY=proj_XY)

                if settings["manual_cropping"] or (
                    settings["semi_automatic_cropping"] and area > 200000
                ):
                    print(
                        f"Organoid {os.path.basename(organoid)} requires manual cropping (area: {area})"
                    )
                    crop_fixed(
                        proj_XY=proj_XY,
                        input_file=input_file,
                        output_directory=os.path.join(organoid, "cropped"),
                        nuclei=nuclei_channel,
                        name=os.path.basename(organoid),
                        manual=True,
                    )

                else:
                    crop_fixed(
                        proj_XY=proj_XY,
                        input_file=input_file,
                        output_directory=os.path.join(organoid, "cropped"),
                        nuclei=nuclei_channel,
                        name=os.path.basename(organoid),
                        manual=False,
                    )

        summary_results = []
        results = []

        for i, organoid in enumerate(organoids):
            print(f"Processing organoid: {os.path.basename(organoid)}")

            try:
                if settings["specific_measurements"]:
                    count_dapi, count_lyz, count_aldob = count_cell_types(
                        input_image_path=os.path.join(
                            organoid, "cropped", "Frame-0.tif"
                        ),
                        model=cell_model,
                        specific_model=specific_model,
                        channel_names=final_channel_names,
                        output_directory=organoid,
                        ratio_threshold=settings["ratio_threshold"],
                        filter_specific_size=settings["filter_specific_size"],
                    )
                    cell_type_measurements = pd.DataFrame(
                        [
                            {
                                "sample": os.path.basename(organoid),
                                "count_nuclei": count_dapi,
                                "count_lyz": count_lyz,
                                "count_aldob": count_aldob,
                                "percentage_lyz": (
                                    (count_lyz / count_dapi) * 100
                                    if count_dapi > 0
                                    else 0
                                ),
                                "percentage_aldob": (
                                    (count_aldob / count_dapi) * 100
                                    if count_dapi > 0
                                    else 0
                                ),
                            }
                        ]
                    )
                    summary_results.append(cell_type_measurements)

                if (
                    settings["semi_automatic_cropping"] or settings["manual_cropping"]
                ) and not settings["specific_measurements"]:
                    analyse_organoid(
                        organoid,
                        cell_model=cell_model,
                        organoid_model=organoid_model,
                        channel_names=final_channel_names,
                        cropped_existing=True,
                        is_fixed=settings["is_fixed"],
                    )
                if (
                    not (
                        settings["semi_automatic_cropping"]
                        or settings["manual_cropping"]
                    )
                    and not settings["specific_measurements"]
                ):
                    analyse_organoid(
                        organoid,
                        cell_model=cell_model,
                        organoid_model=organoid_model,
                        channel_names=final_channel_names,
                        cropped_existing=settings["cropped_exists"],
                        is_fixed=settings["is_fixed"],
                    )

                if not settings["is_fixed"]:
                    df = pd.read_csv(
                        os.path.join(organoid, "summary_results_organoid.csv")
                    )
                    summary_results.append(df)

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

            # Cleanup
            if settings["delete_cropped"]:
                cropped_path = os.path.join(organoid, "cropped")
                if os.path.exists(cropped_path):
                    shutil.rmtree(cropped_path)

            max_proj = [f for f in os.listdir(organoid) if f.endswith("projXY.tif")]
            max_proj_tracked = [
                f for f in os.listdir(organoid) if f.endswith("projXY_tracked.tif")
            ]

            if settings["delete_max_proj"] and max_proj:
                path = os.path.join(organoid, max_proj[0])
                if os.path.exists(path):
                    os.remove(path)

            if settings["delete_max_proj_tracked"] and max_proj_tracked:
                path = os.path.join(organoid, max_proj_tracked[0])
                if os.path.exists(path):
                    os.remove(path)

            print(f"Finished processing organoid: {organoid}")
            progress_count.set(i + 1)

        # Save results
        if summary_results:
            summary_df = pd.concat(summary_results, ignore_index=True)
            summary_df.to_csv(
                os.path.join(base_path, "summary_results.csv"), index=False
            )

        if results:
            full_df = pd.concat(results, ignore_index=True)
            full_df.to_csv(os.path.join(base_path, "full_results.csv"), index=False)

        print("✅ Segmentation complete.")

    progress_count = reactive.Value(0)

    @render.ui
    def segmentation_progress():
        total = len(input.organoid_select_segmentation() or [])
        completed = progress_count.get()

        if total == 0:
            return ui.markdown("**No organoids selected.**")

        percent = int((completed / total) * 100)

        return ui.div(
            ui.tags.label(f"Segmented: {completed}/{total} organoids ({percent}%)"),
            ui.div(
                ui.div(
                    style=f"width: {percent}%; background-color: #4caf50; height: 20px; border-radius: 4px;"
                ),
                style="width: 100%; background-color: #eee; border-radius: 4px;",
            ),
        )

    recalculate_stats_progress_state = reactive.Value(None)

    @render.ui
    def recalculate_stats_progress():
        return recalculate_stats_progress_state.get()

    @reactive.effect
    @reactive.event(input.recalculate_statistics)
    def recalculate_statistics():

        # Show spinner that shows function is running
        recalculate_stats_progress_state.set(
            ui.tags.div(
                ui.tags.span("⏳", style="font-size: 1.5em; margin-right: 8px;"),
                "Recalculating...",
                style="display: flex; align-items: center;",
            )
        )

        from main_functions.recalculate_statistics import recalculate_statistics

        # Get organoid paths
        base_path = selected_path.get()
        # Make path normal in case of different OS
        base_path = os.path.normpath(base_path)
        organoid_names = input.organoid_select_segmentation()
        organoids_props = [
            os.path.join(base_path, name, "properties") for name in organoid_names
        ]

        # Here we will create a dataframe that has the data of all selected mixed organoids and all frames
        all_data = []
        for organoid in organoids_props:
            sample_data = []
            props = [file for file in os.listdir(organoid) if file.endswith(".csv")]
            for prop in props:
                data = pd.read_csv(os.path.join(organoid, prop))
                sample_data.append(data)
            sample_data = pd.concat(sample_data, ignore_index=True)
            sample_data.insert(0, "sample", os.path.basename(os.path.dirname(organoid)))
            all_data.append(sample_data)
        all_data = pd.concat(all_data, ignore_index=True)

        # Now we recalculate the statistics for this dataframe
        all_data = recalculate_statistics(all_data)

        # Now we split the data back into the individual organoids and save them in a new folder called properties_recalculated
        for organoid in organoids_props:
            data = all_data[
                all_data["sample"] == os.path.basename(os.path.dirname(organoid))
            ]
            os.makedirs(
                os.path.join(os.path.dirname(organoid), "properties_recalculated"),
                exist_ok=True,
            )
            for _, frame in data.groupby("frame"):
                frame_number = frame["frame"].iloc[0]
                save_path = os.path.join(
                    os.path.dirname(organoid), "properties_recalculated"
                )
                frame.to_csv(
                    os.path.join(save_path, f"Frame-{frame_number:02d}_props.csv"),
                    index=False,
                )

        print("✅ Recalculation complete.")

        # Show checkmark when function is finished in ui
        recalculate_stats_progress_state.set(
            ui.tags.div(
                ui.tags.span("✅", style="font-size: 1.5em; margin-right: 8px;"),
                "Done!",
                style="display: flex; align-items: center; color: green;",
            )
        )

    @reactive.effect
    @reactive.event(input.run_cropper)
    def run_cropper():
        is_cropping.set(True)  # Show spinner
        from sam2.build_sam import build_sam2_video_predictor

        from main_functions.crop_organoid import crop_organoid

        # Load models once
        organoid_model = build_sam2_video_predictor(
            r"models\sam2.1_hiera_s.yaml",
            r"models\sam2.1_hiera_small.pt",
        )

        # Get organoid paths
        base_path = selected_path.get()
        organoid_names = input.organoid_select_cropper()
        organoids = [os.path.join(base_path, name) for name in organoid_names]

        # Get channel names
        channel_names = [
            getattr(input, f"channel_cropper_{i}")().strip() for i in range(5)
        ]
        channel_names = [name for name in channel_names if name]

        channel_colors = [
            getattr(input, f"channel_color_cropper_{i}")() for i in range(5)
        ]
        channel_colors = [
            color
            for i, color in enumerate(channel_colors)
            if channel_names and i < len(channel_names)
        ]

        # Get advanced settings
        settings = {
            "cropped_exists": input.cropped_exists_cropper(),
            "delete_cropped": input.delete_cropped_cropper(),
            "delete_max_proj": input.delete_max_proj_cropper(),
            "delete_max_proj_tracked": input.delete_max_proj_tracked_cropper(),
        }

        # Loop through organoids and run cropping
        for i, organoid in enumerate(organoids):
            print(f"Cropping organoid: {os.path.basename(organoid)}")
            try:
                crop_organoid(
                    input_directory=organoid,
                    organoid_model=organoid_model,
                    channel_names=channel_names,
                    channel_colors=channel_colors,
                    crop_existing=settings["cropped_exists"],
                )
            except Exception as e:
                print(f"Failed to crop {organoid}: {e}")
            else:
                print(f"Finished cropping: {organoid}")

            # Cleanup
            if settings["delete_cropped"]:
                cropped_path = os.path.join(organoid, "cropped")
                if os.path.exists(cropped_path):
                    shutil.rmtree(cropped_path)

            max_proj = [f for f in os.listdir(organoid) if f.endswith("projXY.tif")]
            max_proj_tracked = [
                f for f in os.listdir(organoid) if f.endswith("projXY_tracked.tif")
            ]

            if settings["delete_max_proj"] and max_proj:
                path = os.path.join(organoid, max_proj[0])
                if os.path.exists(path):
                    os.remove(path)

            if settings["delete_max_proj_tracked"] and max_proj_tracked:
                path = os.path.join(organoid, max_proj_tracked[0])
                if os.path.exists(path):
                    os.remove(path)

            progress_count_cropper.set(i + 1)
            is_cropping.set(False)  # Hide spinner

    is_cropping = reactive.Value(False)

    @render.ui
    def cropper_spinner():
        if is_cropping.get():
            return ui.tags.span(
                "⏳ Cropping...", style="margin-left: 10px; color: #888;"
            )
        else:
            return ui.tags.span("")  # Empty when not running

    progress_count_cropper = reactive.Value(0)

    @render.ui
    def cropper_progress():
        total = len(input.organoid_select_cropper() or [])
        completed = progress_count_cropper.get()

        if total == 0:
            return ui.markdown("**No organoids selected.**")

        percent = int((completed / total) * 100)

        return ui.div(
            ui.tags.label(f"Segmented: {completed}/{total} organoids ({percent}%)"),
            ui.div(
                ui.div(
                    style=f"width: {percent}%; background-color: #4caf50; height: 20px; border-radius: 4px;"
                ),
                style="width: 100%; background-color: #eee; border-radius: 4px;",
            ),
        )

    @reactive.calc
    def organoid_data():
        path = selected_path.get()
        selected = input.organoid_select_plotting()
        if not path or not selected:
            return pd.DataFrame()

        # Check the switch value to determine which folder to use
        use_recalculated = input.use_recalculated_statistics()
        folder_name = "properties_recalculated" if use_recalculated else "properties"

        data = []
        for organoid in selected:
            property_path = os.path.join(path, organoid, folder_name)
            if not os.path.isdir(property_path):
                continue
            props = []
            for file in os.listdir(property_path):
                file_path = os.path.join(property_path, file)
                if file.endswith(".csv"):
                    try:
                        prop = pd.read_csv(file_path)
                        props.append(prop)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            if props:
                df = pd.concat(props, ignore_index=True)
                df.insert(0, "organoid", organoid)
                df.insert(0, "type", classify_type(df))
                data.append(df)

        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    @render.download(
        filename=lambda: f"organoid_data_{datetime.now():%Y-%m-%d_%H-%M}.csv"
    )
    def download_data():
        import io

        df = organoid_data()
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    @reactive.calc
    def growth_summary():
        df = organoid_data()
        if df.empty or "phenotype" not in df.columns or "frame" not in df.columns:
            return pd.DataFrame()
        return summarize_growth(df)

    @render.plot
    def growth_by_sample_plot():
        df = growth_summary()
        ylim_max = input.ylim_max()

        if df.empty or "type" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.axis("off")
            return fig

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)

        for i, type_name in enumerate(df["type"].unique()):
            ax = axes[i]
            subset = df[df["type"] == type_name]

            for organoid, group in subset.groupby("organoid"):
                ax.plot(
                    group["time"],
                    group["relative_wt"],
                    "m-",
                    marker="o",
                    label=f"{organoid} wt",
                    alpha=0.6,
                )
                ax.plot(
                    group["time"],
                    group["relative_crc"],
                    "g-",
                    marker="o",
                    label=f"{organoid} crc",
                    alpha=0.6,
                )
            ax.set_title(f"Type: {type_name}")
            ax.set_xticks(sorted(subset["time"].unique())[::2])
            # ax.legend(fontsize="small", loc="upper left", ncol=1)

            # Apply y-axis limit if provided
            if ylim_max is not None:
                ax.set_ylim(top=ylim_max)

        fig.suptitle("Relative cell growth per sample", x=0.54)
        fig.supxlabel("Time (h)", x=0.54)
        fig.supylabel("Number of cells per organoid\n       (normalized to t=0)")
        plt.tight_layout()
        return fig

    @render.plot
    def growth_by_type_plot():
        df = growth_summary()
        ylim_max = input.ylim_max()

        if df.empty or "type" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.axis("off")
            return fig

        grouped = df.groupby(["type", "time"])
        data_mean = grouped[["relative_wt", "relative_crc"]].mean().reset_index()
        data_std = grouped[["relative_wt", "relative_crc"]].std().reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)

        for i, type_name in enumerate(data_mean["type"].unique()):
            mean = data_mean[data_mean["type"] == type_name]
            std = data_std[data_std["type"] == type_name]

            ax = axes[i]
            ax.errorbar(
                mean["time"],
                mean["relative_wt"],
                yerr=std["relative_wt"],
                fmt="m-o",
                label="wt",
                capsize=4,
            )
            ax.errorbar(
                mean["time"],
                mean["relative_crc"],
                yerr=std["relative_crc"],
                fmt="g-o",
                label="crc",
                capsize=4,
            )
            ax.set_title(f"Type: {type_name}")
            ax.set_xticks(mean["time"][::2])
            ax.legend()

            # Apply y-axis limit if provided
            if ylim_max is not None:
                ax.set_ylim(top=ylim_max)

        fig.suptitle("Relative cell growth", x=0.54)
        fig.supxlabel("Time (h)", x=0.54)
        fig.supylabel("Number of cells per organoid\n       (normalized to t=0)")
        plt.tight_layout()
        return fig

    @reactive.effect
    @reactive.event(input.launch_napari)
    def open_napari():
        import json

        path = selected_path.get()
        selected = input.organoid_select_napari() or []
        full_paths = [os.path.join(path, name) for name in selected]

        files = []
        for path in full_paths:
            if os.path.isdir(path):
                files.extend(
                    [
                        f
                        for f in os.listdir(path)
                        if f.endswith(".tif") or f.endswith(".ims")
                    ]
                )

        to_visualize = []
        for organoid_path in full_paths:
            organoid_files = [
                f for f in os.listdir(organoid_path) if f.endswith((".tif", ".ims"))
            ]

            if input.segmentation_result():
                to_visualize += [
                    os.path.join(organoid_path, f)
                    for f in organoid_files
                    if "segmented" in f
                ]
            if input.projXY():
                to_visualize += [
                    os.path.join(organoid_path, f)
                    for f in organoid_files
                    if "projXY.tif" in f
                ]
            if input.projXY_tracked():
                to_visualize += [
                    os.path.join(organoid_path, f)
                    for f in organoid_files
                    if "projXY_tracked" in f
                ]

        if input.full_movie():
            for organoid_name in selected:
                movie_path = os.path.join(path, f"{organoid_name}.tif")
                if os.path.exists(movie_path):
                    to_visualize.append(movie_path)

        # Save paths to a temp file
        with open(r"miscellaneous\napari_paths.json", "w") as f:
            json.dump(to_visualize, f)

        # Launch Napari viewer
        subprocess.Popen([sys.executable, r"shiny\napari_launcher.py"])
