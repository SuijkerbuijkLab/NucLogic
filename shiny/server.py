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


def summarize_growth(data):
    filtered = data[data["phenotype"].isin(["WT", "CRC"])].copy()

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

    # Merge back the type info
    types = data[["organoid", "type"]].drop_duplicates()
    pivoted = pivoted.merge(types, on="organoid", how="left")

    # Normalize by frame 0 per organoid
    def normalize(group):
        baseline = group[group["frame"] == 0][["WT", "CRC"]]
        if baseline.empty:
            group["relative_wt"] = None
            group["relative_crc"] = None
        else:
            base_vals = baseline.iloc[0]
            group["relative_wt"] = (
                group["WT"] / base_vals["WT"] if base_vals["WT"] > 0 else None
            )
            group["relative_crc"] = (
                group["CRC"] / base_vals["CRC"] if base_vals["CRC"] > 0 else None
            )
        return group

    summary = pivoted.groupby("organoid").apply(normalize).reset_index(drop=True)

    # Compute % composition
    summary["%wt"] = summary["WT"] / (summary["WT"] + summary["CRC"])
    summary["%crc"] = summary["CRC"] / (summary["WT"] + summary["CRC"])
    summary = summary.rename(columns={"frame": "time"})

    return summary


def classify_type(name):
    name_lower = name.lower()

    if "mix" in name_lower:
        return "Mixed"
    elif "wt" in name_lower:
        return "WT"
    elif "crc" in name_lower:
        return "CRC"
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

    @reactive.effect
    @reactive.event(input.run_segmenter)
    def run_segmentation():
        from sam2.build_sam import build_sam2_video_predictor

        from main_functions.analyse_organoid import analyse_organoid
        from utils.load_model import load_model

        def extract_frame_number(filename):
            match = re.search(r"Frame-(\d+)", filename)
            return int(match.group(1)) if match else -1

        # Load models once
        cell_model = load_model(r"models\cell_segmentation_4")
        organoid_model = build_sam2_video_predictor(
            r"models\sam2.1_hiera_s.yaml",
            r"models\sam2.1_hiera_small.pt",
        )

        # Get organoid paths
        base_path = selected_path.get()
        organoid_names = input.organoid_select_segmentation()
        organoids = [os.path.join(base_path, name) for name in organoid_names]

        # Get channel names
        channel_names = [getattr(input, f"channel_{i}")().strip() for i in range(5)]
        channel_names = [name for name in channel_names if name]

        # Get advanced settings
        settings = {
            "cropped_exists": input.cropped_exists(),
            "delete_cropped": input.delete_cropped(),
            "delete_max_proj": input.delete_max_proj(),
            "delete_max_proj_tracked": input.delete_max_proj_tracked(),
        }

        summary_results = []
        results = []

        for i, organoid in enumerate(organoids):
            print(f"Processing organoid: {os.path.basename(organoid)}")
            try:
                analyse_organoid(
                    organoid,
                    cell_model=cell_model,
                    organoid_model=organoid_model,
                    channel_names=channel_names,
                    croped_existing=settings["cropped_exists"],
                )

                df = pd.read_csv(os.path.join(organoid, "summary_results_organoid.csv"))
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
        progress_count.set(i + 1)

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

    @reactive.effect
    @reactive.event(input.launch_napari)
    def open_napari():
        import json

        path = selected_path.get()
        selected = input.organoid_select_napari() or []
        full_paths = [os.path.join(path, name) for name in selected]

        # Save paths to a temp file
        with open(r"miscellaneous\napari_paths.json", "w") as f:
            json.dump(full_paths, f)

        # Launch Napari viewer
        subprocess.Popen([sys.executable, r"shiny\napari_launcher.py"])

    @reactive.calc
    def organoid_data():
        path = selected_path.get()
        selected = input.organoid_select_plotting()
        if not path or not selected:
            return pd.DataFrame()

        data = []
        for organoid in selected:
            property_path = os.path.join(path, organoid, "properties")
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
                df.insert(0, "type", classify_type(organoid))
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
                    label=f"{organoid} WT",
                    alpha=0.6,
                )
                ax.plot(
                    group["time"],
                    group["relative_crc"],
                    "g-",
                    marker="o",
                    label=f"{organoid} CRC",
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
                label="WT",
                capsize=4,
            )
            ax.errorbar(
                mean["time"],
                mean["relative_crc"],
                yerr=std["relative_crc"],
                fmt="g-o",
                label="CRC",
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
