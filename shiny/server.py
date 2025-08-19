from shiny import reactive, render, ui

import pandas as pd
import matplotlib.pyplot as plt

import os


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
    @render.ui
    @reactive.event(input.show_slider)
    def ui_slider():
        if input.show_slider():
            value = input.slider() if "slider" in input else 5
            return ui.input_slider(
                "slider", "Choose a number", min=1, max=10, value=value
            )

    selection_state = reactive.Value(True)

    @reactive.effect
    @reactive.event(input.toggle_select)
    def _():
        selection_state.set(not selection_state.get())

    @render.text
    def organoid_list():
        path = input.path()
        if path and os.path.isdir(path):
            try:
                files = os.listdir(path)
                newfiles = [
                    file
                    for file in files
                    if os.path.isdir(os.path.join(path, file))
                    and "properties" in os.listdir(os.path.join(path, file))
                ]
                if not newfiles:
                    return ui.markdown("**No valid organoids found.**")

                selected = newfiles if selection_state.get() else []

                return ui.input_selectize(
                    "organoid_select",
                    "Select organoids",
                    multiple=True,
                    choices=newfiles,
                    selected=selected,
                )
            except Exception as e:
                return ui.markdown(f"**Error reading directory:** {e}")
        else:
            return ui.markdown("**Invalid or empty path.**")

    @reactive.calc
    def organoid_data():
        path = input.path()
        selected = input.organoid_select()
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

    @reactive.calc
    def growth_summary():
        df = organoid_data()
        if df.empty or "phenotype" not in df.columns or "frame" not in df.columns:
            return pd.DataFrame()
        return summarize_growth(df)

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
            ax.set_title(f"Sample: {type_name}")
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
