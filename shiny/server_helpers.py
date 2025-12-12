import os
import re
import shutil
import pandas as pd


def needs_manual_cropping(settings, area):
    """Determine if manual cropping is required."""
    if settings["manual_cropping"]:
        return True
    if settings["semi_automatic_cropping"] and area > 200000:
        return True
    return False


def process_fixed_cropping(organoids, settings, final_channel_names):
    """Handle cropping for fixed samples."""
    from utils.find_input_file import find_input_file
    from utils.max_project import max_project
    from utils.find_mask_area import find_mask_area
    from utils.crop_fixed import crop_fixed

    if not (settings["is_fixed"] and not settings["cropped_exists"]):
        return

    channel_indices = {
        ch.lower(): i for i, ch in enumerate(final_channel_names) if ch.strip()
    }
    nuclei_channel = channel_indices.get("nuclei", -1)

    for organoid in organoids:
        input_file = find_input_file(input_directory=organoid)
        proj_XY = max_project(
            input_file,
            fixed=True,
            name=os.path.basename(organoid),
        )
        area = find_mask_area(proj_XY=proj_XY)

        manual = needs_manual_cropping(settings, area)
        if manual:
            print(
                f"Organoid {os.path.basename(organoid)} requires manual cropping (area: {area})"
            )

        crop_fixed(
            proj_XY=proj_XY,
            input_file=input_file,
            output_directory=os.path.join(organoid, "cropped"),
            nuclei=nuclei_channel,
            name=os.path.basename(organoid),
            manual=manual,
        )


def perform_specific_measurements(
    organoid, settings, cell_model, specific_model, final_channel_names
):
    """Perform cell type counting for specific measurements."""
    from main_functions import count_cell_types

    count_dapi, count_lyz, count_aldob = count_cell_types(
        input_image_path=os.path.join(organoid, "cropped", "Frame-0.tif"),
        model=cell_model,
        specific_model=specific_model,
        channel_names=final_channel_names,
        output_directory=organoid,
        ratio_threshold=settings["ratio_threshold"],
        filter_specific_size=settings["filter_specific_size"],
    )

    return pd.DataFrame(
        [
            {
                "sample": os.path.basename(organoid),
                "count_nuclei": count_dapi,
                "count_lyz": count_lyz,
                "count_aldob": count_aldob,
                "percentage_lyz": (
                    (count_lyz / count_dapi) * 100 if count_dapi > 0 else 0
                ),
                "percentage_aldob": (
                    (count_aldob / count_dapi) * 100 if count_dapi > 0 else 0
                ),
            }
        ]
    )


def perform_standard_analysis(
    organoid, settings, cell_model, organoid_model, final_channel_names
):
    """Perform standard organoid analysis."""
    from main_functions.analyse_organoid import analyse_organoid

    # Determine if cropped files already exist
    cropped_existing = (
        settings["cropped_exists"]
        or settings["semi_automatic_cropping"]
        or settings["manual_cropping"]
    )

    analyse_organoid(
        organoid,
        cell_model=cell_model,
        organoid_model=organoid_model,
        channel_names=final_channel_names,
        cropped_existing=cropped_existing,
        is_fixed=settings["is_fixed"],
        dual_nuclei=settings["dual_nuclei"],
    )


def load_all_properties(organoid):
    """Load and concatenate all property files."""

    def extract_frame_number(filename):
        match = re.search(r"Frame-(\d+)", filename)
        return int(match.group(1)) if match else -1

    all_properties = []
    properties_path = os.path.join(organoid, "properties")

    if not os.path.exists(properties_path):
        return None

    files = [f for f in os.listdir(properties_path) if f.endswith(".csv")]
    files = sorted(files, key=extract_frame_number)

    for file in files:
        df = pd.read_csv(os.path.join(properties_path, file))
        df.insert(0, "organoid", os.path.basename(organoid))
        all_properties.append(df)

    return pd.concat(all_properties, ignore_index=True) if all_properties else None


def process_organoid(organoid, settings, models, final_channel_names):
    """Process a single organoid based on settings."""
    import traceback

    summary_result = None

    try:
        # Path 1: Specific measurements (fixed samples only)
        if settings["specific_measurements"]:
            summary_result = perform_specific_measurements(
                organoid,
                settings,
                models["cell"],
                models["specific"],
                final_channel_names,
            )

        # Path 2: Standard analysis (live or fixed without specific measurements)
        else:
            perform_standard_analysis(
                organoid,
                settings,
                models["cell"],
                models["organoid"],
                final_channel_names,
            )

            # Load summary results for live cell imaging
            if not settings["is_fixed"]:
                summary_result = pd.read_csv(
                    os.path.join(organoid, "summary_results_organoid.csv")
                )

        # Load all properties
        all_properties = load_all_properties(organoid)

        return summary_result, all_properties

    except Exception as e:
        print(f"Skipped organoid {os.path.basename(organoid)} due to error: {e}")
        traceback.print_exc()
        return None, None


def cleanup_organoid_files(organoid, settings):
    """Clean up temporary files based on settings."""
    if settings["delete_cropped"]:
        cropped_path = os.path.join(organoid, "cropped")
        if os.path.exists(cropped_path):
            shutil.rmtree(cropped_path)

    if settings["delete_max_proj"]:
        max_proj = [f for f in os.listdir(organoid) if f.endswith("projXY.tif")]
        if max_proj:
            path = os.path.join(organoid, max_proj[0])
            if os.path.exists(path):
                os.remove(path)

    if settings["delete_max_proj_tracked"]:
        max_proj_tracked = [
            f for f in os.listdir(organoid) if f.endswith("projXY_tracked.tif")
        ]
        if max_proj_tracked:
            path = os.path.join(organoid, max_proj_tracked[0])
            if os.path.exists(path):
                os.remove(path)


def save_results(base_path, summary_results, results):
    """Save summary and full results."""
    if summary_results:
        summary_df = pd.concat(summary_results, ignore_index=True)
        summary_df.to_csv(os.path.join(base_path, "summary_results.csv"), index=False)

    if results:
        full_df = pd.concat(results, ignore_index=True)
        full_df.to_csv(os.path.join(base_path, "full_results.csv"), index=False)
