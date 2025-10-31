import tifffile
import numpy as np
import pandas as pd
import os
from cellpose.utils import stitch3D
from skimage.measure import regionprops

from utils.stitch_organoid import stitch_organoid
from utils.measure_dilated import measure_dilated


def count_cell_types(
    input_image_path,
    model,
    specific_model=None,
    output_directory=None,
    channel_names=None,
):
    if output_directory is None:
        output_directory = os.path.dirname(input_image_path)

    # Extract channel indices with a dictionary for dynamic access
    channel_indices = {
        ch.lower(): i for i, ch in enumerate(channel_names) if ch.strip()
    }

    # Always process the nuclei channel
    nuclei_channel = channel_indices.get("nuclei", -1)
    if nuclei_channel == -1:
        raise ValueError("Nuclei channel is required.")

    # Full Lyz Alb pipeline
    image = tifffile.imread(input_image_path)

    # Dapi segmentation
    mtmg = (
        image[:, channel_indices.get("mtmg", channel_indices.get("wt", -1))]
        if "mtmg" in channel_indices or "wt" in channel_indices
        else None
    )
    dapi = image[:, nuclei_channel]
    aldob = (
        image[:, channel_indices.get("aldob", -1)]
        if "aldob" in channel_indices
        else None
    )
    lyz = image[:, channel_indices.get("lyz", -1)] if "lyz" in channel_indices else None

    dapi_masks = []
    for z in range(dapi.shape[0]):
        mask, _, _ = model.eval(
            dapi[z],
            diameter=None,
            normalize=True,
            flow_threshold=0.4,
            invert=False,
            resample=True,
            do_3D=False,
            min_size=40,
            niter=200,
        )
        dapi_masks.append(mask)
    dapi_masks = np.stack(dapi_masks, axis=0)
    tifffile.imwrite(
        rf"{output_directory}\dapi_basic_mask.tif",
        dapi_masks,
        compression="zlib",
        compressionargs={"level": 8},
    )

    unfiltered_dapi_masks, dapi_masks, stats_df = stitch_organoid(
        dapi_masks, max_search_range=6, max_height=5
    )

    tifffile.imwrite(
        rf"{output_directory}\dapi_stitched.tif",
        dapi_masks,
        compression="zlib",
        compressionargs={"level": 8},
    )

    count_dapi = len(np.unique(dapi_masks)) - 1  # Subtract 1 to exclude background
    print(f"Counted {count_dapi} DAPI positive cells")

    # Lyz segmentation
    lyz_masks = []
    for z in range(lyz.shape[0]):
        mask, _, _ = specific_model.eval(
            lyz[z],
            diameter=None,
            normalize=True,
            flow_threshold=0.4,
            invert=False,
            resample=True,
            do_3D=False,
            niter=200,
        )
        lyz_masks.append(mask)
    lyz_masks = np.stack(lyz_masks, axis=0)
    tifffile.imwrite(
        rf"{output_directory}\lyz_basic_mask.tif",
        lyz_masks,
        compression="zlib",
        compressionargs={"level": 8},
    )

    lyz_masks = stitch3D(lyz_masks)
    props = regionprops(lyz_masks)
    lyz_tracked = []
    for prop in props:
        lyz_tracked.append(
            {
                "height": prop.bbox[3] - prop.bbox[0],
                "particle": prop.label,  # original 2D label
                "volume": prop.area,
            }
        )
    lyz_tracked = pd.DataFrame(lyz_tracked)

    lyz_tracked_filtered = lyz_tracked[lyz_tracked["height"] > 1]
    lyz_tracked_filtered = lyz_tracked_filtered[lyz_tracked_filtered["volume"] > 600]

    lyz_masks_filtered = np.zeros_like(lyz_masks, dtype=np.uint32)
    for _, row in lyz_tracked_filtered.iterrows():
        # z = int(row["z"])
        particle_id = int(row["particle"])

        # Copy the original mask pixels and relabel with new particle_id
        lyz_masks_filtered[lyz_masks == particle_id] = particle_id

    tifffile.imwrite(
        rf"{output_directory}\lyz_stitched.tif",
        lyz_masks_filtered,
        compression="zlib",
        compressionargs={"level": 8},
    )

    count_lyz = (
        len(np.unique(lyz_masks_filtered)) - 1
    )  # Subtract 1 to exclude background
    print(f"Counted {count_lyz} Lyz positive cells")

    stats, dilated_mask = measure_dilated(
        dapi_masks,
        dilation_size=12,
        channels=[mtmg, dapi, aldob, lyz],
        channel_names=["mTmG", "DAPI", "AldoB", "Lyz"],
    )

    aldob_filtered = stats[stats["log_ratio_aldob_dapi"] > -0.5]

    aldob_masks = np.zeros_like(dapi_masks, dtype=np.uint32)
    for _, row in aldob_filtered.iterrows():
        # z = int(row["z"])
        particle_id = int(row["label"])

        # Copy the original mask pixels and relabel with new particle_id
        aldob_masks[dapi_masks == particle_id] = particle_id

    tifffile.imwrite(
        rf"{output_directory}\aldob_stitched.tif",
        aldob_masks,
        compression="zlib",
        compressionargs={"level": 8},
    )

    count_aldob = len(aldob_filtered)
    print(f"Counted {count_aldob} Aldob positive cells.")

    print(
        f"Found {count_lyz} paneth cells out of {count_dapi} total cells. Which is {count_lyz / count_dapi * 100:.2f}%"
    )
    print(
        f"Found {count_aldob} AldoB positive cells out of {count_dapi} total cells. Which is {count_aldob / count_dapi * 100:.2f}%"
    )

    # rename particle to label in stats_df and merge with stats
    stats_df = stats_df.rename(columns={"particle": "label"})
    final_stats = stats_df.merge(stats, on="label")
    final_stats.to_csv(rf"{output_directory}\cell_stats.csv", index=False)

    return count_dapi, count_lyz, count_aldob
