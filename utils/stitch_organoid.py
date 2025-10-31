import numpy as np
import pandas as pd

from skimage.measure import regionprops
import trackpy as tp

from utils.compute_knn_features import compute_knn_features


def stitch_organoid(masks, max_search_range=6, max_height=5):

    # Step 1: Extract centroids from each Z-slice
    all_centroids = []
    for z in range(masks.shape[0]):
        slice_mask = masks[z]
        props = regionprops(slice_mask)

        for prop in props:
            all_centroids.append(
                {
                    "z": z,  # frame
                    "y": prop.centroid[0],
                    "x": prop.centroid[1],
                    "label_2d": prop.label,  # original 2D label
                    "area": prop.area,
                    "bounding_box": prop.bbox,
                }
            )

    df = pd.DataFrame(all_centroids)

    tp.linking.Linker.MAX_SUB_NET_SIZE = 100

    tracked = tp.link(
        df,
        search_range=max_search_range,  # max movement in XY
        memory=0,  # allow gaps in tracking
        pos_columns=["y", "x"],  # columns for position
        t_column="z",  # use Z as time
    )

    # In the tracked df, add 1 to particle to avoid 0 labels
    tracked["particle"] = tracked["particle"].astype(int) + 1

    tracked = break_tracks(tracked, max_length=max_height, area_column="area")

    masks_3d = np.zeros_like(masks, dtype=np.uint32)

    for _, row in tracked.iterrows():
        z = int(row["z"])
        particle_id = int(row["particle"])  # trackpy assigns unique IDs
        original_label = int(row["label_2d"])

        # Copy the original mask pixels and relabel with particle_id
        masks_3d[z][masks[z] == original_label] = particle_id

    # Make a pandas data frame from the data
    stats_df = get_mask_stats(masks_3d)
    stats_df = stats_df[(stats_df["height_pixel"] != 1)]
    stats_df = compute_knn_features(
        stats_df,
        k=1,
        position_columns=["x_center", "y_center", "z_center"],
        get_phenotype_score=False,
    )
    stats_df["aspect_ratio_score"] = (
        stats_df["aspect_ratio"] - stats_df["aspect_ratio"].min()
    ) / (stats_df["aspect_ratio"].max() - stats_df["aspect_ratio"].min())
    stats_df["density_normalized"] = 1 - (
        stats_df["mean_knn_distance"] - stats_df["mean_knn_distance"].min()
    ) / (stats_df["mean_knn_distance"].max() - stats_df["mean_knn_distance"].min())
    stats_df["background_score"] = (1 - stats_df["aspect_ratio_score"]) * (
        1 - stats_df["density_normalized"]
    )

    # Optionally normalize to 0-1 range
    stats_df["background_score_normalized"] = (
        stats_df["background_score"] - stats_df["background_score"].min()
    ) / (stats_df["background_score"].max() - stats_df["background_score"].min())
    filtered = stats_df  # [stats_df["aspect_ratio"] < 1.8]
    tracked_filtered = tracked[tracked["particle"].isin(filtered["particle"])]

    masks_3d_filtered = np.zeros_like(masks, dtype=np.uint32)

    for _, row in tracked_filtered.iterrows():
        z = int(row["z"])
        particle_id = int(row["particle"])
        original_label = int(row["label_2d"])

        # Copy the original mask pixels and relabel with new particle_id
        masks_3d_filtered[z][masks[z] == original_label] = particle_id

    return masks_3d, masks_3d_filtered, stats_df


def get_mask_stats(mask):
    particle_stats = []
    props = regionprops(mask)

    # For every mask found, get the label, centeroid, boundingbox, and volume
    for prop in props:
        label = prop.label
        centroid = prop.centroid  # (z, y, x)
        bounding_box = prop.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)
        volume = prop.area

        width_x = bounding_box[5] - bounding_box[2]
        width_y = bounding_box[4] - bounding_box[1]
        height_pixel = bounding_box[3] - bounding_box[0]
        height = (bounding_box[3] - bounding_box[0]) * 8.125
        aspect_ratio = height / np.mean([width_x, width_y])

        particle_stats.append(
            {
                "particle": label,
                "z_center": centroid[0],
                "y_center": centroid[1],
                "x_center": centroid[2],
                "bounding_box": bounding_box,
                "width_x": width_x,
                "width_y": width_y,
                "height_pixel": height_pixel,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "volume": volume,
            }
        )
    stats_df = pd.DataFrame(particle_stats)
    return stats_df


def break_tracks(tracked, max_length=10, area_column="area"):
    new_tracked = tracked.copy()
    next_particle_id = tracked["particle"].max() + 1

    while True:
        long_tracks = new_tracked.groupby("particle").size()
        long_tracks = long_tracks[long_tracks > max_length]

        if len(long_tracks) == 0:
            break  # No more long tracks

        for particle_id in long_tracks.index:
            particle_data = new_tracked[
                new_tracked["particle"] == particle_id
            ].sort_values("z")

            # Find weakest link
            areas = particle_data[area_column].values
            deltas = []
            for i in range(len(areas[:-1])):
                area1 = areas[i]
                area2 = areas[i + 1]
                delta = abs(area2 - area1)
                pct = delta / max(area1, area2)
                deltas.append(pct)

            weakest_idx = np.argmax(deltas)
            break_z = particle_data.iloc[weakest_idx + 1]["z"]

            # Split track
            mask = (new_tracked["particle"] == particle_id) & (
                new_tracked["z"] >= break_z
            )
            new_tracked.loc[mask, "particle"] = next_particle_id
            next_particle_id += 1

            # print(f"Split particle {particle_id} between z={break_z-1} and z={break_z}")

    return new_tracked
