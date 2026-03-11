import tifffile
import pandas as pd
from skimage.measure import regionprops, regionprops_table
import napari
import numpy as np
import os
from dataclasses import dataclass
import scipy.spatial


@dataclass
class cell_3d:
    label: int
    centroid: tuple
    coords_set: set
    volume: int
    labels_2d: list | None = None
    centroids_2d: list | None = None
    coords_set_2d: list | None = None
    volumes_2d: list | None = None
    # per-slice per-channel intensities, dict of lists keyed by channel name
    intensities_2d: dict | None = None
    # aggregated per-channel intensities (mean across slices), keys like "intensity_wt"
    intensities: dict | None = None
    break_scores: list | None = None
    parents: list | None = None
    touching_cells: list | None = None


@dataclass
class cell_2d:
    label: int
    centroid: tuple
    coords_set: set
    volume: int

    def __post_init__(self):
        self._intensities = {}


def get_cell_properties_2d(image):
    # Get the properties of every mask in the image
    props = regionprops(image)

    organoid = []

    # For every mask found, get the label, centeroid, boundingbox, and volume
    for prop in props:
        # Pre-compute coordinate set for fast IOU calculations later
        coords_2d = prop.coords[:, 1:]
        coords_set = set(map(tuple, coords_2d))

        cell = cell_2d(
            label=prop.label,
            centroid=prop.centroid,  # (z, y, x)
            coords_set=coords_set,
            volume=prop.area,
        )
        organoid.append(cell)

    return organoid


def properties_channel(organoid, mask, image, channel_type=None):
    props_channel = regionprops_table(
        mask, intensity_image=image, properties=["label", "mean_intensity"]
    )
    intensity_dict = dict(zip(props_channel["label"], props_channel["mean_intensity"]))

    for cell in organoid:
        attr_name = f"intensity_{channel_type}" if channel_type else "intensity"
        setattr(cell, attr_name, intensity_dict.get(cell.label, 0))

    return organoid


def make_unique_mask(mask):
    """Makes a fresh mask from a already stitched mask"""
    unique_mask = []
    number = 1
    for z in range(mask.shape[0]):
        # go over every unique number in the mask and create a new slice with a new unique number 1 higher
        slice = mask[z]
        unique_slice = mask[z].copy()
        for val in range(1, slice.max() + 1):
            unique_slice[slice == val] = number
            number += 1
        unique_mask.append(unique_slice)
    unique_mask = np.stack(unique_mask, axis=0)

    return unique_mask


def get_2d_mask_properties(
    mask, image1, image_type_1=None, image2=None, image_type_2=None
):
    unique_mask = make_unique_mask(mask)
    organoid = get_cell_properties_2d(unique_mask)

    organoid = properties_channel(
        organoid, unique_mask, image1, channel_type=image_type_1
    )

    if image2 is not None:
        organoid = properties_channel(
            organoid, unique_mask, image2, channel_type=image_type_2
        )

    return organoid


def calculate_iou(coords_sets, idx1, idx2):
    """
    Calculate overlap metric for cell matching.
    Returns 1 if the smaller cell is completely contained in the larger cell.
    Otherwise returns intersection over union (standard IOU).
    """
    set1 = coords_sets[idx1]
    set2 = coords_sets[idx2]

    # Check if either set is empty
    len1, len2 = len(set1), len(set2)
    if len2 == 0:
        return 0

    # Calculate intersection (overlapping pixels)
    intersection = len(set1 & set2)

    if intersection == 0:
        return 0  # No overlap = different cells

    # If the smaller cell is completely contained in the larger cell, return 1
    min_len = min(len1, len2)
    if intersection == min_len:
        return 1.0  # One cell completely covers the other

    # Otherwise, use standard IOU
    union = len1 + len2 - intersection
    return intersection / union


def IOU_stitching(organoid_2d, distance_threshold=10, iou_threshold=0.3):
    print("Starting IOU based 3D stitching...")
    n = len(organoid_2d)
    if n == 0:
        return []
    z_array = np.array([c.centroid[0] for c in organoid_2d])
    y_array = np.array([c.centroid[1] for c in organoid_2d])
    x_array = np.array([c.centroid[2] for c in organoid_2d])
    coords_sets = {i: c.coords_set for i, c in enumerate(organoid_2d)}

    # group by integer z to increase speed
    z_groups = {}
    for idx, z_val in enumerate(z_array):
        z_int = int(round(z_val))
        z_groups.setdefault(z_int, []).append(idx)

    labels_3d = np.full(n, -1, dtype=np.int32)

    def stitch_forward(idx, current_label):
        labels_3d[idx] = current_label
        next_z = int(round(z_array[idx])) + 1
        if next_z not in z_groups:
            return
        # get all cells that havent been stitched yet
        candidates = [c for c in z_groups[next_z] if labels_3d[c] == -1]
        if not candidates:
            return
        cand_arr = np.array(candidates)
        dists = np.hypot(
            x_array[cand_arr] - x_array[idx], y_array[cand_arr] - y_array[idx]
        )
        closest_cell = np.argmin(dists)
        if dists[closest_cell] > distance_threshold:
            return
        cand_idx = cand_arr[closest_cell]
        if calculate_iou(coords_sets, idx, cand_idx) > iou_threshold:
            stitch_forward(cand_idx, current_label)

    current_label = 0
    for i in range(n):
        if labels_3d[i] == -1:
            current_label += 1
            stitch_forward(i, current_label)

    # build organoid_3d list
    organoid_3d = []
    intensity_attrs = [a for a in organoid_2d[0].__dict__ if a.startswith("intensity")]
    for cell in np.unique(labels_3d):
        if cell <= 0:
            continue
        slice_idxs = np.where(labels_3d == cell)[0]
        labels_2d = [organoid_2d[i].label for i in slice_idxs]
        centroids_2d = [organoid_2d[i].centroid for i in slice_idxs]
        coords_set_2d = [organoid_2d[i].coords_set for i in slice_idxs]
        volumes_2d = [organoid_2d[i].volume for i in slice_idxs]
        # collect per-channel per-slice intensities
        intensities_2d = {
            attr: [getattr(organoid_2d[i], attr, None) for i in slice_idxs]
            for attr in intensity_attrs
        }

        coords_3d = set()
        for z_idx, (z, _, _) in enumerate(centroids_2d):
            z_int = int(round(z))
            for y, x in coords_set_2d[z_idx]:
                coords_3d.add((z_int, int(y), int(x)))

        cell3d = cell_3d(
            label=cell,
            centroid=np.mean(centroids_2d, axis=0),
            coords_set=coords_3d,
            volume=sum(volumes_2d),
            labels_2d=labels_2d,
            centroids_2d=centroids_2d,
            coords_set_2d=coords_set_2d,
            volumes_2d=volumes_2d,
            intensities_2d=intensities_2d,
        )
        organoid_3d.append(cell3d)

    print(f"Completed IOU based stitching: {len(organoid_3d)} cells formed.")

    return organoid_3d


def mask_from_organoid(organoid, shape):
    mask = np.zeros(shape, dtype=np.uint16)
    for cell in organoid:
        for z_idx, (z, y, x) in enumerate(cell.centroids_2d):
            z_int = int(round(z))
            if 0 <= z_int < shape[0]:
                for yy, xx in cell.coords_set_2d[z_idx]:
                    if 0 <= yy < shape[1] and 0 <= xx < shape[2]:
                        mask[z_int, yy, xx] = cell.label
    return mask


def break_score(values, i):
    # Calculate intensity-based anomalies by getting a score that resembles high-low-high intensity value patterns
    # Calculate ratio between slice i and below i, and i and above i
    values_bottom = max(values[:i])
    values_top = max(values[i + 1 :])
    ratio_1 = values_bottom / values[i]
    ratio_2 = values[i] / values_top

    # divide this again, now high low high gets higher score then other patterns
    return ratio_1 / ratio_2


def get_centroid_positions(centroid):
    z_positions = [c[0] for c in centroid]
    y_positions = [c[1] for c in centroid]
    x_positions = [c[2] for c in centroid]
    return z_positions, y_positions, x_positions


def calculate_line_3d(z_positions, y_positions, x_positions, points=3):
    # Fit line to specified points (int, tuple, or slice)
    if isinstance(points, tuple):
        points = slice(*points)
    elif isinstance(points, int):
        points = slice(None, points)

    z_fit = np.array(z_positions[points])
    x_fit = np.array(x_positions[points])
    y_fit = np.array(y_positions[points])

    # Calculate line parameters
    A = np.vstack([z_fit, np.ones(len(z_fit))]).T
    m_x, c_x = np.linalg.lstsq(A, x_fit, rcond=None)[0]
    m_y, c_y = np.linalg.lstsq(A, y_fit, rcond=None)[0]

    # Create line function: given z, returns (x, y) in that order
    line_function = lambda z: (m_x * z + c_x, m_y * z + c_y)

    return line_function


def calculate_point_line_distances(
    z_positions, y_positions, x_positions, line_function
):
    distances = []
    for z, x, y in zip(z_positions, x_positions, y_positions):
        line_x, line_y = line_function(z)
        dist = np.sqrt((x - line_x) ** 2 + (y - line_y) ** 2)
        distances.append(dist)
    return distances


def get_slope_changes(distances):
    # Calculate slope changes in the distance array
    slopes = np.diff(distances)
    slope_changes = np.diff(slopes)

    # Pad to match original length
    slope_changes = np.concatenate([[0], slope_changes, [0]])
    return slope_changes


def get_volume_break_scores(cell):
    num_slices = len(cell.volumes_2d)
    volumes_break_score = [0] * num_slices

    for i in range(1, num_slices - 1):
        volumes_break_score[i] = break_score(cell.volumes_2d, i)

    return volumes_break_score


def get_intensity_break_scores(cell):
    num_slices = len(cell.volumes_2d)
    intensity_break_score = [0] * num_slices

    if cell.intensities_2d:
        per_channel = list(cell.intensities_2d.values())
        max_intensity = [max(vals) for vals in zip(*per_channel)]
    else:
        max_intensity = [0] * num_slices

    for i in range(1, num_slices - 1):
        intensity_break_score[i] = break_score(max_intensity, i)

    return intensity_break_score


def get_line_break_scores(z_positions, y_positions, x_positions):
    # Calculate line-based break scores (from bottom_up)
    line = calculate_line_3d(z_positions, y_positions, x_positions, points=3)
    distances = calculate_point_line_distances(
        z_positions, y_positions, x_positions, line
    )
    break_score = abs(get_slope_changes(distances))
    return break_score


def get_IOU_break_scores(cell):
    num_slices = len(cell.volumes_2d)
    iou_break_score = [0] * num_slices

    for i in range(1, num_slices - 1):
        iou_break_score[i] = 1 - calculate_iou(cell.coords_set_2d, i - 1, i)

    return iou_break_score


def get_break_scores(organoid):
    for cell in organoid:
        num_slices = len(cell.volumes_2d)
        if num_slices < 5:
            cell.break_scores = [0] * num_slices
            continue

        # Calculate volume-based break scores
        volumes_break_score = get_volume_break_scores(cell)
        # Calculate intensity-based break scores
        intensity_break_score = get_intensity_break_scores(cell)

        # Calculate line-based break scores (from bottom_up)
        z_positions, y_positions, x_positions = get_centroid_positions(
            cell.centroids_2d
        )
        break_score_up = get_line_break_scores(z_positions, y_positions, x_positions)
        # pad 1 at start and remove last to match and better represent breaks between slices
        break_score_up = np.concatenate([[0], break_score_up[:-1]])
        # Calculate line-based break scores (from top_down)
        break_score_down = get_line_break_scores(
            z_positions[::-1], y_positions[::-1], x_positions[::-1]
        )[::-1]
        IOU_break_score = get_IOU_break_scores(cell)

        # Calculate final break scores
        final_scores = []
        for v_score, i_score, up_score, down_score, IOU_score in zip(
            volumes_break_score,
            intensity_break_score,
            break_score_up,
            break_score_down,
            IOU_break_score,
        ):
            final_score = (
                np.max([up_score, down_score])
                * v_score
                * (i_score**2)
                * (IOU_score * 2)
            )
            # if cell.label == 83:
            #     print(f"v:{v_score:.2f}, i:{i_score:.2f}, up:{up_score:.2f}, down:{down_score:.2f}, IOU:{IOU_score:.2f}, final:{final_score:.2f}")
            final_scores.append(final_score)

        cell.break_scores = final_scores

    return organoid


def split_cell_at_index(cell, break_position, new_label):
    # Build parent lineage: existing parents + current label
    parent_lineage = (cell.parents or []) + [cell.label]

    # Build 3D coords_set for bottom cell
    coords_3d_bottom = set()
    for z_idx in range(break_position):
        z_int = int(round(cell.centroids_2d[z_idx][0]))
        for y, x in cell.coords_set_2d[z_idx]:
            coords_3d_bottom.add((z_int, int(y), int(x)))

    bottom_cell = cell_3d(
        label=cell.label,
        centroid=np.mean(cell.centroids_2d[:break_position], axis=0),
        coords_set=coords_3d_bottom,
        volume=sum(cell.volumes_2d[:break_position]),
        labels_2d=cell.labels_2d[:break_position],
        centroids_2d=cell.centroids_2d[:break_position],
        coords_set_2d=cell.coords_set_2d[:break_position],
        volumes_2d=cell.volumes_2d[:break_position],
        intensities_2d={
            k: v[:break_position] for k, v in (cell.intensities_2d or {}).items()
        },
        parents=parent_lineage,
    )

    # Build 3D coords_set for top cell
    max_z = len(cell.volumes_2d)
    coords_3d_top = set()
    for z_idx in range(break_position, max_z):
        z_int = int(round(cell.centroids_2d[z_idx][0]))
        for y, x in cell.coords_set_2d[z_idx]:
            coords_3d_top.add((z_int, int(y), int(x)))

    top_cell = cell_3d(
        label=new_label,
        centroid=np.mean(cell.centroids_2d[break_position:max_z], axis=0),
        coords_set=coords_3d_top,
        volume=sum(cell.volumes_2d[break_position:max_z]),
        labels_2d=cell.labels_2d[break_position:max_z],
        centroids_2d=cell.centroids_2d[break_position:max_z],
        coords_set_2d=cell.coords_set_2d[break_position:max_z],
        volumes_2d=cell.volumes_2d[break_position:max_z],
        intensities_2d={
            k: v[break_position:max_z] for k, v in (cell.intensities_2d or {}).items()
        },
        parents=parent_lineage,
    )

    return bottom_cell, top_cell


def break_stitching(organoid, breaking_threshold=2.5):
    new_organoid = []
    breaks_made = 0
    new_label = max(cell.label for cell in organoid) + 1
    for cell in organoid:
        if len(cell.volumes_2d) < 7:
            new_organoid.append(cell)
            continue
        # if cell.label == 103:
        #     print(cell.break_scores)
        # Find indices where break score exceeds threshold
        break_scores = [score for score in cell.break_scores]
        break_scores[:3] = [0, 0, 0]  # First 3 slices get score 0
        break_scores[-3:] = [0, 0, 0]  # Last 3 slices get score 0
        if max(break_scores) < breaking_threshold:
            new_organoid.append(cell)
            continue

        # break indice is the highest scoring indices above threshold
        max_break_value = max(break_scores)
        break_position = break_scores.index(max_break_value)

        # Create new cells by breaking at the identified indices
        bottom_cell, top_cell = split_cell_at_index(
            cell, break_position, new_label=new_label
        )
        new_organoid.append(bottom_cell)
        new_organoid.append(top_cell)
        new_label += 1
        breaks_made += 1
    return new_organoid, breaks_made


def find_touching_pairs(organoid):
    neighbor_offsets = [(-1, 0, 0), (1, 0, 0)]  # Only z-direction neighbors

    # Build spatial index using KDTree for fast nearest neighbor search
    centroids = np.array([cell.centroid for cell in organoid])
    tree = scipy.spatial.cKDTree(centroids)

    for i, cell in enumerate(organoid):
        if len(cell.volumes_2d) <= 1:
            cell.touching_cells = []
            continue

        # Query k=10 nearest neighbors (increase if needed, but 5-10 is usually enough)
        distances, indices = tree.query(cell.centroid, k=min(5, len(organoid)))

        # Skip self (first result is always self)
        candidate_indices = indices[1:]

        # Check only nearby candidates for actual touching
        touching_cells = []
        current_boundary = cell.coords_set

        for idx in candidate_indices:
            other_cell = organoid[idx]
            other_coords = other_cell.coords_set

            # Fast boundary check using set operations
            # Create neighbor coordinates for current cell
            neighbor_coords = {
                (z + dz, y, x)
                for z, y, x in current_boundary
                for dz, _, _ in neighbor_offsets
            }

            # Check if any neighbors overlap with other cell
            if neighbor_coords & other_coords:  # Set intersection
                touching_cells.append(other_cell.label)

                # Early exit if we found enough touching cells
                if len(touching_cells) >= 5:
                    break

        cell.touching_cells = touching_cells

    return organoid


def find_touching_cells(organoid):
    # return a list of unique pairs of touching cells
    touching_pairs = set()
    for cell in organoid:
        for touching_label in cell.touching_cells:
            pair = tuple(sorted((cell.label, touching_label)))
            touching_pairs.add(pair)

    return list(touching_pairs)


def calculate_split_coords(cell, index, pred_point_1, pred_point_2):
    # Create splitting line perpendicular to the line between the two predicted points
    # Direction vector between predicted points
    direction = pred_point_2 - pred_point_1
    # Perpendicular vector (rotate 90 degrees)
    normal = direction / (np.linalg.norm(direction) + 1e-8)
    # Midpoint between predicted points
    midpoint = (pred_point_1 + pred_point_2) / 2

    part_1_coords = []
    part_2_coords = []

    for coord in cell.coords_set_2d[index]:
        y, x = coord
        point = np.array([y, x])
        # Vector from midpoint to this point
        to_point = point - midpoint
        # Dot product determines which side of the line
        side = np.dot(to_point, normal)

        if side <= 0:
            part_1_coords.append(coord)
        else:
            part_2_coords.append(coord)

    # Calculate new centroids
    part_1_coords = np.array(part_1_coords)
    part_2_coords = np.array(part_2_coords)

    return part_1_coords, part_2_coords


def are_split_cells_valid(cell, index, part_1_coords, part_2_coords):
    min_size = max(
        3, len(cell.coords_set_2d[index]) * 0.1
    )  # At least 10% of pixels or 3 pixels

    if len(part_1_coords) < min_size or len(part_2_coords) < min_size:
        # print(
        #     f"  Warning: Split too unbalanced ({len(part_1_coords)} vs {len(part_2_coords)} pixels)"
        # )
        return False
    return True


def calculate_new_distances(part_1_coords, part_2_coords, pred_point_1, pred_point_2):
    centroid_1 = np.mean(part_1_coords, axis=0)  # (y, x)
    centroid_2 = np.mean(part_2_coords, axis=0)  # (y, x)

    # Calculate distances from new centroids to predicted points
    dist_c1_to_pred1 = np.sqrt(
        (centroid_1[0] - pred_point_1[0]) ** 2 + (centroid_1[1] - pred_point_1[1]) ** 2
    )
    dist_c2_to_pred2 = np.sqrt(
        (centroid_2[0] - pred_point_2[0]) ** 2 + (centroid_2[1] - pred_point_2[1]) ** 2
    )
    # Check if splitting improves the fit
    avg_new_distance = (dist_c1_to_pred1 + dist_c2_to_pred2) / 2

    return avg_new_distance


def check_split_cells(cell, index, pred_point_1, pred_point_2):
    # centroids_2d stores (z, y, x), so extract y, x
    original_centroid = cell.centroids_2d[index]
    original_y = original_centroid[1]  # y is at index 1
    original_x = original_centroid[2]  # x is at index 2

    original_distance = np.sqrt(
        (original_y - pred_point_1[0]) ** 2 + (original_x - pred_point_1[1]) ** 2
    )

    part_1_coords, part_2_coords = calculate_split_coords(
        cell, index, pred_point_1, pred_point_2
    )

    if not are_split_cells_valid(cell, index, part_1_coords, part_2_coords):
        # print(f"    ✗ Split cells not valid")
        return None

    avg_new_distance = calculate_new_distances(
        part_1_coords, part_2_coords, pred_point_1, pred_point_2
    )

    if avg_new_distance < original_distance:
        # print(
        #     f"    ✓ Splitting improves fit! cell {cell.label} (reduction: {original_distance - avg_new_distance:.2f})"
        # )
        return (part_1_coords, part_2_coords)
    else:
        # print(f"    ✗ Splitting does not improve fit: increase of {avg_new_distance:.2f} >= {original_distance:.2f}")
        return None


def split_multiple_cell_layers(cell_1, cell_2):
    z_to_predicts = [1, 2]
    split_cells = 0
    for z_to_predict in z_to_predicts:
        # Split cell_1 at its top
        # print("bottom cell z:", z_to_predict)
        split_results = split_single_cell_layer(
            cell_1, cell_2, z_to_predict=z_to_predict, bottom_up=True
        )
        if split_results is not None:
            split_cells += 1
            part_1_coords, part_2_coords, z_idx, last_z = split_results
            part_1_coords_2d = set(map(tuple, part_1_coords))
            part_2_coords_2d = set(map(tuple, part_2_coords))
            update_coords(
                cell_1, cell_2, z_idx, part_1_coords_2d, part_2_coords_2d, last_z
            )

        # print("top cell z:", z_to_predict)
        # Split cell_2 at its bottom
        split_results = split_single_cell_layer(
            cell_2, cell_1, z_to_predict=z_to_predict, bottom_up=False
        )
        if split_results is not None:
            split_cells += 1
            part_1_coords, part_2_coords, z_idx, first_z = split_results
            part_1_coords_2d = set(map(tuple, part_1_coords))
            part_2_coords_2d = set(map(tuple, part_2_coords))
            update_coords_bottom(
                cell_2, cell_1, z_idx, part_1_coords_2d, part_2_coords_2d, first_z
            )

        # When no splits are made on the 1st level, we dont need to check splitting the second level.
        if split_cells == 0:
            break
    return split_cells


def split_single_cell_layer(cell_1, cell_2, z_to_predict=1, bottom_up=True):
    """
    Split a cell layer at the touching interface.
    bottom_up=True: split cell_1 at its top (for cell_1 splitting cell_2 from above)
    bottum_up=False: split cell_1 at its bottom (for cell_1 splitting from below)
    """
    if len(cell_1.volumes_2d) <= 2 or len(cell_2.volumes_2d) <= 2:
        return None

    z_positions_1, y_positions_1, x_positions_1 = get_centroid_positions(
        cell_1.centroids_2d
    )
    z_positions_2, y_positions_2, x_positions_2 = get_centroid_positions(
        cell_2.centroids_2d
    )

    # Get line fits
    if bottom_up:
        line_1 = calculate_line_3d(
            z_positions_1, y_positions_1, x_positions_1, points=(0, -z_to_predict)
        )
        line_2 = calculate_line_3d(
            z_positions_2[::-1],
            y_positions_2[::-1],
            x_positions_2[::-1],
            points=(0, -z_to_predict),
        )
        z_idx = len(cell_1.volumes_2d) - z_to_predict
        target_z = z_positions_1[-z_to_predict]
    else:
        line_1 = calculate_line_3d(
            z_positions_1[::-1],
            y_positions_1[::-1],
            x_positions_1[::-1],
            points=(0, -z_to_predict),
        )
        line_2 = calculate_line_3d(
            z_positions_2, y_positions_2, x_positions_2, points=(0, -z_to_predict)
        )
        z_idx = z_to_predict - 1
        target_z = z_positions_1[z_to_predict - 1]

    # Get predictions
    predict_x_1, predict_y_1 = line_1(target_z)
    predict_x_2, predict_y_2 = line_2(target_z)

    # Check if predicted points are in cell_1's coords_set
    predicted_line_1_in_cell_1 = (
        int(round(target_z)),
        int(round(predict_y_1)),
        int(round(predict_x_1)),
    ) in cell_1.coords_set
    predicted_line_2_in_cell_1 = (
        int(round(target_z)),
        int(round(predict_y_2)),
        int(round(predict_x_2)),
    ) in cell_1.coords_set

    # print(predicted_line_1_in_cell_1, predicted_line_2_in_cell_1)
    # print(int(round(predict_y_1*0.64)), int(round(predict_x_1*0.64)), int(round(predict_y_2*0.64)), int(round(predict_x_2*0.64)))
    if not (predicted_line_1_in_cell_1 and predicted_line_2_in_cell_1):
        # print(
        #     f"  ✗ Predicted splitting points not in cell {cell_1.label} at z={int(round(target_z))}"
        # )
        return None

    pred_point_1 = np.array([predict_y_1, predict_x_1])
    pred_point_2 = np.array([predict_y_2, predict_x_2])

    # print(f"Attempting to split touching cells {cell_1.label} and {cell_2.label}")
    split_results = check_split_cells(cell_1, z_idx, pred_point_1, pred_point_2)
    if split_results is None:
        return None

    part_1_coords, part_2_coords = split_results
    target_z_int = int(round(target_z))
    return part_1_coords, part_2_coords, z_idx, target_z_int


def update_coords(cell_1, cell_2, z_idx, part_1_coords_2d, part_2_coords_2d, last_z):
    """Update both cells after splitting cell_1 at its top"""
    cell_1.coords_set_2d[z_idx] = part_1_coords_2d

    for y, x in part_2_coords_2d:
        cell_1.coords_set.discard((last_z, int(y), int(x)))

    cell_1.volumes_2d[z_idx] = len(part_1_coords_2d)
    part_1_array = np.array(list(part_1_coords_2d))
    cell_1.centroids_2d[z_idx] = (last_z, *np.mean(part_1_array, axis=0))

    # Append to cell_2
    cell_2.coords_set_2d.append(part_2_coords_2d)
    cell_2.labels_2d.append(cell_1.labels_2d[z_idx])

    part_2_array = np.array(list(part_2_coords_2d))
    cell_2.centroids_2d.append((last_z, *np.mean(part_2_array, axis=0)))
    cell_2.volumes_2d.append(len(part_2_coords_2d))

    for y, x in part_2_coords_2d:
        cell_2.coords_set.add((last_z, int(y), int(x)))

    _recalculate_cell_properties(cell_1)
    _recalculate_cell_properties(cell_2)


def update_coords_bottom(
    cell_2, cell_1, z_idx, part_1_coords_2d, part_2_coords_2d, first_z
):
    """Update both cells after splitting cell_2 at its bottom"""
    cell_2.coords_set_2d[z_idx] = part_1_coords_2d

    for y, x in part_2_coords_2d:
        cell_2.coords_set.discard((first_z, int(y), int(x)))

    cell_2.volumes_2d[z_idx] = len(part_1_coords_2d)
    part_1_array = np.array(list(part_1_coords_2d))
    cell_2.centroids_2d[z_idx] = (first_z, *np.mean(part_1_array, axis=0))

    # Insert at beginning of cell_1
    cell_1.coords_set_2d.insert(0, part_2_coords_2d)
    cell_1.labels_2d.insert(0, cell_2.labels_2d[z_idx])

    part_2_array = np.array(list(part_2_coords_2d))
    cell_1.centroids_2d.insert(0, (first_z, *np.mean(part_2_array, axis=0)))
    cell_1.volumes_2d.insert(0, len(part_2_coords_2d))

    for y, x in part_2_coords_2d:
        cell_1.coords_set.add((first_z, int(y), int(x)))

    _recalculate_cell_properties(cell_1)
    _recalculate_cell_properties(cell_2)


def _recalculate_cell_properties(cell):
    """Helper to recalculate volume and centroid after modification"""
    cell.volume = sum(cell.volumes_2d)
    cell.centroid = np.mean(cell.centroids_2d, axis=0)


import copy


def split_organoid_cells(organoid):
    print("Starting to split touching cells...")
    organoid = find_touching_pairs(organoid)
    touching_cells = find_touching_cells(organoid)
    print(f"Found {len(touching_cells)} pairs of touching cells.")
    number_of_splits = 0
    for pair in touching_cells:
        cell_1 = next(c for c in organoid if c.label == pair[0])
        cell_2 = next(c for c in organoid if c.label == pair[1])
        # Ensure cell_1 is below cell_2 (lower z centroid)
        if cell_1.centroid[0] > cell_2.centroid[0]:
            cell_1, cell_2 = cell_2, cell_1
        number_of_splits += split_multiple_cell_layers(cell_1, cell_2)

    print(f"Splitted {number_of_splits} touching cell pairs.")
    return organoid


def organoid_from_mask(mask):
    """Reconstruct organoid list directly from mask by splitting coords by z-level"""
    props = regionprops(mask)

    organoid = []
    for prop in props:
        # Get 3D coordinates (z, y, x)
        coords_3d = prop.coords

        # Split coordinates by z-level
        coords_by_z = {}
        for z, y, x in coords_3d:
            if z not in coords_by_z:
                coords_by_z[z] = []
            coords_by_z[z].append((y, x))

        # Sort by z to maintain order
        z_sorted = sorted(coords_by_z.keys())

        # Build per-slice data
        centroids_2d = []
        coords_set_2d = []
        volumes_2d = []

        for z in z_sorted:
            coords_2d = np.array(coords_by_z[z])
            centroid_2d = np.mean(coords_2d, axis=0)

            centroids_2d.append((z, centroid_2d[0], centroid_2d[1]))
            coords_set_2d.append(set(map(tuple, coords_2d)))
            volumes_2d.append(len(coords_2d))

        cell = cell_3d(
            label=prop.label,
            centroid=prop.centroid,
            coords_set=set(map(tuple, prop.coords)),
            volume=prop.area,
            centroids_2d=centroids_2d,
            coords_set_2d=coords_set_2d,
            volumes_2d=volumes_2d,
        )

        organoid.append(cell)

    return organoid


def stitch_small_cells(organoid, max_slices=2, iou_threshold=0.9):
    """Merge small cells (1-2 slices) with neighboring cells above or below"""
    cells_to_remove = []
    number_of_cells_stitched = 0
    number_of_cells_removed = 0

    print(f"Stitching cells with {max_slices} or fewer slices...")

    for cell in organoid:
        if len(cell.volumes_2d) > max_slices:
            continue  # Only process cells with max_slices or fewer

        # Get the z positions of this cell's slices
        z_positions = [int(round(c[0])) for c in cell.centroids_2d]

        # Find neighboring cells in organoid that are above or below
        candidates = []
        for c in organoid:
            if c.label == cell.label or len(c.volumes_2d) <= max_slices:
                continue

            # Check if candidate cell is adjacent to any of the current cell's slices
            for z_pos in z_positions:
                # Check adjacent z-levels
                for neighbor_z in [z_pos - 1, z_pos + 1]:
                    # Get the slice index in the candidate cell at neighbor_z
                    candidate_z_positions = [
                        int(round(c_cent[0])) for c_cent in c.centroids_2d
                    ]

                    if neighbor_z in candidate_z_positions:
                        # Found an adjacent slice, now check IOU
                        candidate_slice_idx = candidate_z_positions.index(neighbor_z)
                        cell_slice_idx = z_positions.index(z_pos)

                        iou = calculate_iou(
                            {
                                0: cell.coords_set_2d[cell_slice_idx],
                                1: c.coords_set_2d[candidate_slice_idx],
                            },
                            0,
                            1,
                        )

                        if iou > iou_threshold:
                            candidates.append(c)
                            break  # Found a valid neighbor, no need to check other slices of this candidate

                if c in candidates:
                    break  # Already added this candidate

        if not candidates:
            # Only remove 1-slice cells, keep 2-slice cells
            if len(cell.volumes_2d) == 1:
                cells_to_remove.append(cell.label)
                number_of_cells_removed += 1
            continue

        number_of_cells_stitched += 1
        closest_neighbor = candidates[0]

        # Add all of cell's slices to neighbor
        for i in range(len(cell.coords_set_2d)):
            # Find correct insertion position to maintain z-order
            cell_z = int(round(cell.centroids_2d[i][0]))
            neighbor_z_positions = [
                int(round(c[0])) for c in closest_neighbor.centroids_2d
            ]

            # Find insertion index
            insert_idx = len(neighbor_z_positions)
            for idx, neighbor_z in enumerate(neighbor_z_positions):
                if cell_z < neighbor_z:
                    insert_idx = idx
                    break

            # Insert at correct position
            closest_neighbor.coords_set_2d.insert(insert_idx, cell.coords_set_2d[i])
            closest_neighbor.centroids_2d.insert(insert_idx, cell.centroids_2d[i])
            closest_neighbor.volumes_2d.insert(insert_idx, cell.volumes_2d[i])

            # Update 3D coords_set
            closest_neighbor.coords_set.update(
                {(cell_z, int(y), int(x)) for y, x in cell.coords_set_2d[i]}
            )

        # Update neighbor properties
        _recalculate_cell_properties(closest_neighbor)

        # Mark cell for removal
        cells_to_remove.append(cell.label)

    # Remove merged cells from organoid
    organoid = [c for c in organoid if c.label not in cells_to_remove]

    print(
        f"Stitched {number_of_cells_stitched} small cells, and removed {number_of_cells_removed} cells without suitable neighbors."
    )

    return organoid


import copy


def stitch_3d(
    mask,
    image1,
    image_type_1=None,
    image2=None,
    image_type_2=None,
    breaking_threshold=2.5,
):
    print("Starting 3D stitching process...")
    organoid = get_2d_mask_properties(
        mask,
        image1,
        image_type_1=image_type_1,
        image2=image2,
        image_type_2=image_type_2,
    )
    print(f"Initial organoid has {len(organoid)} 2d sliced cells.")
    organoid = IOU_stitching(organoid, distance_threshold=10, iou_threshold=0.3)

    # Iterative break and restitching
    max_iterations = 10
    iteration = 0
    print("Starting iterative breaking of cells")
    while iteration < max_iterations:
        organoid = get_break_scores(organoid)
        organoid, breaks_made = break_stitching(
            organoid, breaking_threshold=breaking_threshold
        )
        if breaks_made == 0:
            break  # No more breaks made

        print(f"    Iteration {iteration + 1}: Made {breaks_made} breaks.")
        iteration += 1

    # before_splitting = copy.deepcopy(organoid)
    # before_splitting_mask = mask_from_organoid(before_splitting, shape=mask.shape)
    organoid = split_organoid_cells(organoid)

    stitched_mask = mask_from_organoid(organoid, shape=mask.shape)
    organoid = organoid_from_mask(stitched_mask)

    organoid = stitch_small_cells(organoid)

    stitched_mask = mask_from_organoid(organoid, shape=mask.shape)

    print(f"Final stitched mask has {len(organoid)} cells.")

    return stitched_mask, organoid  # , before_splitting, before_splitting_mask
