from skimage.measure import regionprops, regionprops_table
import numpy as np
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
    shape_2d: tuple | None = None

    def __post_init__(self):
        self._intensities = {}


def get_cell_properties_2d(image):
    # Vectorized alternative to regionprops for label, centroid, volume, coords_set
    nonzero = image != 0
    if not np.any(nonzero):
        return []

    z, y, x = np.nonzero(nonzero)
    labels = image[z, y, x]
    order = np.argsort(labels)
    labels_sorted = labels[order]
    z_sorted = z[order]
    y_sorted = y[order]
    x_sorted = x[order]

    unique_labels, idx, counts = np.unique(
        labels_sorted, return_index=True, return_counts=True
    )
    z_sum = np.add.reduceat(z_sorted, idx)
    y_sum = np.add.reduceat(y_sorted, idx)
    x_sum = np.add.reduceat(x_sorted, idx)
    centroids = np.vstack((z_sum / counts, y_sum / counts, x_sum / counts)).T

    height, width = image.shape[1], image.shape[2]
    organoid = []
    for label, start, count, centroid in zip(unique_labels, idx, counts, centroids):
        y_part = y_sorted[start : start + count]
        x_part = x_sorted[start : start + count]
        flat_indices = np.ravel_multi_index((y_part, x_part), (height, width))
        coords_set = set(flat_indices.tolist())
        cell = cell_2d(
            label=int(label),
            centroid=tuple(centroid),
            coords_set=coords_set,
            volume=int(count),
            shape_2d=(height, width),
        )
        organoid.append(cell)

    return organoid


def properties_channel(organoid, mask, image, channel_type=None):
    # print(
    #     f"Calculating intensity properties for channel '{channel_type}'..."
    #     if channel_type
    #     else "Calculating intensity properties for image..."
    # )

    try:  # SCIkit changed the name of this property at some point, so we try both just in case
        props_channel = regionprops_table(
            mask, intensity_image=image, properties=["label", "intensity_mean"]
        )
        intensity_dict = dict(
            zip(props_channel["label"], props_channel["intensity_mean"])
        )
    except KeyError:
        props_channel = regionprops_table(
            mask, intensity_image=image, properties=["label", "mean_intensity"]
        )
        intensity_dict = dict(
            zip(props_channel["label"], props_channel["mean_intensity"])
        )

    for cell in organoid:
        attr_name = f"intensity_{channel_type}" if channel_type else "intensity"
        setattr(cell, attr_name, intensity_dict.get(cell.label, 0))

    # print(
    #     f"Calculated intensity properties for channel '{channel_type}'."
    #     if channel_type
    #     else "Calculated intensity properties for image."
    # )

    return organoid


def make_unique_mask(mask):
    """Makes a fresh mask from a already stitched mask"""
    print("Creating unique mask for properties calculation...")

    mask_uint = mask.astype(np.uint32, copy=False)
    z_slices = mask_uint.shape[0]
    slice_max = mask_uint.reshape(z_slices, -1).max(axis=1).astype(np.uint32)
    offsets = np.concatenate(([0], np.cumsum(slice_max[:-1], dtype=np.uint32)))
    unique_mask = mask_uint.copy()

    for z, offset in enumerate(offsets):
        if offset == 0:
            continue
        slice_data = unique_mask[z]
        nonzero = slice_data != 0
        if np.any(nonzero):
            slice_data[nonzero] = slice_data[nonzero] + offset

    label_counts = [np.unique(mask_uint[z]).size - 1 for z in range(z_slices)]
    total_unique = int(np.sum(label_counts))
    print(f"Unique mask created with {total_unique} unique cells.")

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

    print(f"Calculated 2D (intensity) properties for all cells")

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
    shape_2d = organoid_2d[0].shape_2d
    if shape_2d is None:
        raise ValueError("shape_2d missing; ensure get_cell_properties_2d sets it.")

    # group by integer z to increase speed
    z_groups = {}
    for idx, z_val in enumerate(z_array):
        z_int = int(round(z_val))
        z_groups.setdefault(z_int, []).append(idx)

    # build per-slice KDTree for nearest neighbor queries
    z_trees = {}
    z_indices = {}
    for z_int, indices in z_groups.items():
        coords = np.column_stack((x_array[indices], y_array[indices]))
        z_trees[z_int] = scipy.spatial.cKDTree(coords)
        z_indices[z_int] = np.array(indices, dtype=np.int64)

    labels_3d = np.full(n, -1, dtype=np.int32)

    def stitch_forward(idx, current_label):
        labels_3d[idx] = current_label
        next_z = int(round(z_array[idx])) + 1
        if next_z not in z_groups:
            return
        tree = z_trees[next_z]
        idxs = z_indices[next_z]
        if idxs.size == 0:
            return
        point = (x_array[idx], y_array[idx])
        cand_pos = tree.query_ball_point(point, r=distance_threshold)
        if not cand_pos:
            return
        cand_idx = idxs[np.array(cand_pos, dtype=np.int64)]
        cand_idx = cand_idx[labels_3d[cand_idx] == -1]
        if cand_idx.size == 0:
            return
        dists = np.hypot(
            x_array[cand_idx] - x_array[idx], y_array[cand_idx] - y_array[idx]
        )
        closest_cell = np.argmin(dists)
        cand_idx = int(cand_idx[closest_cell])
        if calculate_iou(coords_sets, idx, cand_idx) > iou_threshold:
            stitch_forward(cand_idx, current_label)

    current_label = 0
    for i in range(n):
        if labels_3d[i] == -1:
            current_label += 1
            stitch_forward(i, current_label)

    # build organoid_3d list (skip 3D coords to avoid heavy conversions)
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

        cell3d = cell_3d(
            label=cell,
            centroid=np.mean(centroids_2d, axis=0),
            coords_set=set(),
            volume=sum(volumes_2d),
            labels_2d=labels_2d,
            centroids_2d=centroids_2d,
            coords_set_2d=coords_set_2d,
            volumes_2d=volumes_2d,
            intensities_2d=intensities_2d,
        )
        cell3d.shape_2d = shape_2d
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
    z_arr = np.asarray(z_positions)
    line_x, line_y = line_function(z_arr)
    return np.sqrt(
        (np.asarray(x_positions) - line_x) ** 2
        + (np.asarray(y_positions) - line_y) ** 2
    )


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


def get_break_scores(organoid, skip_labels=None):
    skip_labels = skip_labels or set()
    for cell in organoid:
        if cell.label in skip_labels and cell.break_scores is not None:
            continue
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
            final_scores.append(final_score)

        cell.break_scores = final_scores

    return organoid


def split_cell_at_index(cell, break_position, new_label):
    # Build parent lineage: existing parents + current label
    parent_lineage = (cell.parents or []) + [cell.label]

    bottom_cell = cell_3d(
        label=cell.label,
        centroid=np.mean(cell.centroids_2d[:break_position], axis=0),
        coords_set=set(),
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

    max_z = len(cell.volumes_2d)
    top_cell = cell_3d(
        label=new_label,
        centroid=np.mean(cell.centroids_2d[break_position:max_z], axis=0),
        coords_set=set(),
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

    bottom_cell.shape_2d = getattr(cell, "shape_2d", None)
    top_cell.shape_2d = getattr(cell, "shape_2d", None)
    return bottom_cell, top_cell


def break_stitching(organoid, breaking_threshold=2.5):
    new_organoid = []
    breaks_made = 0
    broken_labels = set()
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
        broken_labels.add(cell.label)
        broken_labels.add(new_label)
        new_label += 1
        breaks_made += 1
    return new_organoid, breaks_made, broken_labels


def find_touching_pairs(organoid):
    # Build spatial index using KDTree for fast nearest neighbor search
    centroids = np.array([cell.centroid for cell in organoid])
    tree = scipy.spatial.cKDTree(centroids)

    z_to_coords_list = []
    z_keys_list = []
    for cell in organoid:
        z_to_coords = {}
        if cell.centroids_2d and cell.coords_set_2d:
            for idx, centroid in enumerate(cell.centroids_2d):
                z_int = int(round(centroid[0]))
                z_to_coords[z_int] = cell.coords_set_2d[idx]
        z_to_coords_list.append(z_to_coords)
        z_keys_list.append(set(z_to_coords.keys()))

    for i, cell in enumerate(organoid):
        if len(cell.volumes_2d) <= 1:
            cell.touching_cells = []
            continue

        _, indices = tree.query(cell.centroid, k=min(5, len(organoid)))
        candidate_indices = indices[1:]

        touching_cells = []
        z_to_coords = z_to_coords_list[i]
        z_keys = z_keys_list[i]
        if not z_to_coords:
            cell.touching_cells = []
            continue

        for idx in candidate_indices:
            other_z_to_coords = z_to_coords_list[idx]
            if not other_z_to_coords:
                continue

            is_touching = False
            for z in z_keys:
                coords = z_to_coords.get(z)
                if not coords:
                    continue
                coords_other = other_z_to_coords.get(z - 1)
                if coords_other and (coords & coords_other):
                    is_touching = True
                    break
                coords_other = other_z_to_coords.get(z + 1)
                if coords_other and (coords & coords_other):
                    is_touching = True
                    break

            if is_touching:
                touching_cells.append(organoid[idx].label)
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

    # Check if predicted points are in cell_1's 2D coords at this z
    coords_at_z = cell_1.coords_set_2d[z_idx]
    predicted_line_1_in_cell_1 = (
        int(round(predict_y_1)),
        int(round(predict_x_1)),
    ) in coords_at_z
    predicted_line_2_in_cell_1 = (
        int(round(predict_y_2)),
        int(round(predict_x_2)),
    ) in coords_at_z

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

    cell_1.volumes_2d[z_idx] = len(part_1_coords_2d)
    part_1_array = np.array(list(part_1_coords_2d))
    cell_1.centroids_2d[z_idx] = (last_z, *np.mean(part_1_array, axis=0))

    # Append to cell_2
    cell_2.coords_set_2d.append(part_2_coords_2d)
    cell_2.labels_2d.append(cell_1.labels_2d[z_idx])

    part_2_array = np.array(list(part_2_coords_2d))
    cell_2.centroids_2d.append((last_z, *np.mean(part_2_array, axis=0)))
    cell_2.volumes_2d.append(len(part_2_coords_2d))

    _recalculate_cell_properties(cell_1)
    _recalculate_cell_properties(cell_2)


def update_coords_bottom(
    cell_2, cell_1, z_idx, part_1_coords_2d, part_2_coords_2d, first_z
):
    """Update both cells after splitting cell_2 at its bottom"""
    cell_2.coords_set_2d[z_idx] = part_1_coords_2d

    cell_2.volumes_2d[z_idx] = len(part_1_coords_2d)
    part_1_array = np.array(list(part_1_coords_2d))
    cell_2.centroids_2d[z_idx] = (first_z, *np.mean(part_1_array, axis=0))

    # Insert at beginning of cell_1
    cell_1.coords_set_2d.insert(0, part_2_coords_2d)
    cell_1.labels_2d.insert(0, cell_2.labels_2d[z_idx])

    part_2_array = np.array(list(part_2_coords_2d))
    cell_1.centroids_2d.insert(0, (first_z, *np.mean(part_2_array, axis=0)))
    cell_1.volumes_2d.insert(0, len(part_2_coords_2d))

    _recalculate_cell_properties(cell_1)
    _recalculate_cell_properties(cell_2)


def _recalculate_cell_properties(cell):
    """Helper to recalculate volume and centroid after modification"""
    cell.volume = sum(cell.volumes_2d)
    cell.centroid = np.mean(cell.centroids_2d, axis=0)


import copy


def split_organoid_cells(organoid):
    # Convert flat-index coords_set_2d to (y, x) tuples if needed
    for cell in organoid:
        if not cell.coords_set_2d:
            continue
        first_set = next((s for s in cell.coords_set_2d if s), None)
        if not first_set:
            continue
        sample = next(iter(first_set))
        if isinstance(sample, (int, np.integer)):
            shape_2d = getattr(cell, "shape_2d", None)
            if shape_2d is None:
                raise ValueError(
                    f"Cell {cell.label} has flat-index coords_set_2d but no shape_2d for conversion"
                )
            width = shape_2d[1]
            cell.coords_set_2d = [
                (
                    set(zip(*np.divmod(np.fromiter(s, dtype=np.int64), width)))
                    if s
                    else set()
                )
                for s in cell.coords_set_2d
            ]

    print("Finding touching cell pairs...")
    organoid = find_touching_pairs(organoid)

    touching_cells = find_touching_cells(organoid)
    print(f"Found {len(touching_cells)} pairs of touching cells.")

    number_of_splits = 0
    cell_map = {c.label: c for c in organoid}
    count = 0
    for pair in touching_cells:
        count += 1
        cell_1 = cell_map.get(pair[0])
        cell_2 = cell_map.get(pair[1])
        if cell_1 is None or cell_2 is None:
            continue
        # Ensure cell_1 is below cell_2 (lower z centroid)
        if cell_1.centroid[0] > cell_2.centroid[0]:
            cell_1, cell_2 = cell_2, cell_1
        number_of_splits += split_multiple_cell_layers(cell_1, cell_2)

    print(f"Splitted {number_of_splits} touching cell pairs.")

    return organoid


def stitch_small_cells(organoid, max_slices=2, iou_threshold=0.9, knn=5):
    """Merge small cells (1-2 slices) with neighboring cells above or below"""
    from scipy.spatial import KDTree

    print(f"Stitching cells with {max_slices} or fewer slices...")

    # z->slice-index map per cell for O(1) slice lookup
    z_to_idx_maps = [
        {int(round(c[0])): si for si, c in enumerate(cell.centroids_2d)}
        for cell in organoid
    ]

    # z-group index with sets: O(1) add/discard for index maintenance after merges
    z_to_cell_indices = {}
    for i, z_map in enumerate(z_to_idx_maps):
        for z in z_map:
            z_to_cell_indices.setdefault(z, set()).add(i)

    # Index-based large-cell set avoids repeated len(volumes_2d) checks in inner loop
    large_cell_indices = {
        i for i, cell in enumerate(organoid) if len(cell.volumes_2d) > max_slices
    }

    # Per-z KDTree over large-cell 2D (y, x) centroids for fast spatial lookup
    z_kdtrees = {}
    z_kd_idx = {}
    for z, cell_set in z_to_cell_indices.items():
        large_at_z = [j for j in cell_set if j in large_cell_indices]
        if not large_at_z:
            continue
        pts = [
            (
                organoid[j].centroids_2d[z_to_idx_maps[j][z]][1],
                organoid[j].centroids_2d[z_to_idx_maps[j][z]][2],
            )
            for j in large_at_z
        ]
        z_kdtrees[z] = KDTree(pts)
        z_kd_idx[z] = large_at_z

    cells_to_remove = set()
    number_of_cells_stitched = 0
    number_of_cells_removed = 0

    for i, cell in enumerate(organoid):
        if i in large_cell_indices or cell.label in cells_to_remove:
            continue

        merged_into = None
        for z_pos, cell_si in z_to_idx_maps[i].items():
            s1 = cell.coords_set_2d[cell_si]
            len1 = len(s1)
            if len1 == 0:
                continue
            cy = cell.centroids_2d[cell_si][1]
            cx = cell.centroids_2d[cell_si][2]
            for neighbor_z in (z_pos - 1, z_pos + 1):
                kd = z_kdtrees.get(neighbor_z)
                if kd is None:
                    continue
                cands = z_kd_idx[neighbor_z]
                k_actual = min(knn, len(cands))
                _, nn_idxs = kd.query((cy, cx), k=k_actual)
                for nn_i in np.atleast_1d(nn_idxs):
                    j = cands[nn_i]
                    s2 = organoid[j].coords_set_2d[z_to_idx_maps[j][neighbor_z]]
                    len2 = len(s2)
                    if len2 == 0:
                        continue
                    intersection = len(s1 & s2)
                    if intersection == 0:
                        continue
                    min_len = min(len1, len2)
                    iou = (
                        1.0
                        if intersection == min_len
                        else intersection / (len1 + len2 - intersection)
                    )
                    if iou > iou_threshold:
                        merged_into = j
                        break
                if merged_into is not None:
                    break
            if merged_into is not None:
                break

        if merged_into is None:
            if len(cell.volumes_2d) == 1:
                cells_to_remove.add(cell.label)
                number_of_cells_removed += 1
            continue

        number_of_cells_stitched += 1
        closest_neighbor = organoid[merged_into]
        neighbor_z_list = [int(round(c[0])) for c in closest_neighbor.centroids_2d]

        for si in range(len(cell.coords_set_2d)):
            cell_z = int(round(cell.centroids_2d[si][0]))
            insert_idx = next(
                (k for k, nz in enumerate(neighbor_z_list) if cell_z < nz),
                len(neighbor_z_list),
            )
            closest_neighbor.coords_set_2d.insert(insert_idx, cell.coords_set_2d[si])
            closest_neighbor.centroids_2d.insert(insert_idx, cell.centroids_2d[si])
            closest_neighbor.volumes_2d.insert(insert_idx, cell.volumes_2d[si])
            neighbor_z_list.insert(insert_idx, cell_z)

        # Rebuild z_to_idx_maps for merged-into cell (indices shifted after inserts)
        z_to_idx_maps[merged_into] = {
            int(round(c[0])): k for k, c in enumerate(closest_neighbor.centroids_2d)
        }
        # Transfer small cell z-slots in the group index to the merged-into cell
        for z in z_to_idx_maps[i]:
            z_group = z_to_cell_indices.get(z)
            if z_group is not None:
                z_group.discard(i)
                z_group.add(merged_into)

        _recalculate_cell_properties(closest_neighbor)
        cells_to_remove.add(cell.label)

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
    breaking_threshold=2,
):
    print("Starting 3D stitching process...")
    organoid = get_2d_mask_properties(
        mask,
        image1,
        image_type_1=image_type_1,
        image2=image2,
        image_type_2=image_type_2,
    )
    organoid = IOU_stitching(organoid, distance_threshold=10, iou_threshold=0.3)

    # Iterative break and restitching
    max_iterations = 10
    iteration = 0
    checked_labels = set()
    print("Starting iterative breaking of cells")
    while iteration < max_iterations:
        organoid = get_break_scores(organoid, skip_labels=checked_labels)
        checked_labels = {cell.label for cell in organoid}
        organoid, breaks_made, broken_labels = break_stitching(
            organoid, breaking_threshold=breaking_threshold
        )
        checked_labels -= broken_labels
        if breaks_made == 0:
            break  # No more breaks made

        print(f"    Iteration {iteration + 1}: Made {breaks_made} breaks.")
        iteration += 1

    organoid = split_organoid_cells(organoid)

    organoid = stitch_small_cells(organoid)

    stitched_mask = mask_from_organoid(organoid, shape=mask.shape)

    print(f"Final stitched mask has {len(organoid)} cells.")

    return stitched_mask
