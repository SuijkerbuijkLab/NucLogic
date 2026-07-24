from matplotlib import scale
from skimage.measure import regionprops, regionprops_table
import numpy as np
from dataclasses import dataclass
import scipy.spatial


@dataclass
class cell_3d:
    label: int
    centroid: tuple
    coords_set: np.ndarray
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
    # cached per-gap "1 - adjacent-slice IOU" array (index i = gap between slice
    # i-1 and i; index 0 is 0). Invariant under breaking, so it is sliced onto the
    # halves in split_cell_at_index instead of being recomputed each round.
    iou_gaps: list | None = None


@dataclass
class cell_2d:
    label: int
    centroid: tuple
    coords_set: set
    volume: int
    shape_2d: tuple | None = None

    def __post_init__(self):
        self._intensities = {}


# --- per-slice coords are stored as sorted, unique uint32 arrays of flat indices
# (y * width + x). These helpers centralize the operations the pipeline needs on
# them, replacing the old Python-set encoding (~15x less memory, mostly faster). ---

def _intersect_count(a, b):
    """Number of shared elements between two sorted, unique 1-D index arrays."""
    if a.size == 0 or b.size == 0:
        return 0
    return int(np.isin(a, b, assume_unique=True).sum())


def _overlaps(a, b):
    """True if two sorted, unique 1-D index arrays share any element."""
    if a.size == 0 or b.size == 0:
        return False
    # Both inputs are already sorted & unique, so probe the smaller into the larger
    # with searchsorted. Avoids np.isin re-sorting b on every call.
    if a.size > b.size:
        a, b = b, a
    pos = np.searchsorted(b, a)
    pos = np.clip(pos, 0, b.size - 1)
    return bool((b[pos] == a).any())


def _flatten_yx(coords_yx, width):
    """(N, 2) integer (y, x) array -> sorted uint32 array of flat indices y*width + x."""
    if len(coords_yx) == 0:
        return np.empty(0, dtype=np.uint32)
    coords_yx = np.asarray(coords_yx)
    flat = coords_yx[:, 0].astype(np.int64) * width + coords_yx[:, 1].astype(np.int64)
    flat.sort()
    return flat.astype(np.uint32)


def _unflatten(flat, width):
    """Sorted uint32 flat array -> (ys, xs) int64 arrays."""
    return np.divmod(np.asarray(flat, dtype=np.int64), width)


def _centroid_yx_from_flat(flat, width):
    """Mean (y, x) of a flat index array."""
    ys, xs = _unflatten(flat, width)
    return float(ys.mean()), float(xs.mean())


def _contains_yx(flat_sorted, y, x, width, height):
    """Bounds-checked membership test for pixel (y, x) in a sorted flat index array."""
    if not (0 <= y < height and 0 <= x < width):
        return False
    key = y * width + x
    pos = np.searchsorted(flat_sorted, key)
    return pos < flat_sorted.size and int(flat_sorted[pos]) == key


def get_cell_properties_2d(mask):
    # Cellpose labels are only unique within a z-slice. Iterate slice by slice and
    # assign globally-unique ids on the fly, which subsumes the old make_unique_mask
    # step (no separate uint32 volume, no full-volume relabel pass). Working per slice
    # also keeps peak memory to one slice's coords instead of the whole volume.
    height, width = mask.shape[1], mask.shape[2]
    organoid = []
    new_label = 0
    for z in range(mask.shape[0]):
        sl = mask[z]
        ys, xs = np.nonzero(sl)
        if ys.size == 0:
            continue
        labs = sl[ys, xs]
        order = np.argsort(labs, kind="stable")  # stable keeps raster order within a label
        labs = labs[order]
        ys = ys[order].astype(np.int64)
        xs = xs[order].astype(np.int64)

        _, idx, counts = np.unique(labs, return_index=True, return_counts=True)
        for start, count in zip(idx, counts):
            y_part = ys[start : start + count]
            x_part = xs[start : start + count]
            # flat = y*W + x, already ascending thanks to the stable sort above
            coords_set = (y_part * width + x_part).astype(np.uint32)
            new_label += 1
            organoid.append(
                cell_2d(
                    label=new_label,
                    centroid=(float(z), float(y_part.mean()), float(x_part.mean())),
                    coords_set=coords_set,
                    volume=int(count),
                    shape_2d=(height, width),
                )
            )

    print(f"Created {new_label} unique 2D cells.")
    return organoid


def properties_channel(organoid, image, channel_type=None):
    # Mean intensity per 2D cell, computed directly from its flat (y, x) coords at its
    # slice. Each unique label lives entirely in one z-slice, so this reproduces
    # regionprops' mean_intensity exactly, without a full-volume regionprops pass.
    attr_name = f"intensity_{channel_type}" if channel_type else "intensity"
    z_max = image.shape[0] - 1
    for cell in organoid:
        z_int = min(max(int(round(cell.centroid[0])), 0), z_max)
        flat = cell.coords_set
        if flat.size:
            mean_val = float(image[z_int].reshape(-1)[flat].mean())
        else:
            mean_val = 0.0
        setattr(cell, attr_name, mean_val)

    return organoid


def filter_big_2d_cells(organoid_2d, size_2d_filter_multiplier = 15):
    # Filter out 2D cells that are multiplier times bigger than the median cell size in that slice, 
    # as these are likely mistakes by cellpose 2D in empty slices.
    # since cellpose SAM is size agnostic.
    median_size = np.median([cell.volume for cell in organoid_2d])
    threshold = size_2d_filter_multiplier * median_size
    return [cell for cell in organoid_2d if cell.volume <= threshold]


def get_2d_mask_properties(
    mask, image1, image_type_1=None, image2=None, image_type_2=None, size_2d_filter_multiplier=15
):
    organoid = get_cell_properties_2d(mask)
    organoid = filter_big_2d_cells(organoid, size_2d_filter_multiplier)

    organoid = properties_channel(organoid, image1, channel_type=image_type_1)

    if image2 is not None:
        organoid = properties_channel(organoid, image2, channel_type=image_type_2)

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
    len1, len2 = set1.size, set2.size
    if len2 == 0:
        return 0

    # Calculate intersection (overlapping pixels)
    intersection = _intersect_count(set1, set2)

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
    # adjacent-slice IOU captured at the moment two 2D cells are stitched, keyed by
    # the upper (z+1) 2D-cell index. Reused later as cell.iou_gaps so get_break_scores
    # never has to recompute consecutive-slice overlaps.
    gap_iou = {}

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
        iou = calculate_iou(coords_sets, idx, cand_idx)
        if iou > iou_threshold:
            gap_iou[cand_idx] = iou
            stitch_forward(cand_idx, current_label)

    current_label = 0
    for i in range(n):
        if labels_3d[i] == -1:
            current_label += 1
            stitch_forward(i, current_label)

    # build organoid_3d list (skip 3D coords to avoid heavy conversions)
    organoid_3d = []
    intensity_attrs = [a for a in organoid_2d[0].__dict__ if a.startswith("intensity")]
    # Group 2D-cell indices by their assigned 3D label in one O(n log n) pass instead
    # of rescanning labels_3d per label (which was O(n_labels * n)).
    order = np.argsort(labels_3d, kind="stable")
    sorted_labels = labels_3d[order]
    uniq, starts = np.unique(sorted_labels, return_index=True)
    starts = np.append(starts, labels_3d.size)
    for k, cell in enumerate(uniq):
        if cell <= 0:
            continue
        slice_idxs = order[starts[k] : starts[k + 1]]  # ascending original indices
        labels_2d = [organoid_2d[i].label for i in slice_idxs]
        centroids_2d = [organoid_2d[i].centroid for i in slice_idxs]
        coords_set_2d = [organoid_2d[i].coords_set for i in slice_idxs]
        volumes_2d = [organoid_2d[i].volume for i in slice_idxs]
        # collect per-channel per-slice intensities
        intensities_2d = {
            attr: [getattr(organoid_2d[i], attr, None) for i in slice_idxs]
            for attr in intensity_attrs
        }

        # per-gap "1 - adjacent IOU", reusing the IOUs already computed while
        # stitching. slice_idxs are z-ascending and contiguous, so gap j sits
        # between slice_idxs[j-1] and slice_idxs[j]; its IOU was stored under the
        # upper index slice_idxs[j]. Fall back to a direct compute only if missing.
        iou_gaps = [0.0]
        for j in range(1, len(slice_idxs)):
            upper = int(slice_idxs[j])
            iou = gap_iou.get(upper)
            if iou is None:
                iou = calculate_iou(coords_sets, int(slice_idxs[j - 1]), upper)
            iou_gaps.append(1 - iou)

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
            iou_gaps=iou_gaps,
        )
        cell3d.shape_2d = shape_2d
        organoid_3d.append(cell3d)

    print(f"Completed IOU based stitching: {len(organoid_3d)} cells formed.")

    return organoid_3d


def mask_from_organoid(organoid, shape):
    mask = np.zeros(shape, dtype=np.uint16)
    for cell in organoid:
        width = cell.shape_2d[1]
        for z_idx, (z, y, x) in enumerate(cell.centroids_2d):
            z_int = int(round(z))
            if 0 <= z_int < shape[0]:
                flat = cell.coords_set_2d[z_idx]
                if flat.size == 0:
                    continue
                ys, xs = _unflatten(flat, width)
                valid = (ys >= 0) & (ys < shape[1]) & (xs >= 0) & (xs < shape[2])
                mask[z_int, ys[valid], xs[valid]] = cell.label
    return mask


# def break_score(values, i):
#     # Calculate intensity-based anomalies by getting a score that resembles high-low-high intensity value patterns
#     # Calculate ratio between slice i and below i, and i and above i
#     bottom = values[max(0, i - 2):i]
#     top = values[i + 1:min(len(values), i + 3)]

#     values_bottom = max(bottom) if len(bottom) else values[i]
#     values_top = max(top) if len(top) else values[i]
#     ratio_1 = values_bottom / values[i]
#     ratio_2 = values[i] / values_top

#     # divide this again, now high low high gets higher score then other patterns
#     return ratio_1 / ratio_2
from scipy.signal import find_peaks, peak_prominences, peak_widths
def break_score(values, min_prominence=0.0, rel_height=0.5, sigma_floor=0.6):
    v = np.asarray(values, dtype=float)
    vmax = v.max()
    if vmax <= 0:
        return np.zeros_like(v)
    inv = vmax - v
    valleys, props = find_peaks(inv, prominence=min_prominence * vmax)
    scores = np.zeros_like(v)
    if len(valleys) == 0:
        return scores
    widths = peak_widths(inv, valleys, rel_height=rel_height)[0]   # width in slices
    idx = np.arange(len(v))
    for vi, prom, w in zip(valleys, props["prominences"], widths):
        sigma = max(w / 2.355, sigma_floor)          # width -> gaussian sigma (FWHM->sigma), with a floor\
        depth = prom / (v[vi] )
        bump = depth * np.exp(-(idx - vi) ** 2 / (2 * sigma ** 2))
        scores = np.maximum(scores, bump)
    scores = np.round(scores, 4)  # clip to only 4 decimals
    return scores


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

    z_fit = z_positions[points]
    x_fit = x_positions[points]
    y_fit = y_positions[points]
    n = len(z_fit)

    # Closed-form ordinary least squares (identical result to lstsq for a full-rank
    # line fit, but ~3x faster here: these fits are tiny (n ~ 3-30) and pure-Python
    # arithmetic avoids the array-allocation overhead lstsq pays per call).
    if n >= 2:
        inv = 1.0 / n
        zbar = sum(z_fit) * inv
        xbar = sum(x_fit) * inv
        ybar = sum(y_fit) * inv
        denom = num_x = num_y = 0.0
        for zi, xi, yi in zip(z_fit, x_fit, y_fit):
            dz = zi - zbar
            denom += dz * dz
            num_x += dz * (xi - xbar)
            num_y += dz * (yi - ybar)
        if denom > 0.0:
            m_x = num_x / denom
            m_y = num_y / denom
            c_x = xbar - m_x * zbar
            c_y = ybar - m_y * zbar
            return lambda z: (m_x * z + c_x, m_y * z + c_y)

    # Rank-deficient (<2 points, or all z equal): fall back to lstsq's min-norm solution.
    A = np.vstack([np.asarray(z_fit, dtype=np.float64), np.ones(n)]).T
    m_x, c_x = np.linalg.lstsq(A, np.asarray(x_fit, dtype=np.float64), rcond=None)[0]
    m_y, c_y = np.linalg.lstsq(A, np.asarray(y_fit, dtype=np.float64), rcond=None)[0]

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
    # volumes_break_score = [0] * num_slices

    # for i in range(1, num_slices - 1):
    #     volumes_break_score[i] = break_score(cell.volumes_2d, i)
    volumes_break_score = break_score(cell.volumes_2d) * 10

    return volumes_break_score


def get_intensity_break_scores(cell):
    num_slices = len(cell.volumes_2d)
    intensity_break_score = [0] * num_slices

    if cell.intensities_2d:
        per_channel = list(cell.intensities_2d.values())
        max_intensity = [max(vals) for vals in zip(*per_channel)]
    else:
        max_intensity = [0] * num_slices

    # for i in range(1, num_slices - 1):
    #     intensity_break_score[i] = break_score(max_intensity, i)
    intensity_break_score = break_score(max_intensity) * 10 # scale to match volume break score scale

    return intensity_break_score


def get_line_break_scores(z_positions, y_positions, x_positions):
    # Calculate line-based break scores (from bottom_up)
    line = calculate_line_3d(z_positions, y_positions, x_positions, points=3)
    distances = calculate_point_line_distances(
        z_positions, y_positions, x_positions, line
    )
    break_score = abs(get_slope_changes(distances))
    return break_score, distances


def _get_iou_gaps(cell):
    # Raw per-gap array: index i = 1 - IOU(slice i-1, slice i) for i >= 1, index 0 = 0.
    # Normally already populated by IOU_stitching (which computed these IOUs while
    # forming the cell); split_cell_at_index then slices it onto the halves. This
    # lazy compute is only a fallback for cells that arrive without it.
    gaps = cell.iou_gaps
    if gaps is None:
        n = len(cell.volumes_2d)
        gaps = [0.0] * n
        for i in range(1, n):
            gaps[i] = 1 - calculate_iou(cell.coords_set_2d, i - 1, i)
        cell.iou_gaps = gaps
    return gaps


def get_IOU_break_scores(cell):
    # Derive the score array from the cached raw gaps. The original left the first
    # and last positions unscored (0); reproduce that exactly by zeroing the ends.
    iou_break_score = list(_get_iou_gaps(cell))
    iou_break_score[0] = 0
    iou_break_score[-1] = 0
    return iou_break_score


def get_break_scores(organoid, skip_labels=None, combine_fn=None):
    # combine_fn(v, i, up, down, iou) -> score lets callers (e.g. an F1
    # optimizer) swap in a different channel combination. None = default formula.
    skip_labels = skip_labels or set()
    for cell in organoid:
        if cell.label in skip_labels and cell.break_scores is not None:
            continue
        num_slices = len(cell.volumes_2d)
        if num_slices < 7:
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
        break_score_up, distances_up = get_line_break_scores(z_positions, y_positions, x_positions)
        # pad 1 at start and remove last to match and better represent breaks between slices
        break_score_up = np.concatenate([[0], break_score_up[:-1]])
        # Calculate line-based break scores (from top_down)
        break_score_down, distances_down = get_line_break_scores(
            z_positions[::-1], y_positions[::-1], x_positions[::-1]
        )
        break_score_down = break_score_down[::-1]
        distances_down = distances_down[::-1]
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
            if combine_fn is not None:
                final_score = combine_fn(
                    v_score, i_score, up_score, down_score, IOU_score
                )
            else:
                final_score = (
                    np.max([up_score, down_score])
                    * (IOU_score * 3)
                    + v_score
                    + i_score
                )
            final_scores.append(final_score)
        
        # if cell.label == 4307:
        #     print(f"Cell {cell.label} break scores:")
        #     print(f"  Number of slices: {len(cell.volumes_2d)}")
        #     print(f"  Volume break scores: {volumes_break_score}")
        #     print(f"  Intensity break scores: {intensity_break_score}")
        #     print(f"  Line break scores (up): {break_score_up}")
        #     # print(f"  Line distances (up): {distances_up}")
        #     print(f"  Line break scores (down): {break_score_down}")
        #     # print(f"  Line distances (down): {distances_down}")
        #     print(f"  IOU break scores: {IOU_break_score}")
        #     print(f"  Final break scores: {final_scores}")

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

    # Slice the cached adjacent-slice IOU gaps onto the halves instead of letting
    # them be recomputed. Internal gaps are identical to the parent's; each half's
    # first gap has no "below" neighbour, so it resets to 0 (get_IOU_break_scores
    # zeroes the trailing end when it derives its score array).
    parent_gaps = _get_iou_gaps(cell)
    bottom_cell.iou_gaps = list(parent_gaps[:break_position])
    top_cell.iou_gaps = [0.0] + list(parent_gaps[break_position + 1:max_z])
    return bottom_cell, top_cell

def cell_bbox(cell):
    """z extent + the LARGEST per-slice y and x extents (all in voxels)."""
    zs = [int(round(c[0])) for c in cell.centroids_2d]
    z_size = max(zs) - min(zs) + 1

    width = cell.shape_2d[1]
    y_size = x_size = 0
    for s in cell.coords_set_2d:
        if s.size == 0:
            continue
        ys, xs = _unflatten(s, width)
        y_size = max(y_size, int(ys.max() - ys.min()) + 1)  # this slice's y extent
        x_size = max(x_size, int(xs.max() - xs.min()) + 1)  # this slice's x extent
    return z_size, y_size, x_size

def break_stitching(organoid, breaking_threshold=2.5, scale=(1,1,1), skip_labels=None):
    skip_labels = skip_labels or set()
    new_organoid = []
    breaks_made = 0
    broken_labels = set()
    new_label = max(cell.label for cell in organoid) + 1
    for cell in organoid:
        # A cell that already survived a previous round with unchanged scores/geometry
        # would make the identical (no-break) decision again, so skip re-evaluating it.
        if cell.label in skip_labels:
            new_organoid.append(cell)
            continue
        if len(cell.volumes_2d) < 7:
            new_organoid.append(cell)
            continue

        break_scores = [score for score in cell.break_scores]
        break_scores[:3] = [0, 0, 0]  # First 3 slices get score 0
        break_scores[-3:] = [0, 0, 0]  # Last 3 slices get score 0
        max_value = max(break_scores)
        # get the z y x bounding box sizes

        if max_value < breaking_threshold:
            # bbox is invariant unless the cell is split; cache it so re-evaluated
            # (but unbroken) cells don't re-unflatten every slice each round. The
            # halves are fresh cell_3d objects without _bbox, so they recompute.
            bbox = getattr(cell, "_bbox", None)
            if bbox is None:
                bbox = cell_bbox(cell)
                cell._bbox = bbox
            z_size, y_size, x_size = bbox
            sizes = np.sort(np.array([z_size * scale[0], y_size * scale[1], x_size * scale[2]]))[::-1]                
            elongation = sizes[0] / sizes[1] - 1
            elongation = elongation if elongation > 0.5 else 0.5
            new_threshold = breaking_threshold / ((elongation*2)**2) # scale the threshold based on elongation, more elongated cells are more likely to be broken

            # if cell.label == 4307:
            #     print(f"Break scores for cell {cell.label}: {cell.break_scores}")
            #     print(f"New threshold: {new_threshold}")
            #     print(f"Elongation: {elongation}, sizes: {sizes}, scale: {scale}")
        elif max_value >= breaking_threshold:
            new_threshold = breaking_threshold

        if max_value < new_threshold:
            new_organoid.append(cell)
            continue

        # Peak = highest-scoring slice
        peak = break_scores.index(max_value)

        # Cut between the peak and its HIGHER-scoring neighbour, so the peak
        # slice stays with its LOWER-scoring side.
        left  = break_scores[peak - 1]
        right = break_scores[peak + 1]
        if right > left:
            split_index = peak + 1   # higher neighbour above -> peak joins bottom
        else:
            split_index = peak       # higher neighbour below (or tie) -> peak joins top

        bottom_cell, top_cell = split_cell_at_index(
            cell, split_index, new_label=new_label
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

    # (min_z, max_z) per cell; empty cells get an impossible range so they never
    # pass the adjacency prefilter below. Two cells can only touch at a z-interface
    # if their z-extents come within one slice of each other.
    z_ranges = [(min(zk), max(zk)) if zk else (1, -1) for zk in z_keys_list]

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

        lo_i, hi_i = z_ranges[i]
        for idx in candidate_indices:
            other_z_to_coords = z_to_coords_list[idx]
            if not other_z_to_coords:
                continue

            # Skip candidates whose z-extent can't be adjacent to this cell's,
            # before paying for the per-z overlap scan below. Necessary condition
            # for touching, so it never changes which pairs are found.
            lo_j, hi_j = z_ranges[idx]
            if lo_i > hi_j + 1 or lo_j > hi_i + 1:
                continue

            is_touching = False
            for z in z_keys:
                coords = z_to_coords.get(z)
                if coords is None or coords.size == 0:
                    continue
                coords_other = other_z_to_coords.get(z - 1)
                if coords_other is not None and _overlaps(coords, coords_other):
                    is_touching = True
                    break
                coords_other = other_z_to_coords.get(z + 1)
                if coords_other is not None and _overlaps(coords, coords_other):
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

    # Vectorized half-plane split: which side of the line each pixel falls on
    ys, xs = _unflatten(cell.coords_set_2d[index], cell.shape_2d[1])
    pts = np.stack((ys, xs), axis=1)                         # (N, 2) as (y, x)
    side = (pts.astype(np.float64) - midpoint) @ normal
    part_1_coords = pts[side <= 0]
    part_2_coords = pts[side > 0]

    return part_1_coords, part_2_coords


def are_split_cells_valid(cell, index, part_1_coords, part_2_coords):
    min_size = max(
        3, len(cell.coords_set_2d[index]) * 0.1
    )  # At least 10% of pixels or 3 pixels

    if len(part_1_coords) < min_size or len(part_2_coords) < min_size:
        # if cell.label == 130:
            # print(
            #     f"  Warning: Split too unbalanced ({len(part_1_coords)} vs {len(part_2_coords)} pixels)"
            # )
            # print(f"  Warning: Split too unbalanced ({len(part_1_coords)} vs {len(part_2_coords)} pixels)")
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

    original_distance_1 = np.sqrt(
        (original_y - pred_point_1[0]) ** 2 + (original_x - pred_point_1[1]) ** 2
    )
    original_distance_2 = np.sqrt(
        (original_y - pred_point_2[0]) ** 2 + (original_x - pred_point_2[1]) ** 2
    )
    average_original_distance = (original_distance_1 + original_distance_2) / 2

    part_1_coords, part_2_coords = calculate_split_coords(
        cell, index, pred_point_1, pred_point_2
    )

    if not are_split_cells_valid(cell, index, part_1_coords, part_2_coords):
        # print(f"    ✗ Split cells not valid")
        return None

    avg_new_distance = calculate_new_distances(
        part_1_coords, part_2_coords, pred_point_1, pred_point_2
    )

    # to_check = 130
    # if cell.label == to_check:
    #     print(f"Cell {cell.label} at index {index}:")
    #     print(f"  Original centroid: ({original_y:.2f}, {original_x:.2f})")
    #     print(f"  Predicted points: ({pred_point_1[0]:.2f}, {pred_point_1[1]:.2f}), ({pred_point_2[0]:.2f}, {pred_point_2[1]:.2f})")
    #     print(f"  Original distances: {original_distance_1:.2f}, {original_distance_2:.2f}")
    #     print(f"  Average original distance: {average_original_distance:.2f}")
    #     print(f"  New centroids: ({np.mean(part_1_coords, axis=0)[0]:.2f}, {np.mean(part_1_coords, axis=0)[1]:.2f}), ({np.mean(part_2_coords, axis=0)[0]:.2f}, {np.mean(part_2_coords, axis=0)[1]:.2f})")
    #     print(f"  New average distance: {avg_new_distance:.2f}")

    if avg_new_distance < average_original_distance:
        # if cell.label == to_check:
        #     print(
        #         f"    ✓ Splitting improves fit! cell {cell.label} (reduction: {average_original_distance - avg_new_distance:.2f})"
        #     )
        return (part_1_coords, part_2_coords)
    else:
        # if cell.label == to_check:
        #     print(f"    ✗ Splitting does not improve fit: increase of {avg_new_distance:.2f} >= {average_original_distance:.2f}")
        return None


def split_multiple_cell_layers(cell_1, cell_2):
    z_to_predicts = [1, 2]
    split_cells = 0
    z1 = {int(round(c[0])) for c in cell_1.centroids_2d}
    z2 = {int(round(c[0])) for c in cell_2.centroids_2d}
    share_z = bool(z1 & z2)

    for z_to_predict in z_to_predicts:
        # Split cell_1 at its top
        # print("bottom cell z:", z_to_predict)
        split_results = split_single_cell_layer(
            cell_1, cell_2, z_to_predict=z_to_predict, bottom_up=True
        )
        if split_results is not None:
            split_cells += 1
            part_1_coords, part_2_coords, z_idx, last_z = split_results
            width = cell_1.shape_2d[1]
            part_1_coords_2d = _flatten_yx(part_1_coords, width)
            part_2_coords_2d = _flatten_yx(part_2_coords, width)
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
            width = cell_2.shape_2d[1]
            part_1_coords_2d = _flatten_yx(part_1_coords, width)
            part_2_coords_2d = _flatten_yx(part_2_coords, width)
            update_coords_bottom(
                cell_2, cell_1, z_idx, part_1_coords_2d, part_2_coords_2d, first_z
            )

        # When no splits are made on the 1st level, we dont need to check splitting the second level.
        if split_cells == 0 and not share_z:
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
    width = cell_1.shape_2d[1]
    height = cell_1.shape_2d[0]
    coords_at_z = cell_1.coords_set_2d[z_idx]
    predicted_line_1_in_cell_1 = _contains_yx(
        coords_at_z, int(round(predict_y_1)), int(round(predict_x_1)), width, height
    )
    predicted_line_2_in_cell_1 = _contains_yx(
        coords_at_z, int(round(predict_y_2)), int(round(predict_x_2)), width, height
    )

    # print(predicted_line_1_in_cell_1, predicted_line_2_in_cell_1)
    # print(int(round(predict_y_1*0.64)), int(round(predict_x_1*0.64)), int(round(predict_y_2*0.64)), int(round(predict_x_2*0.64)))
    if not (predicted_line_1_in_cell_1 and predicted_line_2_in_cell_1):
        # if cell_1.label == 130 or cell_2.label == 130:
        #     print(
        #         f"  ✗ Predicted splitting points not in cell {cell_1.label} at z={int(round(target_z))}"
        #     )
        return None

    pred_point_1 = np.array([predict_y_1, predict_x_1])
    pred_point_2 = np.array([predict_y_2, predict_x_2])

    # print(f"Attempting to split touching cells {cell_1.label} and {cell_2.label}")
    split_results = check_split_cells(cell_1, z_idx, pred_point_1, pred_point_2)
    if split_results is None:
        return None

    part_1_coords, part_2_coords = split_results

    # Connectivity guard: the donated piece (part_2) must actually overlap the
    # receiving cell's interface slice. Without this, a half-plane cut can carve a
    # chunk out of cell_1's interior and relabel it cell_2 even though it forms an
    # island unconnected to cell_2's body. cell_2's interface slice is its bottom
    # (index 0) when splitting cell_1 from below, else its top (index -1).
    if cell_2.coords_set_2d:
        neighbor_slice = cell_2.coords_set_2d[0 if bottom_up else -1]
        part_2_flat = _flatten_yx(part_2_coords, width)
        if not _overlaps(part_2_flat, neighbor_slice):
            # print(f"  ✗ Split piece does not touch cell {cell_2.label}'s body")
            return None

    target_z_int = int(round(target_z))
    return part_1_coords, part_2_coords, z_idx, target_z_int


def _insert_slice_in_z_order(cell, z, coords_flat, label, width):
    """Insert a donated 2D slice into a cell, keeping all per-slice lists sorted by z.

    The previous code blindly appended (or inserted at index 0), which left
    centroids_2d non-monotonic in z. get_centroid_positions / calculate_line_3d
    assume z-ordered input (note the [::-1] reversals), so an out-of-order point
    corrupted the line fit on the z_to_predict=2 round and misplaced the split.
    """
    cy, cx = _centroid_yx_from_flat(coords_flat, width)
    centroid = (z, cy, cx)

    pos = 0
    while pos < len(cell.centroids_2d) and cell.centroids_2d[pos][0] < z:
        pos += 1

    cell.coords_set_2d.insert(pos, coords_flat)
    cell.labels_2d.insert(pos, label)
    cell.centroids_2d.insert(pos, centroid)
    cell.volumes_2d.insert(pos, int(coords_flat.size))


def update_coords(cell_1, cell_2, z_idx, part_1_coords_2d, part_2_coords_2d, last_z):
    """Update both cells after splitting cell_1 at its top"""
    width = cell_1.shape_2d[1]
    cell_1.coords_set_2d[z_idx] = part_1_coords_2d

    cell_1.volumes_2d[z_idx] = int(part_1_coords_2d.size)
    cy, cx = _centroid_yx_from_flat(part_1_coords_2d, width)
    cell_1.centroids_2d[z_idx] = (last_z, cy, cx)

    # Donate part_2 to cell_2 in z-order (interface z sits below cell_2's body)
    _insert_slice_in_z_order(
        cell_2, last_z, part_2_coords_2d, cell_1.labels_2d[z_idx], width
    )

    _recalculate_cell_properties(cell_1)
    _recalculate_cell_properties(cell_2)


def update_coords_bottom(
    cell_2, cell_1, z_idx, part_1_coords_2d, part_2_coords_2d, first_z
):
    """Update both cells after splitting cell_2 at its bottom"""
    width = cell_2.shape_2d[1]
    cell_2.coords_set_2d[z_idx] = part_1_coords_2d

    cell_2.volumes_2d[z_idx] = int(part_1_coords_2d.size)
    cy, cx = _centroid_yx_from_flat(part_1_coords_2d, width)
    cell_2.centroids_2d[z_idx] = (first_z, cy, cx)

    # Donate part_2 to cell_1 in z-order (interface z sits above cell_1's body)
    _insert_slice_in_z_order(
        cell_1, first_z, part_2_coords_2d, cell_2.labels_2d[z_idx], width
    )

    _recalculate_cell_properties(cell_1)
    _recalculate_cell_properties(cell_2)


def _recalculate_cell_properties(cell):
    """Helper to recalculate volume and centroid after modification"""
    cell.volume = sum(cell.volumes_2d)
    cell.centroid = np.mean(cell.centroids_2d, axis=0)


import copy


def split_organoid_cells(organoid):
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


def stitch_small_cells(organoid, max_slices=2, iou_threshold=0.7, knn=5):
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
                # if cell.label == 98:
                #     print(f"Cell {cell.label} at z={z_pos} has {len(cands)} candidates at z={neighbor_z}")
                k_actual = min(knn, len(cands))
                _, nn_idxs = kd.query((cy, cx), k=k_actual)
                for nn_i in np.atleast_1d(nn_idxs):
                    j = cands[nn_i]
                    s2 = organoid[j].coords_set_2d[z_to_idx_maps[j][neighbor_z]]
                    len2 = s2.size
                    if len2 == 0:
                        continue
                    intersection = _intersect_count(s1, s2)
                    if intersection == 0:
                        continue
                    min_len = min(len1, len2)
                    iou = (
                        1.0
                        if intersection == min_len
                        else intersection / (len1 + len2 - intersection)
                    )
                    # if cell.label == 98:
                    #     print(f"  Candidate neighbor {organoid[j].label} at z={neighbor_z}: len1={len1}, len2={len2}, intersection={intersection}, iou={iou:.3f}")
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
    size_2d_filter_multiplier=15,
    combine_fn=None,
    scale=(1, 1, 1),
):
    print("Starting 3D stitching process...")
    organoid = get_2d_mask_properties(
        mask,
        image1,
        image_type_1=image_type_1,
        image2=image2,
        image_type_2=image_type_2,
        size_2d_filter_multiplier=size_2d_filter_multiplier
    )
    organoid = IOU_stitching(organoid, distance_threshold=10, iou_threshold=0.3)

    # Iterative break and restitching
    max_iterations = 10
    iteration = 0
    checked_labels = set()
    print("Starting iterative breaking of cells")
    while iteration < max_iterations:
        skip = checked_labels  # survivors from the previous round (empty on the first)
        organoid = get_break_scores(
            organoid, skip_labels=skip, combine_fn=combine_fn
        )
        organoid, breaks_made, broken_labels = break_stitching(
            organoid, breaking_threshold=breaking_threshold, scale=scale, skip_labels=skip
        )
        # survivors = all current cells minus the ones that were (re)broken this round
        checked_labels = {cell.label for cell in organoid} - broken_labels
        if breaks_made == 0:
            break  # No more breaks made

        print(f"    Iteration {iteration + 1}: Made {breaks_made} breaks.")
        iteration += 1


    organoid = split_organoid_cells(organoid)    

    organoid = stitch_small_cells(organoid)

    stitched_mask = mask_from_organoid(organoid, shape=mask.shape)

    print(
        f"Final stitched mask has {len(np.unique([cell.label for cell in organoid]))} cells."
    )

    return stitched_mask