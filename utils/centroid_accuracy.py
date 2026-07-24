import numpy as np
from skimage.measure import regionprops


def centroid_accuracy(gt, pred_mask, chunk=32):
    """Centroid-matching accuracy of a labelled prediction against GT.

    Each GT label is located by its centroid voxel; for concave nuclei whose
    centre of mass falls outside the object, the nearest interior voxel is used
    instead, so a self-comparison scores a perfect 1.0. Memory stays bounded:
    point selection works one object-bounding-box at a time (regionprops) and
    the volume-wide Dice / label enumeration is streamed in z-chunks.
    """
    gt = np.asarray(gt)
    pred_mask = np.asarray(pred_mask)
    if gt.shape != pred_mask.shape:
        raise ValueError("Masks must share shape")

    ndim = gt.ndim
    dim_arr = np.array(gt.shape)

    gt_props = regionprops(gt)
    hit_pred_labels = set()
    tp = fn = merged_cells = 0
    fn_list = []

    for prop in gt_props:
        # regionprops already builds prop.image (the label's mask crop) to
        # compute moments, so the "is the centroid inside?" check is free.
        local = np.rint(prop.local_centroid).astype(int)
        local = np.clip(local, 0, np.array(prop.image.shape) - 1)
        offset = np.asarray(prop.bbox[:ndim])
        if prop.image[tuple(local)]:
            # Read the exact voxel just verified in the mask crop. (Deriving it
            # from prop.centroid instead can round to a neighbouring voxel and
            # pick up an adjacent label.)
            centroid = tuple(local + offset)
        else:
            # concave: pick the interior voxel nearest the centre of mass
            coords = np.argwhere(prop.image)
            c = np.asarray(prop.local_centroid)
            j = np.argmin(((coords - c) ** 2).sum(axis=1))
            centroid = tuple(coords[j] + offset)

        pred_label = pred_mask[centroid]
        if pred_label > 0:
            if pred_label in hit_pred_labels:
                fn += 1
                merged_cells += 1
            else:
                tp += 1
                hit_pred_labels.add(int(pred_label))
        else:
            fn += 1
            fn_list.append(int(prop.label))

    # --- Dice + prediction-label enumeration, streamed in z-chunks ----------
    # (avoids whole-volume boolean temporaries and the full-volume sort that
    # np.unique(pred_mask) would do)
    gt_fg = pred_fg = inter = 0
    maxlab = int(pred_mask.max()) if pred_mask.size else 0
    hist = np.zeros(maxlab + 1, dtype=np.int64)
    for z0 in range(0, gt.shape[0], chunk):
        g = gt[z0:z0 + chunk]
        p = pred_mask[z0:z0 + chunk]
        gm = g > 0
        pm = p > 0
        gt_fg += int(np.count_nonzero(gm))
        pred_fg += int(np.count_nonzero(pm))
        inter += int(np.count_nonzero(gm & pm))
        hist += np.bincount(p.reshape(-1), minlength=hist.size)
    total = gt_fg + pred_fg
    dice = (2 * inter) / total if total else 0.0

    pred_labels = set(int(l) for l in np.flatnonzero(hist))
    pred_labels.discard(0)
    fp_list = list(pred_labels - hit_pred_labels)
    fp = len(fp_list)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "merged_cells": merged_cells,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "voxel_dice": dice,
        "tp_list": list(hit_pred_labels),
        "fp_list": fp_list,
        "fn_list": fn_list,
    }
