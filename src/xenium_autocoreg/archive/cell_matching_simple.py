"""Cell matching by mask IoU (curated from alignment_fucntions.py).
`find_mask_matches_fast` is the sparse-cooccurrence version used to produce the
ground-truth `cell_matching/section_N_cell_matching.csv`."""
import numpy as np
from scipy import sparse


def calculate_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union else 0.0


def calculate_centroid(mask):
    y, x = np.where(mask)
    return (np.mean(x), np.mean(y)) if len(x) else (np.nan, np.nan)


def find_mask_matches_fast(labels1, labels2, threshold=0.2):
    """Greedy IoU matching of two label images (0=bg) via a sparse co-occurrence matrix.
    Returns (matches[(i,j)], unmatched1, unmatched2, iou_matrix), where i/j are 0-based
    label indices (add 1 for the label id)."""
    n1, n2 = int(labels1.max()), int(labels2.max())
    if n1 == 0 or n2 == 0:
        return [], list(range(n1)), list(range(n2)), np.zeros((n1, n2))
    overlap = sparse.coo_matrix(
        (np.ones(labels1.size, np.uint32), (labels1.ravel(), labels2.ravel())),
        shape=(n1 + 1, n2 + 1)).toarray()
    inter = overlap[1:, 1:].astype(np.float64)
    s1 = overlap[1:, :].sum(1)[:, None]
    s2 = overlap[:, 1:].sum(0)[None, :]
    union = s1 + s2 - inter
    iou = np.zeros_like(inter); ok = union > 0
    iou[ok] = inter[ok] / union[ok]
    best_j = iou.argmax(1); best = iou[np.arange(n1), best_j]
    matches, m1, m2 = [], set(), set()
    for i in range(n1):
        if best[i] >= threshold:
            matches.append((i, int(best_j[i]))); m1.add(i); m2.add(int(best_j[i]))
    return (matches,
            [i for i in range(n1) if i not in m1],
            [j for j in range(n2) if j not in m2], iou)


def match_table(xenium_masks, zstack_masks, threshold=0.2):
    """Return a DataFrame (mask_id_xenium, mask_id_cz, iou) — the GT csv schema."""
    import pandas as pd
    matches, _, _, iou = find_mask_matches_fast(xenium_masks, zstack_masks, threshold)
    return pd.DataFrame({
        "mask_id_xenium": [m[0] + 1 for m in matches],
        "mask_id_cz":     [m[1] + 1 for m in matches],
        "iou":            [iou[m[0], m[1]] for m in matches]})
