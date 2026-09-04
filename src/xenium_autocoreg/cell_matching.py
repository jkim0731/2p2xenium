"""Statistical cell matching: a kNN spatial-shift null model -> Mahalanobis distance -> empirical
p-value on IoU vs. a random-shift background, plus its QC figure. A match is `valid` only when
both the raw IoU and the empirical p-value clear their thresholds -- this rejects incidental
overlaps that a plain IoU cutoff alone would accept."""
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import find_contours
from scipy.stats import chi2, false_discovery_control
from .reference_ported import calculate_centroid, find_mask_matches_fast
from .resources import pool_map

K = 5


def match_section_probability(xenium_dir, ophys_dir, out_dir, section, rng=None):
    rng = rng or np.random.default_rng()
    shifts = [[0, 0]] + rng.integers(-100, -50, size=(100, 2)).tolist() + rng.integers(50, 100, size=(100, 2)).tolist()

    xenium_masks = tiff.imread(Path(xenium_dir) / f"section_{section}_Masks_transformed.tif")
    ophys_masks = tiff.imread(Path(ophys_dir) / f"section_{section}_zstack_warped_masks_plane.tif")
    masks_ids_xenium = np.unique(xenium_masks); masks_ids_xenium = masks_ids_xenium[masks_ids_xenium != 0]
    masks_ids_ophys = np.unique(ophys_masks); masks_ids_ophys = masks_ids_ophys[masks_ids_ophys != 0]

    mask_centroids_ophys = np.array([calculate_centroid(ophys_masks == mid) for mid in masks_ids_ophys])
    center_cells_indices = np.where((mask_centroids_ophys[:, 0] > 0) & (mask_centroids_ophys[:, 0] < 1000) &
                                     (mask_centroids_ophys[:, 1] > 0) & (mask_centroids_ophys[:, 1] < 1000))[0]

    knn_distances = np.empty((mask_centroids_ophys.shape[0], K))
    knn_indices = np.empty((mask_centroids_ophys.shape[0], K), dtype=int)
    for i, c2 in enumerate(mask_centroids_ophys):
        if np.isnan(c2).any():
            knn_distances[i] = np.nan; knn_indices[i] = -1
            continue
        dists = np.linalg.norm(mask_centroids_ophys - c2, axis=1)
        idx = np.argsort(dists)[:K]
        knn_distances[i] = dists[idx]; knn_indices[i] = idx

    iou_max = np.zeros((len(shifts), center_cells_indices.shape[0]))
    iou_max_neighbor = np.zeros((len(shifts), center_cells_indices.shape[0], K))
    matches_best = np.zeros((len(shifts), center_cells_indices.shape[0], 2), dtype=int)

    for i_shift, shift in enumerate(shifts):
        matches, _, _, iou_matrix = find_mask_matches_fast(
            np.roll(xenium_masks, shift, axis=(0, 1)), ophys_masks, threshold=0)
        matched_xenium = np.array([m[0] + 1 for m in matches])
        matched_ophys = np.array([m[1] + 1 for m in matches])
        iou = np.array([iou_matrix[m[0], m[1]] for m in matches])
        for i_mask, idx in enumerate(center_cells_indices):
            if masks_ids_ophys[idx] not in matched_ophys:
                continue
            candidates = np.flatnonzero(matched_ophys == masks_ids_ophys[idx])
            ind_best = candidates[np.argmax(iou[candidates])]
            iou_max[i_shift, i_mask] = iou[ind_best]
            matches_best[i_shift, i_mask] = [matched_xenium[ind_best], matched_ophys[ind_best]]
            for i_k in range(K):
                neighbor_idx = knn_indices[idx, i_k]
                if neighbor_idx == -1 or masks_ids_ophys[neighbor_idx] not in matched_ophys:
                    iou_max_neighbor[i_shift, i_mask, i_k] = 0
                else:
                    iou_max_neighbor[i_shift, i_mask, i_k] = iou[np.where(matched_ophys == masks_ids_ophys[neighbor_idx])[0]].max()

    iou = iou_max[0]
    iou_neighbor = iou_max_neighbor[0]
    iou_random = iou_max[1:]
    iou_neighbor_random = iou_max_neighbor[1:]

    data_best = np.array([iou_neighbor.mean(axis=1), iou])
    data_random = np.array([iou_neighbor_random.mean(axis=2).T, iou_random.T]).transpose(0, 2, 1)
    n_dim, n_random, n_neurons = data_random.shape

    mahal_pvalues = np.full(n_neurons, np.nan)
    empirical_pvalues = np.full(n_neurons, np.nan)
    for i in range(n_neurons):
        if iou_max[0][i] == 0:
            continue
        random_pts = data_random[:, :, i]
        best_pt = data_best[:, i]
        if np.all(random_pts == 0) or np.std(random_pts[0]) == 0 or np.std(random_pts[1]) == 0:
            continue
        mu = random_pts.mean(axis=1)
        cov = np.cov(random_pts)
        if np.linalg.det(cov) < 1e-12:
            continue
        cov_inv = np.linalg.inv(cov)
        diff = best_pt - mu
        d2 = diff @ cov_inv @ diff
        mahal_pvalues[i] = chi2.sf(d2, df=2)
        random_diffs = random_pts.T - mu
        d2_random = np.sum((random_diffs @ cov_inv) * random_diffs, axis=1)
        empirical_pvalues[i] = np.mean(d2_random >= d2)

    valid = ~np.isnan(mahal_pvalues)
    mahal_qvalues = np.full(n_neurons, np.nan)
    if valid.any():
        mahal_qvalues[valid] = false_discovery_control(mahal_pvalues[valid], method="bh")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"section_{section}_iou_data.npz", iou=iou, iou_neighbor=iou_neighbor,
             iou_random=iou_random, iou_neighbor_random=iou_neighbor_random, matches_best=matches_best,
             shifts=shifts, center_cells_indices=center_cells_indices, mahal_pvalues=mahal_pvalues,
             empirical_pvalues=empirical_pvalues, mahal_qvalues=mahal_qvalues)

    matching_df = pd.DataFrame()
    matching_df["mask_id_xenium"] = matches_best[0, :, 0]
    matching_df["mask_id_cz"] = matches_best[0, :, 1]
    matching_df["iou"] = iou
    matching_df["iou_neghbors"] = iou_neighbor.mean(axis=1)
    matching_df["mahal_pvalues"] = mahal_pvalues
    matching_df["mahal_qvalues"] = mahal_qvalues
    matching_df["empirical_pvalues"] = empirical_pvalues
    matching_df = matching_df[matching_df["iou"] > 0]
    matching_df["section"] = section
    matching_df["valid"] = (matching_df["empirical_pvalues"] < 0.05) & (matching_df["iou"] > 0.2)
    matching_df.to_csv(out_dir / f"section_{section}_matching_results.csv", index=False)

    qc_dir = out_dir.parent / "QC" / "cell_matching"
    qc_dir.mkdir(parents=True, exist_ok=True)
    _save_cell_matching_qc(qc_dir, section, xenium_masks, ophys_masks, masks_ids_xenium, masks_ids_ophys, matching_df)
    return matching_df


def _save_cell_matching_qc(qc_dir, section, xenium_masks, ophys_masks, masks_ids_xenium, masks_ids_ophys, matching_table):
    """Reference's exact 3-panel QC (cell_matching_probability.ipynb cell 5)."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for mask_id in masks_ids_xenium:
        for contour in find_contours(xenium_masks == mask_id, 0.5):
            axs[0].fill(contour[:, 1], contour[:, 0], linewidth=1, color="red", alpha=0.5)
    axs[0].set_xticks([]); axs[0].set_yticks([])
    for mask_id in masks_ids_ophys:
        for contour in find_contours(ophys_masks == mask_id, 0.5):
            axs[1].fill(contour[:, 1], contour[:, 0], linewidth=1, color="blue", alpha=0.5)
    axs[1].set_xticks([]); axs[1].set_yticks([])
    for _, row in matching_table.iterrows():
        if not row["valid"]:
            continue
        m1 = xenium_masks == row["mask_id_xenium"]
        m2 = ophys_masks == row["mask_id_cz"]
        for contour in find_contours(m1, 0.5):
            axs[2].fill(contour[:, 1], contour[:, 0], linewidth=1, color="red", alpha=0.5, edgecolor="red")
        for contour in find_contours(m2, 0.5):
            axs[2].fill(contour[:, 1], contour[:, 0], linewidth=1, color="blue", alpha=0.5, edgecolor="blue")
    axs[2].set_xticks([]); axs[2].set_yticks([])
    plt.suptitle(f"Section {section} - Cell Matching", fontsize=16)
    fig.savefig(qc_dir / f"section_{section}_cell_matching.png")
    plt.close(fig)


def _match_one(args):
    xenium_dir, ophys_dir, cell_matching_dir, sec, seed = args
    print(f"[cell_matching_probability] section {sec}", flush=True)
    return match_section_probability(xenium_dir, ophys_dir, cell_matching_dir, sec,
                                      rng=np.random.default_rng(seed))


def run_cell_matching_probability(subject_id, out_dir, sections, num_cpus=None, reserve_cpus=0):
    """Each section is independent (its own random-shift null model) -- parallelized across
    sections. A per-section seed (derived from the section number) keeps results reproducible
    regardless of worker scheduling order. `num_cpus`/`reserve_cpus`: see
    `resources.resolve_num_cpus`."""
    out_dir = Path(out_dir)
    xenium_dir = out_dir / "Xenium_affine_transformed"
    ophys_dir = out_dir / "warped_zstacks"
    cell_matching_dir = out_dir / "cell_matching_probability"
    cell_matching_dir.mkdir(parents=True, exist_ok=True)

    args = [(xenium_dir, ophys_dir, cell_matching_dir, sec, 1000 + sec) for sec in sections]
    all_tables = pool_map(_match_one, args, num_cpus, reserve_cpus=reserve_cpus)

    total = pd.concat(all_tables, axis=0).reset_index(drop=True)
    total.to_csv(cell_matching_dir / f"mouse_{subject_id}_total_matching_results.csv", index=False)
    return total
