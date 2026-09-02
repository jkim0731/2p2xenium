"""Cell populations for soma-print matching: z-stack GCaMP cell centroids, and the Xenium
comparable population per section.

Xenium population priority:
  1. reporter+ (EGFP+SYFP2 transcript count >= min_count) from the raw processed Xenium zarr --
     the validated population (see the coreg-population-choice project memory). Used whenever
     `cfg.reporter_zarr_root` is available.
  2. ALL segmented Xenium cells -- the fallback when no zarr is mounted (most new subjects). This
     is a reasonable substitute, not a hack: on 816462, reporter+ at the validated min_count=2
     threshold already kept ~94% of all cells (19522/20696) -- the reporter filter was barely
     selective at that threshold, so using everyone changes the effective population very little.
"""
import glob
import numpy as np
import tifffile as tiff
import zarr
import scipy.sparse as ssp

from . import XENIUM_S2_UM

REPORTER_GENES = ["EGFP", "SYFP2", "pHaloTag-EGFP"]


def centroids_2d(masks):
    flat = masks.ravel(); nz = np.flatnonzero(flat); lab = flat[nz]
    yy, xx = np.unravel_index(nz, masks.shape)
    cnt = np.bincount(lab)
    cy = np.bincount(lab, yy) / np.maximum(cnt, 1)
    cx = np.bincount(lab, xx) / np.maximum(cnt, 1)
    ids = np.flatnonzero(cnt); ids = ids[ids > 0]
    return ids, np.column_stack([cy[ids], cx[ids]])


def centroids_3d(masks):
    flat = masks.ravel(); nz = np.flatnonzero(flat); lab = flat[nz]
    zz, yy, xx = np.unravel_index(nz, masks.shape)
    cnt = np.bincount(lab)
    cz = np.bincount(lab, zz) / np.maximum(cnt, 1)
    cy = np.bincount(lab, yy) / np.maximum(cnt, 1)
    cx = np.bincount(lab, xx) / np.maximum(cnt, 1)
    ids = np.flatnonzero(cnt); ids = ids[ids > 0]
    return ids, np.column_stack([cz[ids], cy[ids], cx[ids]])


def load_zstack_cells(cfg):
    """(ids, xy_um, plane) for every segmented z-stack cell. xy in the z-stack's own (x,y) frame.
    Uses cfg.zstack_xy_um (per-subject -- NOT always 700/512, see config.py) rather than the
    global ZSTACK_XY_UM constant."""
    seg = tiff.imread(cfg.zstack_segmented_tif)
    ids, czyx = centroids_3d(seg)
    xy_um = czyx[:, 1:][:, ::-1] * cfg.zstack_xy_um
    plane = czyx[:, 0]
    return ids, xy_um, plane


def reporter_counts_by_label(cfg, sec):
    """mask label -> summed reporter transcript count, from the raw processed Xenium zarr.
    Returns {} if no zarr is mounted for this subject (caller should treat that as all-cells)."""
    if cfg.reporter_zarr_root is None:
        return None
    base = glob.glob(str(cfg.reporter_zarr_root / f"section_{sec}.zarr"))
    if not base:
        return None
    base = base[0]
    genes = [str(x) for x in zarr.open(f"{base}/tables/table/var/_index", mode="r")[:]]
    cols = [genes.index(g) for g in REPORTER_GENES if g in genes]
    g = zarr.open(f"{base}/tables/table/X", mode="r")
    n_obs = zarr.open(f"{base}/tables/table/obs/cell_labels", mode="r").shape[0]
    X = ssp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=(n_obs, len(genes)))
    rep = np.asarray(X[:, cols].sum(1)).ravel() if cols else np.zeros(n_obs)
    labels = zarr.open(f"{base}/tables/table/obs/cell_labels", mode="r")[:]
    out = {}
    for lab, c in zip(labels, rep):
        out[int(lab)] = out.get(int(lab), 0) + float(c)
    return out


def load_xenium_cells(cfg, sec, aligned=True, min_count=2):
    """(ids, xy_um) for the comparable Xenium population at one section.
    aligned=True reads Masks_aligned.tif (the shared cross-section frame, for matching);
    aligned=False reads the raw Masks.tif (for converting back / evaluation)."""
    suffix = "_aligned" if aligned else ""
    mask_path = cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks{suffix}.tif"
    masks = np.squeeze(tiff.imread(mask_path))
    ids, cyx = centroids_2d(masks)

    rep = reporter_counts_by_label(cfg, sec)
    if rep is None:
        keep = np.ones(len(ids), bool)          # fallback: all cells
        used_reporter = False
    else:
        keep = np.array([rep.get(int(i), 0) >= min_count for i in ids])
        used_reporter = True

    xy_um = cyx[keep][:, ::-1] * XENIUM_S2_UM
    return ids[keep], xy_um, used_reporter, len(ids)
