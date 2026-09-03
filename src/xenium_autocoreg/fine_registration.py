"""
Fine registration: fine mask-based tile correlation -> TPS control points -> warp -> the
probability-filtered cell-matching metric. This supersedes the coarser approach in `archive/
fine_register_old.py`, which fit the TPS from a coarse intensity tile pass instead of a fine
mask-based one, and never ran the probability-matching step at all.

Produces, per section, the full canonical output layout under `out_dir`:
    Affine matrices/section_N_{affine_matrix,rotation_3d}.npy
    Xenium_affine_transformed/section_N_{Neurons,Masks,Masks_outline}_transformed.tif
    warped_zstacks/section_N_zstack_warped_{plane,masks_plane}.tif
    post_affine_warping/section_N.csv
    QC/xenium_affine_zstack/section_N_xenium_affine_zstack.png
    cell_matching_probability/section_N_{matching_results.csv,iou_data.npz}
See the top-level README for the full structure including the stages this module doesn't produce
(cell_centroids/, mapped_3d_coordinates/, ophys-z-stacks*/ -- those are copy_zstack_assets.py /
transform_xenium_points.py / a plain regionprops pass, run separately by cli.py).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

from .geometry import apply_affine_based_on_reference_2d
from .chain_refine import MARGIN
from .tile_warp import tile_based_warping_parallel as tile_based_warping
from .cell_matching import match_section_probability

FINE_TILE_SIZE = (64, 64)
FINE_OVERLAP = 0.4
FINE_MAX_SHIFT = (10, 10, 10)


def _post_affine_table(zyx_xen, zyx_zst, best_corrs):
    n = len(best_corrs)
    valid = np.array(best_corrs) > 0.2
    ids = [f"pt-{i}-{best_corrs[i]:.2f}-{best_corrs[i]:.2f}".replace("0.", "") for i in range(n)]
    return pd.DataFrame({
        0: ids, 1: valid,
        2: zyx_zst[:, 2], 3: zyx_zst[:, 1], 4: zyx_zst[:, 0],
        5: zyx_xen[:, 2], 6: zyx_xen[:, 1], 7: zyx_xen[:, 0] + np.random.rand(n) * 0.01,
        8: np.array(best_corrs),
    })


def _tps_warp_plane(xyz_xen, xyz_zst, best_corrs, zstack, zstack_masks, z_base, corr_thr=0.2):
    xyz_zst = xyz_zst.copy().astype(float)
    xyz_xen = xyz_xen.copy().astype(float)
    below = np.array(best_corrs) < corr_thr
    xyz_zst[below, :2] = xyz_xen[below, :2]
    xyz_xen[below, 2] = z_base

    zeros = np.zeros_like(xyz_xen[:, 0])
    interp_x = Rbf(xyz_xen[:, 0], xyz_xen[:, 1], zeros, xyz_zst[:, 0], function="thin_plate")
    interp_y = Rbf(xyz_xen[:, 0], xyz_xen[:, 1], zeros, xyz_zst[:, 1], function="thin_plate")
    interp_z = Rbf(xyz_xen[:, 0], xyz_xen[:, 1], zeros, xyz_zst[:, 2], function="thin_plate")

    H, W = zstack.shape[1:]
    yy, xx = np.mgrid[:H, :W]
    xf, yf = xx.ravel().astype(float), yy.ravel().astype(float)
    zf = np.zeros_like(xf)
    x_new = np.clip(interp_x(xf, yf, zf), 0, zstack.shape[2] - 1).astype(int)
    y_new = np.clip(interp_y(xf, yf, zf), 0, zstack.shape[1] - 1).astype(int)
    z_new = np.clip(interp_z(xf, yf, zf), 0, zstack.shape[0] - 1).astype(int)

    warped_plane = zstack[z_new, y_new, x_new].reshape(H, W).astype(zstack.dtype)
    warped_masks = zstack_masks[z_new, y_new, x_new].reshape(H, W).astype(zstack_masks.dtype)
    return warped_plane, warped_masks


def save_alignment_qc(out_path, zstack_warped_plane, xenium_img_transformed, zstack_warped_masks_plane,
                      xenium_masks_transformed, plow=5, phigh=99):
    """Reference's exact 2x3 QC layout (xenium_automatic_coreg.ipynb cell 14)."""
    img = zstack_warped_plane.astype(np.float32)
    vmin, vmax = np.percentile(img, [plow, phigh])
    img_r = np.clip((img - vmin) / (vmax * 1.2 - vmin) * 255, 0, 255).astype(np.uint8)

    img_x = xenium_img_transformed.astype(np.float32)
    vmin, vmax = np.percentile(img_x, [plow, phigh])
    img_g = np.clip((img_x - vmin) / (vmax * 2 - vmin) * 255, 0, 255).astype(np.uint8)

    img_zstack_masks = (zstack_warped_masks_plane > 0).astype(np.uint8) * 255
    img_xenium_masks = (xenium_masks_transformed > 0).astype(np.uint8) * 255

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    rgb1 = np.zeros((*img.shape, 3), np.uint8); rgb1[..., 0] = img_r
    rgb2 = np.zeros((*img.shape, 3), np.uint8); rgb2[..., 1] = img_g
    rgb3 = np.zeros((*img.shape, 3), np.uint8); rgb3[..., 0] = img_r; rgb3[..., 1] = img_g

    axs[0, 0].imshow(rgb1); axs[0, 0].set_title("2p zstack"); axs[0, 0].set_xticks([]); axs[0, 0].set_yticks([])
    axs[0, 1].imshow(rgb2); axs[0, 1].set_title("Xenium"); axs[0, 1].set_xticks([]); axs[0, 1].set_yticks([])
    axs[0, 2].imshow(rgb3); axs[0, 2].set_title("overlap"); axs[0, 2].set_xticks([]); axs[0, 2].set_yticks([])

    zm_rgb = np.zeros((*zstack_warped_masks_plane.shape, 3), np.uint8); zm_rgb[..., 0] = img_zstack_masks
    axs[1, 0].imshow(zm_rgb); axs[1, 0].set_title("Zstack masks"); axs[1, 0].set_xticks([]); axs[1, 0].set_yticks([])
    xm_rgb = np.zeros((*xenium_masks_transformed.shape, 3), np.uint8); xm_rgb[..., 1] = img_xenium_masks
    axs[1, 1].imshow(xm_rgb); axs[1, 1].set_title("Xenium masks"); axs[1, 1].set_xticks([]); axs[1, 1].set_yticks([])
    cm_rgb = np.zeros((*img.shape, 3), np.uint8); cm_rgb[..., 0] = img_zstack_masks; cm_rgb[..., 1] = img_xenium_masks
    axs[1, 2].imshow(cm_rgb); axs[1, 2].set_title("overlap masks"); axs[1, 2].set_xticks([]); axs[1, 2].set_yticks([])

    plt.tight_layout()
    sec = out_path.stem.split("_")[1]
    plt.suptitle(f"Section {sec} alignment QC", fontsize=16)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def register_section(cfg, sec, M, z_base, R_3d, zstack_r, zstack_masks_r, out_dir, rng_seed=None,
                     num_cpus=None):
    """Fine mask-based tile correlation -> TPS warp -> probability cell matching, for one section,
    writing the full canonical output layout under `out_dir`. `M`/`z_base` = this section's
    converged 2D affine + plane (from `chain_refine.run_chain`/`refine_section`, or the anchor's
    own pose for the anchor section itself). `zstack_r`/`zstack_masks_r` = the z-stack intensity/
    segmentation volumes, ALREADY rotated by `R_3d` (rotate once, reuse for every section -- see
    `chain_refine.run_chain`'s docstring). `num_cpus`: forwarded to the tile correlation's
    `ProcessPoolExecutor` -- see `resources.resolve_num_cpus`."""
    out_dir = Path(out_dir)
    aff_dir = out_dir / "Affine matrices"; aff_dir.mkdir(parents=True, exist_ok=True)
    xen_t_dir = out_dir / "Xenium_affine_transformed"; xen_t_dir.mkdir(parents=True, exist_ok=True)
    warp_dir = out_dir / "warped_zstacks"; warp_dir.mkdir(parents=True, exist_ok=True)
    paw_dir = out_dir / "post_affine_warping"; paw_dir.mkdir(parents=True, exist_ok=True)
    qc_sub = out_dir / "QC" / "xenium_affine_zstack"; qc_sub.mkdir(parents=True, exist_ok=True)
    cmp_dir = out_dir / "cell_matching_probability"; cmp_dir.mkdir(parents=True, exist_ok=True)

    xen_masks = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"))
    outline_path = cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_outline_aligned.tif"
    xen_outline = np.squeeze(tiff.imread(outline_path)) if outline_path.exists() else (xen_masks > 0).astype(np.uint8)
    xen_img_raw = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_images" / f"section_{sec}_Neurons_aligned.tif"))
    xen_img_dtype = xen_img_raw.dtype
    xen_img = xen_img_raw.astype(np.float32)

    xen_img_t = apply_affine_based_on_reference_2d(xen_img, M, zstack_r.shape[1:], order=0)
    xen_masks_t = apply_affine_based_on_reference_2d(xen_masks, M, zstack_r.shape[1:], order=0)
    xen_outline_t = apply_affine_based_on_reference_2d(xen_outline, M, zstack_r.shape[1:], order=0)

    np.save(aff_dir / f"section_{sec}_affine_matrix.npy", np.concatenate([M, [[0, 0, z_base]]], axis=0))
    np.save(aff_dir / f"section_{sec}_rotation_3d.npy", R_3d)

    tiff.imwrite(xen_t_dir / f"section_{sec}_Neurons_transformed.tif", xen_img_t.astype(xen_img_dtype))
    tiff.imwrite(xen_t_dir / f"section_{sec}_Masks_transformed.tif", xen_masks_t.astype(np.uint32))
    tiff.imwrite(xen_t_dir / f"section_{sec}_Masks_outline_transformed.tif", xen_outline_t.astype(xen_outline.dtype))

    xen_masks_bin = (xen_masks_t > 0).astype(np.uint8)
    zstack_masks_bin = (zstack_masks_r > 0).astype(np.uint8)
    zyx_xen, zyx_zst, best_corrs, _ = tile_based_warping(
        xen_masks_bin, zstack_masks_bin, z_base=z_base, margin=MARGIN,
        tile_size=FINE_TILE_SIZE, overlap=FINE_OVERLAP, max_shift=FINE_MAX_SHIFT, num_cpus=num_cpus)
    best_corrs = np.nan_to_num(np.array(best_corrs))

    table = _post_affine_table(zyx_xen, zyx_zst, best_corrs)
    table.to_csv(paw_dir / f"section_{sec}.csv", index=False, header=False)

    xyz_zst = zyx_zst[:, ::-1]
    xyz_xen = zyx_xen[:, ::-1]
    warped_plane, warped_masks = _tps_warp_plane(xyz_xen, xyz_zst, best_corrs, zstack_r, zstack_masks_r, z_base)
    tiff.imwrite(warp_dir / f"section_{sec}_zstack_warped_plane.tif", warped_plane)
    tiff.imwrite(warp_dir / f"section_{sec}_zstack_warped_masks_plane.tif", warped_masks)

    save_alignment_qc(qc_sub / f"section_{sec}_xenium_affine_zstack.png",
                      warped_plane, xen_img_t, warped_masks, xen_masks_t)

    rng = np.random.default_rng(rng_seed if rng_seed is not None else 1000 + sec)
    matching_df = match_section_probability(xen_t_dir, warp_dir, cmp_dir, sec, rng=rng)
    n_valid = int(matching_df["valid"].sum())
    return dict(sec=sec, z_base=int(z_base), n_tiles=len(best_corrs),
               n_tiles_ok=int((best_corrs > 0.2).sum()), n_valid_matches=n_valid)
