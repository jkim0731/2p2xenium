"""
Per-subject QC figure for the initial matched (anchor) section: initial landmark search (top row)
vs. final registration after tilt + fine mask-based TPS (bottom row). Every subject goes through
the same `initial_match/` artifact format
(section_N_{affine,rotation_3d,moving,fixed_aligned}.npy + result.json, produced by
`tilt_fit.fit_tilt_and_landmarks`), so no subject-specific branching is needed here.

Panels B ("ROI zoom") and D ("filled masks") are cropped to 1.1x the z-stack FOV's own footprint
(projected into the Xenium-aligned frame via the fit affine) -- not a tight zoom into a handful of
cells. The bottom row shows the same region after the fitted 2D affine + 3D tilt + fine mask-TPS
warp -- the actual final registration, already living in one shared (z-stack-shaped) canvas, so
"1.1x FOV" there is just that canvas plus a 10% margin.

Affine convention (see geometry.py/bigwarp.py): M maps Xenium (row,col) -> z-stack (row,col).
"""
from pathlib import Path
import json
import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from skimage.segmentation import find_boundaries

ZOOM = 1.1
SLAB_HALF_SLICE = 20   # z-stack slab half-thickness (planes) for the "filled masks" segmentation slab


def crop_project(src2d, M_xen_to_z, by0, bx0, Hc, Wc, order=0, cval=0):
    """Sample src2d (z-stack-frame image) onto an Hc x Wc canvas in Xenium-aligned row/col space
    [by0:by0+Hc, bx0:bx0+Wc], using M (Xenium(rc) -> z-stack(rc)) directly (un-inverted)."""
    rows, cols = np.indices((Hc, Wc))
    pts = np.vstack([(rows + by0).ravel(), (cols + bx0).ravel(), np.ones(Hc * Wc)])
    src_rc = (M_xen_to_z @ pts)[:2]
    out = map_coordinates(src2d.astype(np.float32), src_rc, order=order, mode="constant", cval=cval)
    return out.reshape(Hc, Wc)


def fov_crop_box(M3, zstack_hw, zoom=ZOOM):
    """1.1x (default) bounding box, in Xenium-aligned (row,col), of the z-stack FOV's own
    footprint (its 4 corners) projected into the Xenium-aligned frame via M3_inv."""
    H, W = zstack_hw
    corners_z_rc = np.array([[0, 0], [0, W], [H, W], [H, 0]], float)
    M_inv = np.linalg.inv(M3)
    corners_xen_rc = (M_inv @ np.column_stack([corners_z_rc, np.ones(4)]).T).T[:, :2]
    r0, r1 = corners_xen_rc[:, 0].min(), corners_xen_rc[:, 0].max()
    c0, c1 = corners_xen_rc[:, 1].min(), corners_xen_rc[:, 1].max()
    rc_, cc_ = (r0 + r1) / 2, (c0 + c1) / 2
    hh, hw = (r1 - r0) / 2 * zoom, (c1 - c0) / 2 * zoom
    return int(rc_ - hh), int(cc_ - hw), int(rc_ + hh), int(cc_ + hw), corners_xen_rc


def map_zstack_pts_to_xen(xy_zstack, M3):
    M_inv = np.linalg.inv(M3)
    rc = xy_zstack[:, ::-1]
    mapped_rc = (M_inv @ np.column_stack([rc, np.ones(len(rc))]).T).T[:, :2]
    return mapped_rc[:, ::-1]


def zstack_mask_slab(seg3d, plane, half=SLAB_HALF_SLICE):
    z_lo, z_hi = max(0, plane - half), min(seg3d.shape[0] - 1, plane + half)
    return (seg3d[z_lo:z_hi + 1] > 0).max(0).astype(np.uint8)


def _norm01(a, plow=1, phigh=99):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [plow, phigh])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def plot_initial_match_qc(cfg, sec, anchor_dir, results_dir, out_path):
    """cfg: SubjectConfig. sec: anchor section number. anchor_dir: folder holding
    section_{sec}_{affine,rotation_3d,moving,fixed_aligned}.npy + result.json (tilt_fit.
    fit_tilt_and_landmarks's output format). results_dir: the subject's materialized results root
    (must already contain Xenium_affine_transformed/ and warped_zstacks/ for `sec`, i.e.
    fine_registration must have already run)."""
    anchor_dir, results_dir, out_path = Path(anchor_dir), Path(results_dir), Path(out_path)
    M3 = np.load(anchor_dir / f"section_{sec}_affine.npy")
    moving = np.load(anchor_dir / f"section_{sec}_moving.npy")[:, :2]
    fixed_xy = np.load(anchor_dir / f"section_{sec}_fixed_aligned.npy")[:, :2]
    plane = json.load(open(anchor_dir / f"section_{sec}_result.json"))["plane"]

    neurons = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_images" / f"section_{sec}_Neurons_aligned.tif"))
    xmask = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"))
    seg3d = tiff.imread(cfg.zstack_segmented_tif)
    zstack_hw = seg3d.shape[1:]

    by0, bx0, by1, bx1, corners_xen_rc = fov_crop_box(M3, zstack_hw)
    cxy = corners_xen_rc[:, ::-1]
    mapped_xy = map_zstack_pts_to_xen(moving, M3)
    slab_bin = zstack_mask_slab(seg3d, plane)

    xen_img_t = tiff.imread(results_dir / "Xenium_affine_transformed" / f"section_{sec}_Neurons_transformed.tif")
    xen_masks_t = tiff.imread(results_dir / "Xenium_affine_transformed" / f"section_{sec}_Masks_transformed.tif")
    zst_warp_plane = tiff.imread(results_dir / "warped_zstacks" / f"section_{sec}_zstack_warped_plane.tif")
    zst_warp_masks = tiff.imread(results_dir / "warped_zstacks" / f"section_{sec}_zstack_warped_masks_plane.tif")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12.4))

    # --- Row 1: initial landmark search ---
    axA = axes[0, 0]
    disp = np.clip(neurons.astype(np.float32), 0, np.percentile(neurons, 99.5))
    axA.imshow(disp, cmap="gray", origin="upper")
    axA.plot(np.r_[cxy[:, 0], cxy[0, 0]], np.r_[cxy[:, 1], cxy[0, 1]], "-", color="cyan", lw=2)
    inw_full = (fixed_xy[:, 0] >= bx0) & (fixed_xy[:, 0] < bx1) & (fixed_xy[:, 1] >= by0) & (fixed_xy[:, 1] < by1)
    axA.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor="lime", lw=1.5))
    axA.scatter(fixed_xy[:, 0], fixed_xy[:, 1], s=3, c="yellow")
    axA.set_title(f"A) section {sec} + z-stack FOV footprint, plane {plane}, {len(fixed_xy)} pairs\n"
                  f"(cyan=z-stack FOV here; green=1.1x crop for B/D)", weight="bold", fontsize=10)
    axA.set_xlabel("Xenium-aligned x (px)"); axA.set_ylabel("Xenium-aligned y (px)")

    axB = axes[0, 1]
    sub = neurons[by0:by1, bx0:bx1]
    axB.imshow(np.clip(sub, 0, np.percentile(sub, 99.5)) if sub.size else sub, cmap="gray", extent=[bx0, bx1, by1, by0])
    submask = xmask[by0:by1, bx0:bx1]
    yb, xb = np.where(find_boundaries(submask, mode="inner"))
    axB.scatter(xb + bx0, yb + by0, s=0.3, c="tab:blue", alpha=0.3)
    inw = inw_full
    for (xx, yy), (mx, my) in zip(fixed_xy[inw], mapped_xy[inw]):
        axB.plot([xx, mx], [yy, my], "-", color="red", lw=0.6, alpha=0.6)
    axB.scatter(fixed_xy[inw, 0], fixed_xy[inw, 1], s=20, facecolors="none", edgecolors="yellow", lw=1.0,
               label="Xenium cell (certified)")
    axB.scatter(mapped_xy[inw, 0], mapped_xy[inw, 1], s=14, c="red", marker="x", lw=1.0,
               label="z-stack -> Xenium-aligned")
    axB.set_xlim(bx0, bx1); axB.set_ylim(by1, by0); axB.legend(loc="lower right", fontsize=7)
    dres_um = np.linalg.norm(mapped_xy - fixed_xy, axis=1) * cfg.xenium_xy_um
    med_res = np.median(dres_um[inw]) if inw.sum() else float("nan")
    axB.set_title(f"B) ROI (1.1x FOV), {int(inw.sum())} pairs in view\nmedian residual {med_res:.1f} um", weight="bold")
    axB.set_xlabel("Xenium-aligned x (px)")

    axD = axes[0, 2]
    xen_bin = (submask > 0).astype(np.uint8)
    zs_warp = crop_project(slab_bin, M3, by0, bx0, by1 - by0, bx1 - bx0, order=0)
    rgb = np.zeros((by1 - by0, bx1 - bx0, 3), np.float32)
    rgb[xen_bin > 0] = [0.0, 1.0, 1.0]; rgb[zs_warp > 0] = [1.0, 0.0, 1.0]
    rgb[(xen_bin > 0) & (zs_warp > 0)] = [1.0, 1.0, 1.0]
    axD.imshow(rgb, origin="upper", extent=[bx0, bx1, by1, by0])
    axD.set_xlim(bx0, bx1); axD.set_ylim(by1, by0)
    n_xen, n_zs = int(xen_bin.sum()), int(zs_warp.sum()); n_ov = int(((xen_bin > 0) & (zs_warp > 0)).sum())
    iou = n_ov / max(1, (n_xen + n_zs - n_ov))
    axD.set_title(f"D) filled masks (cyan=Xenium, magenta=z-stack), 1.1x FOV\nIoU={iou:.2f}", weight="bold")
    axD.set_xlabel("Xenium-aligned x (px)")

    # --- Row 2: final registration (already in the shared z-stack-sized canvas) ---
    Hc, Wc = zst_warp_plane.shape
    pad_h, pad_w = int(Hc * (ZOOM - 1) / 2), int(Wc * (ZOOM - 1) / 2)

    axA2 = axes[1, 0]
    axA2.imshow(np.clip(zst_warp_plane.astype(np.float32), 0, np.percentile(zst_warp_plane, 99.5)), cmap="gray")
    axA2.add_patch(plt.Rectangle((0, 0), Wc, Hc, fill=False, edgecolor="orange", lw=1.5))
    axA2.set_xlim(-pad_w, Wc + pad_w); axA2.set_ylim(Hc + pad_h, -pad_h)
    axA2.set_title("A') final z-stack plane (shared canvas)\norange = its native extent", weight="bold")
    axA2.set_xlabel("z-stack-canvas x (px)"); axA2.set_ylabel("z-stack-canvas y (px)")

    axB2 = axes[1, 1]
    r = _norm01(zst_warp_plane); g = _norm01(xen_img_t)
    merge = np.zeros((*r.shape, 3), np.float32); merge[..., 0] = r; merge[..., 1] = g
    axB2.imshow(merge)
    axB2.set_xlim(-pad_w, Wc + pad_w); axB2.set_ylim(Hc + pad_h, -pad_h)
    axB2.set_title("B') final overlay (red=z-stack, green=Xenium), 1.1x FOV", weight="bold")
    axB2.set_xlabel("z-stack-canvas x (px)")

    axD2 = axes[1, 2]
    zb = (zst_warp_masks > 0).astype(np.uint8); xb2 = (xen_masks_t > 0).astype(np.uint8)
    rgb2 = np.zeros((*zb.shape, 3), np.float32)
    rgb2[xb2 > 0] = [0.0, 1.0, 1.0]; rgb2[zb > 0] = [1.0, 0.0, 1.0]; rgb2[(xb2 > 0) & (zb > 0)] = [1.0, 1.0, 1.0]
    axD2.imshow(rgb2)
    axD2.set_xlim(-pad_w, Wc + pad_w); axD2.set_ylim(Hc + pad_h, -pad_h)
    n_ov2 = int(((xb2 > 0) & (zb > 0)).sum()); iou2 = n_ov2 / max(1, int((xb2 > 0).sum() + (zb > 0).sum() - n_ov2))
    axD2.set_title(f"D') final filled masks, 1.1x FOV\nIoU={iou2:.2f}", weight="bold")
    axD2.set_xlabel("z-stack-canvas x (px)")

    fig.suptitle(f"{cfg.subject_id} -- section {sec} initial match (top) vs. final registration (bottom)",
                fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
