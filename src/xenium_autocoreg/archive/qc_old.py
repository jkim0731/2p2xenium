"""Three QC figures for the pipeline: (1) initial match cell-level zoom, (2) chain-refined
propagation across sections at one fixed crop, (3) fine-registration matched-cell contours.
No GT overlay -- most subjects run through this pipeline have none."""
import json
import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
import pandas as pd
from coreg import ZSTACK_XY_UM, XENIUM_S2_UM
from .populations import load_zstack_cells, load_xenium_cells

HW = 170          # half-width of the cell-level crop, in Xenium-aligned px
SLAB_HALF = 20
NN_MATCH_UM = 15.0


def _crop_project(src2d, M_out_to_src, by0, bx0, Hc, Wc, order=0, cval=0):
    rows, cols = np.indices((Hc, Wc))
    pts = np.vstack([(rows + by0).ravel(), (cols + bx0).ravel(), np.ones(Hc * Wc)])
    src_rc = (M_out_to_src @ pts)[:2]
    out = map_coordinates(src2d.astype(np.float32), src_rc, order=order, mode="constant", cval=cval)
    return out.reshape(Hc, Wc)


def qc_initial_match(cfg, anchor_result, out_path):
    sec = anchor_result["sec"]
    M = anchor_result["M_aligned"]
    plane = anchor_result["plane"]
    moving_xy = anchor_result["moving"][:, :2]
    fixed_xy = anchor_result["fixed_aligned"][:, :2]
    M_inv = np.linalg.inv(M)

    seg3d = tiff.imread(cfg.zstack_segmented_tif)
    z_lo, z_hi = max(0, plane - SLAB_HALF), min(seg3d.shape[0] - 1, plane + SLAB_HALF)
    slab_bin = (seg3d[z_lo:z_hi + 1] > 0).max(0).astype(np.uint8)

    neurons = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_images" / f"section_{sec}_Neurons_aligned.tif"))
    xmask = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"))

    corners_z_rc = np.array([[0, 0], [0, 512], [512, 512], [512, 0]], float)
    corners_xen_rc = (M_inv @ np.column_stack([corners_z_rc, np.ones(4)]).T).T[:, :2]
    cxy = corners_xen_rc[:, ::-1]

    mapped_xy = (M_inv @ np.column_stack([moving_xy[:, ::-1], np.ones(len(moving_xy))]).T).T[:, :2][:, ::-1]
    dres_um = np.linalg.norm(mapped_xy - fixed_xy, axis=1) * XENIUM_S2_UM

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    axA = axes[0]
    disp = np.clip(neurons.astype(np.float32), 0, np.percentile(neurons, 99.5))
    axA.imshow(disp, cmap="gray", origin="upper")
    axA.plot(np.r_[cxy[:, 0], cxy[0, 0]], np.r_[cxy[:, 1], cxy[0, 1]], "-", color="cyan", lw=2)
    axA.scatter(fixed_xy[:, 0], fixed_xy[:, 1], s=4, c="yellow")
    axA.set_title(f"A) section {sec} (aligned frame) + z-stack FOV footprint\n"
                 f"plane {plane}, {len(fixed_xy)} certified pairs", weight="bold")
    axA.set_xlabel("Xenium-aligned x (px)"); axA.set_ylabel("Xenium-aligned y (px)")

    cx_, cy_ = np.median(fixed_xy[:, 0]), np.median(fixed_xy[:, 1])
    bx0, by0, bx1, by1 = int(cx_ - HW), int(cy_ - HW), int(cx_ + HW), int(cy_ + HW)
    axB = axes[1]
    sub = neurons[by0:by1, bx0:bx1]
    axB.imshow(np.clip(sub, 0, np.percentile(sub, 99.5)), cmap="gray", extent=[bx0, bx1, by1, by0])
    submask = xmask[by0:by1, bx0:bx1]
    yb, xb = np.where(find_boundaries(submask, mode="inner"))
    axB.scatter(xb + bx0, yb + by0, s=0.5, c="tab:blue", alpha=0.3)
    inw = (fixed_xy[:, 0] >= bx0) & (fixed_xy[:, 0] < bx1) & (fixed_xy[:, 1] >= by0) & (fixed_xy[:, 1] < by1)
    for (xx, yy), (mx, my) in zip(fixed_xy[inw], mapped_xy[inw]):
        axB.plot([xx, mx], [yy, my], "-", color="red", lw=0.8, alpha=0.7)
    axB.scatter(fixed_xy[inw, 0], fixed_xy[inw, 1], s=42, facecolors="none", edgecolors="yellow",
               lw=1.2, label="Xenium cell (certified)")
    axB.scatter(mapped_xy[inw, 0], mapped_xy[inw, 1], s=28, c="red", marker="x", lw=1.2,
               label="z-stack -> Xenium-aligned")
    axB.set_xlim(bx0, bx1); axB.set_ylim(by1, by0); axB.legend(loc="lower right", fontsize=8)
    axB.set_title(f"B) ROI zoom, {inw.sum()} pairs in view\nmedian residual {np.median(dres_um[inw]):.1f} um",
                 weight="bold")
    axB.set_xlabel("Xenium-aligned x (px)")

    axD = axes[2]
    xen_bin = (submask > 0).astype(np.uint8)
    zs_warp = _crop_project(slab_bin, M, by0, bx0, by1 - by0, bx1 - bx0, order=0)
    rgb = np.zeros((by1 - by0, bx1 - bx0, 3), np.float32)
    rgb[xen_bin > 0] = [0.0, 1.0, 1.0]
    rgb[zs_warp > 0] = [1.0, 0.0, 1.0]
    rgb[(xen_bin > 0) & (zs_warp > 0)] = [1.0, 1.0, 1.0]
    axD.imshow(rgb, origin="upper", extent=[bx0, bx1, by1, by0])
    axD.set_xlim(bx0, bx1); axD.set_ylim(by1, by0)
    n_xen = int(xen_bin.sum()); n_zs = int(zs_warp.sum()); n_ov = int(((xen_bin > 0) & (zs_warp > 0)).sum())
    iou = n_ov / max(1, (n_xen + n_zs - n_ov))
    axD.set_title(f"D) filled masks (cyan=Xenium, magenta=z-stack, white=overlap)\nIoU={iou:.2f}", weight="bold")
    axD.set_xlabel("Xenium-aligned x (px)")

    fig.suptitle(f"{cfg.subject_id} INITIAL MATCH -- section {sec}, aligned frame, GT-free soma-print search",
                fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    print("wrote", out_path)


def qc_propagation(cfg, chain_dir, sections_qc, out_path):
    chain_dir = Path(chain_dir)
    seg3d = tiff.imread(cfg.zstack_segmented_tif)
    ids_cz_all, cz_all_xy, cz_all_pl = load_zstack_cells(cfg)

    anchor_sec = sections_qc[len(sections_qc) // 2]
    M0 = np.load(chain_dir / f"section_{anchor_sec}_affine.npy")
    corners_z_rc = np.array([[0, 0], [0, 512], [512, 512], [512, 0]], float)
    corners_xen_rc = (np.linalg.inv(M0) @ np.column_stack([corners_z_rc, np.ones(4)]).T).T[:, :2]
    cx_, cy_ = corners_xen_rc[:, 1].mean(), corners_xen_rc[:, 0].mean()
    bx0, by0, bx1, by1 = int(cx_ - HW), int(cy_ - HW), int(cx_ + HW), int(cy_ + HW)

    fig, axes = plt.subplots(2, len(sections_qc), figsize=(5.5 * len(sections_qc), 11.2))
    for col, sec in enumerate(sections_qc):
        M = np.load(chain_dir / f"section_{sec}_affine.npy")
        z_base = json.load(open(chain_dir / f"section_{sec}_result.json"))["z_base"]
        M_inv = np.linalg.inv(M)

        neurons = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_images" / f"section_{sec}_Neurons_aligned.tif"))
        xmask = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"))
        z_lo, z_hi = max(0, z_base - SLAB_HALF), min(seg3d.shape[0] - 1, z_base + SLAB_HALF)
        slab_bin = (seg3d[z_lo:z_hi + 1] > 0).max(0).astype(np.uint8)

        slab_cells_px = cz_all_xy[(cz_all_pl >= z_base - SLAB_HALF) & (cz_all_pl < z_base + SLAB_HALF)] / ZSTACK_XY_UM
        z_rc = slab_cells_px[:, ::-1]
        xen_rc_mapped = (M_inv @ np.column_stack([z_rc, np.ones(len(z_rc))]).T).T[:, :2]
        _, xen_xy_um, _, _ = load_xenium_cells(cfg, sec, aligned=True, min_count=6)
        xen_px_all = xen_xy_um / XENIUM_S2_UM
        tree = cKDTree(xen_px_all)
        d_px, idx = tree.query(xen_rc_mapped[:, ::-1])
        keep = d_px * XENIUM_S2_UM <= NN_MATCH_UM
        moving_xy = slab_cells_px[keep]
        fixed_xy = xen_px_all[idx[keep]]

        mapped_xy = (M_inv @ np.column_stack([moving_xy[:, ::-1], np.ones(len(moving_xy))]).T).T[:, :2][:, ::-1]
        dres_um = np.linalg.norm(mapped_xy - fixed_xy, axis=1) * XENIUM_S2_UM
        inw = (fixed_xy[:, 0] >= bx0) & (fixed_xy[:, 0] < bx1) & (fixed_xy[:, 1] >= by0) & (fixed_xy[:, 1] < by1)

        ax = axes[0, col]
        sub = neurons[by0:by1, bx0:bx1]
        ax.imshow(np.clip(sub, 0, np.percentile(sub, 99.5)), cmap="gray", extent=[bx0, bx1, by1, by0])
        submask = xmask[by0:by1, bx0:bx1]
        yb, xb = np.where(find_boundaries(submask, mode="inner"))
        ax.scatter(xb + bx0, yb + by0, s=0.5, c="tab:blue", alpha=0.3)
        for (xx, yy), (mx, my) in zip(fixed_xy[inw], mapped_xy[inw]):
            ax.plot([xx, mx], [yy, my], "-", color="red", lw=0.8, alpha=0.7)
        ax.scatter(fixed_xy[inw, 0], fixed_xy[inw, 1], s=42, facecolors="none", edgecolors="yellow",
                  lw=1.2, label="Xenium cell")
        ax.scatter(mapped_xy[inw, 0], mapped_xy[inw, 1], s=28, c="red", marker="x", lw=1.2,
                  label="z-stack -> Xenium-aligned")
        ax.set_xlim(bx0, bx1); ax.set_ylim(by1, by0)
        if col == 0:
            ax.legend(loc="lower right", fontsize=7)
        ax.set_title(f"sec {sec}  plane={z_base}\n{len(fixed_xy[inw])} pairs (viz), "
                    f"median residual {np.median(dres_um[inw]) if inw.sum() else float('nan'):.1f} um",
                    fontsize=8.5, weight="bold")

        ax = axes[1, col]
        xen_bin = (submask > 0).astype(np.uint8)
        zs_warp = _crop_project(slab_bin, M, by0, bx0, by1 - by0, bx1 - bx0, order=0)
        rgb = np.zeros((by1 - by0, bx1 - bx0, 3), np.float32)
        rgb[xen_bin > 0] = [0.0, 1.0, 1.0]
        rgb[zs_warp > 0] = [1.0, 0.0, 1.0]
        rgb[(xen_bin > 0) & (zs_warp > 0)] = [1.0, 1.0, 1.0]
        ax.imshow(rgb, origin="upper", extent=[bx0, bx1, by1, by0])
        ax.set_xlim(bx0, bx1); ax.set_ylim(by1, by0)
        n_xen = int(xen_bin.sum()); n_zs = int(zs_warp.sum()); n_ov = int(((xen_bin > 0) & (zs_warp > 0)).sum())
        iou = n_ov / max(1, (n_xen + n_zs - n_ov))
        ax.set_title(f"IoU={iou:.2f}", fontsize=9)

    fig.text(0.5, 0.87, "filled masks: cyan=Xenium, magenta=z-stack, white=overlap", ha="center", fontsize=9)
    fig.suptitle(f"{cfg.subject_id} CHAIN-REFINED PROPAGATION -- SAME crop location shown at every depth",
                fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    print("wrote", out_path)


def qc_fine_registration(fine_dir, sections_qc, out_path, crop=(180, 340, 180, 340)):
    fine_dir = Path(fine_dir)
    summary = json.load(open(fine_dir / "fine_registration_summary.json"))
    smap = {r["sec"]: r for r in summary}
    y0, y1, x0, x1 = crop

    fig, axes = plt.subplots(2, len(sections_qc), figsize=(5.5 * len(sections_qc), 11))
    for col, sec in enumerate(sections_qc):
        xen_t = tiff.imread(fine_dir / f"section_{sec}_xenium_masks_transformed.tif")
        zst_w = tiff.imread(fine_dir / f"section_{sec}_zstack_warped_masks.tif")
        df = pd.read_csv(fine_dir / f"section_{sec}_cell_matching.csv")
        rec = smap[sec]
        matched_xen_ids = set(df.mask_id_xenium); matched_cz_ids = set(df.mask_id_cz)

        ax = axes[0, col]
        rgb = np.ones((*xen_t.shape, 3), np.float32)
        rgb[xen_t > 0] = [1.0, 0.75, 0.75]
        rgb[zst_w > 0] = [0.6, 0.6, 1.0]
        rgb[(xen_t > 0) & (zst_w > 0)] = [0.6, 0.2, 0.8]
        ax.imshow(rgb, origin="upper")
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="black", lw=1.5))
        ax.set_title(f"sec {sec}  plane={rec['plane']}\n{rec['n_cell_matches']} cell matches, "
                    f"{rec['n_tiles_good']}/{rec['n_tiles']} tiles ok", fontsize=10, weight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, col]
        ax.set_facecolor("white")
        xen_crop = xen_t[y0:y1, x0:x1]; zst_crop = zst_w[y0:y1, x0:x1]
        for lbl in np.unique(xen_crop):
            if lbl == 0:
                continue
            color = "red" if lbl in matched_xen_ids else "salmon"
            alpha = 0.6 if lbl in matched_xen_ids else 0.3
            for c in find_contours(xen_crop == lbl, 0.5):
                ax.fill(c[:, 1] + x0, c[:, 0] + y0, color=color, alpha=alpha, lw=0)
        for lbl in np.unique(zst_crop):
            if lbl == 0:
                continue
            color = "blue" if lbl in matched_cz_ids else "lightblue"
            alpha = 0.6 if lbl in matched_cz_ids else 0.3
            for c in find_contours(zst_crop == lbl, 0.5):
                ax.fill(c[:, 1] + x0, c[:, 0] + y0, color=color, alpha=alpha, lw=0)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
        ax.set_title("zoom: red/blue=matched pair, pale=unmatched", fontsize=9)

    fig.suptitle("FINE REGISTRATION on chain-refined affines -- tile-warp + TPS + IoU cell matching\n"
                "top: full (512,512) z-stack-pixel canvas  |  bottom: zoomed matched/unmatched cells",
                fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    print("wrote", out_path)
