"""Chained cross-section propagation: for each section, sequentially, seed from the PREVIOUS
section's own converged affine (offsetting z_base by the estimated per-section plane step), then
iteratively refine the affine by tile-based image correlation against the z-stack volume, composing
a correction affine each round until correlation quality stops improving, with each candidate
correction's scale clipped via SVD -- without this, a weak-signal section can drift into an
unphysical anisotropic-scale affine that then propagates to every section downstream."""
import json
import numpy as np
import tifffile as tiff
from pathlib import Path
from .geometry import apply_affine_based_on_reference_2d, find_affine_transformation_2d
from .tile_warp import tile_based_warping_parallel as tile_based_warping
from .reference_ported import rotate_volume

MARGIN, TILE_SIZE, OVERLAP, MAX_SHIFT = 5, (64, 64), 0.2, (30, 100, 100)
Z_STEP = 20
MAX_ITERS = 8
SCALE_CLIP = (0.95, 1.05)


def clip_affine_scale(M, lo=SCALE_CLIP[0], hi=SCALE_CLIP[1]):
    """Clip a candidate affine's 2x2 linear block to +/-5% scale via SVD, preserving its
    rotation/shear. Without this, a weak-signal section can drift into an unphysical
    anisotropic-scale affine that then propagates to every section downstream. Operates on a
    copy; does not mutate `M`."""
    M = M.copy()
    U, S, Vt = np.linalg.svd(M[:2, :2])
    S_clipped = np.clip(S, lo, hi)
    M[:2, :2] = U @ np.diag(S_clipped) @ Vt
    return M


def refine_section(cfg, sec, M_base, z_base, zstack, num_cpus=None):
    mz = MAX_SHIFT[0]
    if not (mz <= z_base < zstack.shape[0] - mz):
        raise ValueError(f"z_base={z_base} outside the usable z-stack range "
                         f"[{mz}, {zstack.shape[0] - mz}) -- section {sec} is past the acquired depth")
    xen_path = cfg.aligned_dir / "Xenium_images" / f"section_{sec}_Neurons_aligned.tif"
    xenium_img = np.squeeze(tiff.imread(xen_path)).astype(np.float32)

    M = M_base.copy()
    high_corr_tiles_old, mean_corr_old = -1, -1
    high_corr_tiles, mean_corr = 0, 0
    history = []
    for i in range(MAX_ITERS):
        xen_t = apply_affine_based_on_reference_2d(xenium_img, M, zstack.shape[1:], order=0)
        zyx_xen, zyx_zst, best_corrs, base_corrs = tile_based_warping(
            xen_t, zstack, z_base=z_base, margin=MARGIN, tile_size=TILE_SIZE,
            overlap=OVERLAP, max_shift=MAX_SHIFT, num_cpus=num_cpus, progress=False)
        best_corrs = np.nan_to_num(np.array(best_corrs))
        z_base = int(round(zyx_zst[:, 0].mean()))
        corr_threshold = max(float(np.percentile(best_corrs, 75)), 0.3)
        pts_zst = zyx_zst[:, 1:][best_corrs > corr_threshold]
        pts_xen = zyx_xen[:, 1:][best_corrs > corr_threshold]
        high_corr_tiles_old, mean_corr_old = high_corr_tiles, mean_corr
        high_corr_tiles = int((best_corrs > corr_threshold).sum())
        mean_corr = float(best_corrs.mean())
        history.append(dict(iter=i + 1, z_base=z_base, n_tiles_ok=high_corr_tiles,
                            mean_corr=round(mean_corr, 3), corr_threshold=round(corr_threshold, 3)))
        print(f"    [sec{sec}] iter {i+1}: z_base={z_base} mean_corr={mean_corr:.3f} "
              f"tiles_ok={high_corr_tiles} (thr={corr_threshold:.2f})", flush=True)
        if high_corr_tiles <= high_corr_tiles_old and mean_corr <= mean_corr_old:
            print(f"    [sec{sec}] no improvement -- converged", flush=True)
            break
        if len(pts_zst) < 6:
            print(f"    [sec{sec}] too few matched points -- stopping", flush=True)
            break
        M_new = find_affine_transformation_2d(pts_xen, pts_zst)
        M_new = clip_affine_scale(M_new)
        M = M_new @ M
    return M, z_base, history


def run_chain(cfg, anchor_sec, anchor_M, anchor_z, out_dir, sections=None, R_3d=None, num_cpus=None):
    """Refine the anchor section, then chain forward and backward across `sections` (default: all
    sections known for this subject). Writes section_{N}_affine.npy / _result.json to `out_dir`.

    `R_3d`: the 3x3 tilt-correction rotation fit once at the anchor (from the initial landmarks'
    z-spread, `initial_match.search_anchor_section`'s `find_min_z_spread_rotation`) and held fixed
    for every section, matching the reference (`xenium_automatic_coreg.ipynb` cell 4): the z-stack
    volume is rotated once, not the Xenium section, and every section (anchor + propagated) is
    refined against that same rotated volume. None/identity reproduces the untilted behavior.

    `num_cpus`: forwarded to every `refine_section` call's tile correlation (the SAME
    `ProcessPoolExecutor` this function's per-section, per-iteration tile-based warping spawns --
    up to `MAX_ITERS` times per section). See `resources.resolve_num_cpus`."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    zstack = tiff.imread(cfg.zstack_registered_tif)
    print(f"[{cfg.subject_id}] z-stack loaded: {zstack.shape}", flush=True)
    if R_3d is not None and not np.allclose(R_3d, np.eye(3)):
        print(f"[{cfg.subject_id}] applying tilt correction (R_3d diag={np.diag(R_3d).round(4).tolist()})", flush=True)
        zstack = rotate_volume(zstack, R_3d, order=1)

    all_sections = sections if sections is not None else cfg.sections
    forward = [s for s in all_sections if s > anchor_sec]
    backward = [s for s in all_sections if s < anchor_sec][::-1]

    def save(sec, M, z_base, history, seed_desc):
        np.save(out_dir / f"section_{sec}_affine.npy", M)
        rec = dict(sec=sec, seed=seed_desc, z_base=z_base, n_iters=len(history), history=history)
        json.dump(rec, open(out_dir / f"section_{sec}_result.json", "w"), indent=2)
        return rec

    print(f"\n=== section {anchor_sec} (anchor seed) ===", flush=True)
    M, z_base, history = refine_section(cfg, anchor_sec, anchor_M, anchor_z, zstack, num_cpus=num_cpus)
    results = [save(anchor_sec, M, z_base, history, "initial-match seed")]

    for step_sign, seq in ((+1, forward), (-1, backward)):
        M_cur, z_cur, prev_sec = M, z_base, anchor_sec
        # Per-section plane step, re-estimated as a running mean of observed |Δz_base|/|Δsection|
        # across already-converged steps in THIS direction -- starts at Z_STEP (a starting
        # assumption) before any real data exists. A gap in the actual Xenium section numbers
        # (some subjects are missing sections) is scaled by that many section-steps, instead of
        # being treated as a single normal +/-Z_STEP hop, which would badly under/overshoot the
        # seed for the section right after a gap.
        step_sum, step_n = float(Z_STEP), 1
        for sec in seq:
            n_gap = abs(sec - prev_sec)
            step_per_sec = step_sum / step_n
            seed_z = z_cur + step_sign * step_per_sec * n_gap
            seed_desc = (f"converged section {prev_sec} ({'+' if step_sign > 0 else '-'}"
                        f"{step_per_sec:.1f}/section x {n_gap} section gap)")
            print(f"\n=== section {sec} (seed: {seed_desc}, seed z_base={seed_z:.0f}) ===", flush=True)
            try:
                M_new, z_new, history = refine_section(cfg, sec, M_cur, int(round(seed_z)), zstack,
                                                        num_cpus=num_cpus)
            except ValueError as e:
                print(f"  STOPPING this direction: {e}", flush=True)
                break
            step_sum += abs(z_new - z_cur) / n_gap
            step_n += 1
            M_cur, z_cur, prev_sec = M_new, z_new, sec
            results.append(save(sec, M_cur, z_cur, history, seed_desc))

    results.sort(key=lambda r: r["sec"])
    json.dump(results, open(out_dir / "chain_summary.json", "w"), indent=2)
    print(f"\n[{cfg.subject_id}] chain covers sections: {[r['sec'] for r in results]}", flush=True)
    return results
