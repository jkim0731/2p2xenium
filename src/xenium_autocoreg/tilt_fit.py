"""
General 3D-tilt-fitting stage, generalizing the validated procedure from
`s09_tilt_refinement_final/final_protocol.py`'s Phase 1 (SLAB_HALF=30, ACCUMULATE landmarks across
rounds with a bijective constraint, iterate until a round finds no new landmarks) into a function
any subject can call, seeded from an ALREADY-FOUND pose rather than re-running the expensive blind
grid search.

This supersedes `initial_match.search_anchor_section`'s own embedded tilt loop for quality (that
one uses the stale SLAB_HALF=20 and REPLACES rather than accumulates each round's landmark set --
see that function's docstring / s09_tilt_refinement_final/SUMMARY.md's "What was and wasn't
changed" section for the exact gap) -- but does NOT replace `search_anchor_section` itself, which
still does the real work of finding the pose in the first place. Call `fit_tilt_and_landmarks`
AFTER you already have a validated (M3, plane) from either the automatic grid search or a
human-provided seed.
"""
import numpy as np
import tifffile as tiff
from . import somaprint as sp2, bigwarp, ZSTACK_XY_UM, XENIUM_S2_UM
from .geometry import find_affine_transformation_2d
from .populations import load_zstack_cells, load_xenium_cells
from .reference_ported import find_min_z_spread_rotation
from .initial_match import make_seed, _aff, K, N_BEST

SLAB_HALF = 30   # validated value (60um-thick slab) -- NOT initial_match.SLAB_HALF (20, stale)
P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)   # xyz<->zyx permutation


def fit_tilt_and_landmarks(cfg, sec, seed_M3, seed_plane, max_rounds=12, verbose=True):
    """Iteratively: match landmarks in a 60um slab around the current plane (seeded from the
    running affine) -> accumulate new (bijective) landmarks -> fit a 3D tilt R_3d from the FULL
    accumulated set -> de-tilt the z-stack population -> refit the affine from de-tilted landmark
    positions -> repeat, until a round finds no new landmarks.

    `seed_M3`/`seed_plane`: an already-validated starting pose (from
    `initial_match.search_anchor_section`, or a human-provided seed) -- this function does NOT
    re-search for the pose, only refines it and fits the tilt.

    Returns dict(M3, R_3d, plane, moving, fixed_aligned, n_landmarks, n_rounds).
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    ids_cz, cz_xy_um, cz_pl = load_zstack_cells(cfg)
    ids_x, xen_xy, used_rep, n_total = load_xenium_cells(cfg, sec, aligned=True, min_count=2)
    id_to_idx_cz = {int(i): k for k, i in enumerate(ids_cz)}

    zstack_shape = tiff.imread(cfg.zstack_segmented_tif).shape
    center = np.array(zstack_shape) / 2.0
    cz_px = cz_xy_um / cfg.zstack_xy_um
    pts_zyx_orig_all = np.column_stack([cz_pl, cz_px[:, 1], cz_px[:, 0]])

    L_cz_ids, L_xen_ids = [], []
    R_3d = np.eye(3)
    M3 = np.asarray(seed_M3, float)
    plane = int(seed_plane)
    ST = plane - SLAB_HALF
    round_i = 0

    while round_i < max_rounds:
        round_i += 1
        R_vol = P @ R_3d @ P
        rot_zyx = (R_vol @ (pts_zyx_orig_all - center).T).T + center
        new_pl_all = rot_zyx[:, 0]
        new_xy_um_all = rot_zyx[:, [2, 1]] * cfg.zstack_xy_um

        in_slab = (new_pl_all >= ST) & (new_pl_all < ST + 2 * SLAB_HALF)
        slab, pl, ids_slab = new_xy_um_all[in_slab], new_pl_all[in_slab], ids_cz[in_slab]

        M3_inv = np.linalg.inv(M3)

        def seed_fn(p, M_inv=M3_inv, um_per_px=cfg.zstack_xy_um):
            p = np.atleast_2d(p) / um_per_px
            rc = np.column_stack([p[:, 1], p[:, 0], np.ones(len(p))])
            xen_rc = (M_inv @ rc.T).T[:, :2]
            return xen_rc[:, ::-1] * XENIUM_S2_UM

        r = sp2.match(slab, xen_xy, seed_fn, R_cand=30.0, n_best=N_BEST, anchor_frac=0.8, k_cz=K, k_xen=K, max_rounds=3)
        cert = r["accepted"]
        if len(cert) >= 9:
            cert = sp2.flow_filter(cert, seed_fn(slab)[cert[:, 0]], xen_xy)

        found_pairs = set(zip(ids_slab[cert[:, 0]].tolist(), ids_x[cert[:, 1]].tolist())) if len(cert) else set()
        existing_pairs = set(zip(L_cz_ids, L_xen_ids))
        used_zids, used_xids = set(L_cz_ids), set(L_xen_ids)
        new_pairs = []
        for z, x in found_pairs - existing_pairs:
            if z in used_zids or x in used_xids:
                continue
            new_pairs.append((z, x)); used_zids.add(z); used_xids.add(x)
        log(f"[sec{sec}] tilt-fit round {round_i}: slab n={len(slab)}, matched {len(cert)} "
            f"({len(new_pairs)} NEW) | accumulated so far = {len(existing_pairs)}")

        if not new_pairs:
            log(f"[sec{sec}] tilt-fit converged after {round_i} round(s), total landmarks = {len(existing_pairs)}")
            break

        for zid, xid in new_pairs:
            L_cz_ids.append(zid); L_xen_ids.append(xid)

        idx_cz = np.array([id_to_idx_cz[z] for z in L_cz_ids])
        moving_raw = np.column_stack([cz_xy_um[idx_cz] / cfg.zstack_xy_um, cz_pl[idx_cz]])
        idx_xen = np.array([np.flatnonzero(ids_x == x)[0] for x in L_xen_ids])
        fixed_aligned = np.column_stack([xen_xy[idx_xen] / XENIUM_S2_UM, np.zeros(len(idx_xen))])
        R_3d, _ = find_min_z_spread_rotation(moving_raw)

        R_vol = P @ R_3d @ P
        m_zyx_raw = np.column_stack([moving_raw[:, 2], moving_raw[:, 1], moving_raw[:, 0]])
        m_zyx_detilted = (R_vol @ (m_zyx_raw - center).T).T + center
        moving = np.column_stack([m_zyx_detilted[:, 2], m_zyx_detilted[:, 1], m_zyx_detilted[:, 0]])
        M_aligned, plane = bigwarp.affine_from_landmarks(moving, fixed_aligned)
        M3 = M_aligned[:3, :3] if M_aligned.shape == (4, 3) else M_aligned
        ST = plane - SLAB_HALF

    tilt_deg = np.degrees(np.arccos(np.clip((R_3d.T @ np.array([0, 0, 1.0]))[2], -1, 1)))
    log(f"[sec{sec}] FINAL: n_landmarks={len(L_cz_ids)}, tilt={tilt_deg:.3f} deg, plane={plane}")
    return dict(M3=M3, R_3d=R_3d, plane=plane, moving=moving, fixed_aligned=fixed_aligned,
               n_landmarks=len(L_cz_ids), n_rounds=round_i, tilt_deg=float(tilt_deg))
