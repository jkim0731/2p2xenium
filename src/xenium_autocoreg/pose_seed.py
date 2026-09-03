"""
Pose-seeding: three ways to get the rough starting pose (center, rotation, scale) for the initial
landmark search at a subject's anchor Xenium section.

Mode 1 (`seed_from_auto_search`): the blind rotation x position x depth pose grid
(`initial_match.search_anchor_section`) -- fully automatic, no human input, but can be a narrow,
isolated optimum, and can fail outright if the true pose sits outside the searched position
window (see its own docstring).

Mode 2 (`seed_from_center_rotation` + `refine_from_seed`): a human provides a rough center
(Xenium-aligned frame, um) and rotation (deg) -- eyeballed from a confocal/vasculature image, or
however already available -- and only a DEPTH sweep runs automatically from there (no
position/rotation grid). Use this when mode 1 fails.

Mode 3 (`seed_from_corners`): NOT YET IMPLEMENTED, see its docstring for why.
"""
from dataclasses import dataclass
import numpy as np

from . import somaprint as sp2, bigwarp
from .geometry import find_affine_transformation_2d
from .populations import load_zstack_cells, load_xenium_cells
from .reference_ported import find_min_z_spread_rotation
from .initial_match import make_seed, _aff, K, N_BEST, SLAB_HALF, R_CAND, search_anchor_section
from .tilt_fit import fit_tilt_and_landmarks

DEPTH_SWEEP_PLANES = list(range(60, 341, 20))


@dataclass
class PoseSeed:
    center_um: tuple
    rotation_deg: float
    scale: float


def seed_from_auto_search(cfg, sec, num_cpus=None, verbose=True):
    """Mode 1: fully automatic. Runs the blind pose grid (`initial_match.search_anchor_section`),
    then re-derives the final tilt via `tilt_fit.fit_tilt_and_landmarks` (an accumulate+bijective
    landmark procedure -- `search_anchor_section`'s own embedded tilt loop is a simpler
    replace-not-accumulate version, kept there for its own self-contained use but not used for the
    final result here). `num_cpus`: see `resources.resolve_num_cpus` (None/0/over-available ->
    auto; 1 -> serial, no multiprocessing).

    Returns dict(M3, R_3d, plane, moving, fixed_aligned, n_landmarks, n_rounds, tilt_deg).
    """
    result = search_anchor_section(cfg, sec, num_cpus=num_cpus, verbose=verbose)
    if result is None:
        raise RuntimeError(f"[{cfg.subject_id}] sec{sec}: automatic pose grid found nothing "
                          f"plausible -- try seed_from_center_rotation with a human-provided pose")
    M3 = result["M_aligned"][:3, :3] if result["M_aligned"].shape == (4, 3) else result["M_aligned"]
    return fit_tilt_and_landmarks(cfg, sec, M3, result["plane"], verbose=verbose)


def seed_from_center_rotation(center_um, rotation_deg, scale=None, cfg=None):
    """Mode 2, step 1: package a human-provided rough pose. Call `refine_from_seed` next.
    `scale`: the z-stack-to-Xenium linear scale factor; defaults to `cfg.zstack_scale_to_Xenium`
    if `cfg` is given, else the package default."""
    if scale is None:
        from . import DEFAULT_ZSTACK_SCALE_TO_XENIUM
        scale = cfg.zstack_scale_to_Xenium if cfg is not None else DEFAULT_ZSTACK_SCALE_TO_XENIUM
    return PoseSeed(center_um=tuple(center_um), rotation_deg=float(rotation_deg), scale=float(scale))


def refine_from_seed(cfg, sec, seed, plane_range=None, verbose=True):
    """Mode 2, step 2: given a `PoseSeed` (rough center+rotation+scale, no position/rotation
    search), sweep ONLY depth (candidate z-stack slabs) to find the plane that actually certifies
    landmarks, then hand off to `tilt_fit.fit_tilt_and_landmarks` for the real tilt fit + full
    landmark accumulation.

    Returns dict(M3, R_3d, plane, moving, fixed_aligned, n_landmarks, n_rounds, tilt_deg),
    same shape as `seed_from_auto_search`.
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    planes = plane_range if plane_range is not None else DEPTH_SWEEP_PLANES
    ids_cz, cz_xy_um, cz_pl = load_zstack_cells(cfg)
    ids_x, xen_xy, used_rep, n_total = load_xenium_cells(cfg, sec, aligned=True, min_count=2)

    best = None
    for st in planes:
        in_slab = (cz_pl >= st) & (cz_pl < st + 2 * SLAB_HALF)
        slab, pl_slab = cz_xy_um[in_slab], cz_pl[in_slab]
        if len(slab) < 20:
            continue
        seed_fn = make_seed(seed.rotation_deg, slab.mean(0), np.asarray(seed.center_um), seed.scale)
        res = sp2.register_full(slab, xen_xy, seed_fn, R_cand=R_CAND, k_cz=K, k_xen=K, n_best=N_BEST, max_rounds=3)
        if len(res["anchors"]) < 6:
            log(f"[{cfg.subject_id}] sec{sec} depth sweep: plane~{st + SLAB_HALF} n_anchors={len(res['anchors'])}")
            continue
        # M0 here maps z-stack(x,y) -> Xenium(x,y) (fit directly on the um point clouds, for
        # internal soma-print matching use only) -- NOT the same convention as the official
        # M3 (Xenium(row,col) -> z-stack(row,col), from bigwarp.affine_from_landmarks) that
        # fit_tilt_and_landmarks expects. Rebuild the proper M3 from the certified landmarks
        # below rather than reusing M0 directly -- passing M0 as-is silently breaks the
        # downstream tilt fit (wrong direction AND wrong coordinate order).
        M0 = find_affine_transformation_2d(slab[res["anchors"][:, 0]], xen_xy[res["anchors"][:, 1]])
        r = sp2.match(slab, xen_xy, lambda p: _aff(M0, p), R_cand=30.0, n_best=N_BEST, anchor_frac=0.8,
                      k_cz=K, k_xen=K, max_rounds=3)
        cert = r["accepted"]
        if len(cert) >= 9:
            cert = sp2.flow_filter(cert, _aff(M0, slab)[cert[:, 0]], xen_xy)
        log(f"[{cfg.subject_id}] sec{sec} depth sweep: plane~{st + SLAB_HALF} n_cert={len(cert)}")
        if best is None or len(cert) > best[1]:
            best = (st, len(cert), slab, pl_slab, cert)

    if best is None or best[1] < 6:
        raise RuntimeError(f"[{cfg.subject_id}] sec{sec}: depth sweep found nothing plausible from "
                          f"the given seed (center={seed.center_um}, rotation={seed.rotation_deg}) "
                          f"-- check the seed, or widen `plane_range`")

    st_best, n_cert_best, slab_best, pl_best, cert_best = best
    log(f"[{cfg.subject_id}] sec{sec}: depth sweep winner plane~{st_best + SLAB_HALF} n_cert={n_cert_best}")

    real_z = pl_best[cert_best[:, 0]]
    moving = np.column_stack([slab_best[cert_best[:, 0]] / cfg.zstack_xy_um, real_z])
    fixed_aligned = np.column_stack([xen_xy[cert_best[:, 1]] / cfg.xenium_xy_um, np.zeros(len(cert_best))])
    M_aligned, plane_seed = bigwarp.affine_from_landmarks(moving, fixed_aligned)
    M3_seed = M_aligned[:3, :3] if M_aligned.shape == (4, 3) else M_aligned
    return fit_tilt_and_landmarks(cfg, sec, M3_seed, plane_seed, verbose=verbose)


def seed_from_corners(xenium_trapezoid_corners_um, top_edge, zstack_corners_um=None, scale=None):
    """*** TODO -- NOT IMPLEMENTED. ***

    Mode 3: 4 Xenium tissue-trapezoid corners (Xenium ALIGNED frame, um) + `top_edge` (which
    corner/edge is the trapezoid's short/slanted top -- fixes correspondence order/orientation
    against the z-stack's 4 corners) + optional 4 z-stack corners (would default to the z-stack's
    own canonical FOV rectangle).

    Why not implemented: there is no validated real-data precedent yet for a genuine
    corner-to-corner correspondence between the two modalities (as opposed to deriving a rough
    center+rotation from one side's clicked points alone, which `seed_from_center_rotation`
    already covers). Implementing this properly needs real click data validated across more than
    one subject before the correspondence-order/orientation logic can be trusted.
    """
    raise NotImplementedError(
        "seed_from_corners is not yet implemented -- see its docstring. Use "
        "seed_from_center_rotation + refine_from_seed instead (mode 2). If you need corner-based "
        "pose input today, compute center=centroid(corners) and rotation=mean of the 4 edge "
        "angles yourself and pass those directly to seed_from_center_rotation.")
