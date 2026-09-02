"""
Pose-seeding: three ways to get the rough starting pose (center, rotation, scale) for the initial
landmark search at a subject's anchor Xenium section. Only mode 1 was ever real, wired code before
this package; modes 2/3 formalize what was otherwise done ad hoc, by hand, for one subject
(833855) into first-class functions.

Mode 1 (`seed_from_auto_search`): the blind rotation x position x depth pose grid
(`initial_match.search_anchor_section`) -- fully automatic, no human input, but a narrow/isolated
optimum for some subjects (see `s07_k_nbest_sensitivity` investigation) and can fail outright if
the true pose sits outside the searched position window (833855's original failure mode).

Mode 2 (`seed_from_center_rotation` + `refine_from_seed`): a human provides a rough center
(Xenium-aligned frame, um) and rotation (deg) -- eyeballed from a confocal/vasculature image, or
however the experimenter already has -- and only a DEPTH sweep runs automatically from there (no
position/rotation grid). This is the exact procedure validated on 833855 (n_cert=719, the
strongest result of any subject tested) after the blind grid failed for it.

Mode 3 (`seed_from_corners`): NOT YET IMPLEMENTED, see its docstring for why.
"""
from dataclasses import dataclass
import numpy as np

from . import somaprint as sp2, bigwarp, ZSTACK_XY_UM, XENIUM_S2_UM
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
    scale: float = 0.80


def seed_from_auto_search(cfg, sec, max_workers=14, verbose=True):
    """Mode 1: fully automatic. Runs the blind pose grid (`initial_match.search_anchor_section`,
    k_cz=k_xen=30/n_best=10 -- validated in s07_k_nbest_sensitivity #3-4), then re-derives the
    final tilt via `tilt_fit.fit_tilt_and_landmarks` (the validated SLAB_HALF=30 accumulate+
    bijective procedure -- `search_anchor_section`'s OWN embedded tilt loop is the older,
    SLAB_HALF=20, replace-not-accumulate version; kept there for backward compatibility but not
    used for the final result here).

    Returns dict(M3, R_3d, plane, moving, fixed_aligned, n_landmarks, n_rounds, tilt_deg).
    """
    result = search_anchor_section(cfg, sec, max_workers=max_workers, verbose=verbose)
    if result is None:
        raise RuntimeError(f"[{cfg.subject_id}] sec{sec}: automatic pose grid found nothing "
                          f"plausible -- try seed_from_center_rotation with a human-provided pose")
    M3 = result["M_aligned"][:3, :3] if result["M_aligned"].shape == (4, 3) else result["M_aligned"]
    return fit_tilt_and_landmarks(cfg, sec, M3, result["plane"], verbose=verbose)


def seed_from_center_rotation(center_um, rotation_deg, scale=0.80):
    """Mode 2, step 1: package a human-provided rough pose. Call `refine_from_seed` next."""
    return PoseSeed(center_um=tuple(center_um), rotation_deg=float(rotation_deg), scale=float(scale))


def refine_from_seed(cfg, sec, seed, plane_range=None, verbose=True):
    """Mode 2, step 2: given a `PoseSeed` (rough center+rotation+scale, no position/rotation
    search), sweep ONLY depth (candidate z-stack slabs) to find the plane that actually certifies
    landmarks, then hand off to `tilt_fit.fit_tilt_and_landmarks` for the real tilt fit + full
    landmark accumulation. This is the exact 833855 procedure, generalized.

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
        slab = cz_xy_um[(cz_pl >= st) & (cz_pl < st + 2 * SLAB_HALF)]
        if len(slab) < 20:
            continue
        seed_fn = make_seed(seed.rotation_deg, slab.mean(0), np.asarray(seed.center_um), seed.scale)
        res = sp2.register_full(slab, xen_xy, seed_fn, R_cand=R_CAND, k_cz=K, k_xen=K, n_best=N_BEST, max_rounds=3)
        if len(res["anchors"]) < 6:
            log(f"[{cfg.subject_id}] sec{sec} depth sweep: plane~{st + SLAB_HALF} n_anchors={len(res['anchors'])}")
            continue
        M0 = find_affine_transformation_2d(slab[res["anchors"][:, 0]], xen_xy[res["anchors"][:, 1]])
        r = sp2.match(slab, xen_xy, lambda p: _aff(M0, p), R_cand=30.0, n_best=N_BEST, anchor_frac=0.8,
                      k_cz=K, k_xen=K, max_rounds=3)
        cert = r["accepted"]
        if len(cert) >= 9:
            cert = sp2.flow_filter(cert, _aff(M0, slab)[cert[:, 0]], xen_xy)
        log(f"[{cfg.subject_id}] sec{sec} depth sweep: plane~{st + SLAB_HALF} n_cert={len(cert)}")
        if best is None or len(cert) > best[1]:
            best = (st, len(cert), M0)

    if best is None or best[1] < 6:
        raise RuntimeError(f"[{cfg.subject_id}] sec{sec}: depth sweep found nothing plausible from "
                          f"the given seed (center={seed.center_um}, rotation={seed.rotation_deg}) "
                          f"-- check the seed, or widen `plane_range`")

    st_best, n_cert_best, M0_best = best
    plane_seed = st_best + SLAB_HALF
    log(f"[{cfg.subject_id}] sec{sec}: depth sweep winner plane={plane_seed} n_cert={n_cert_best}")

    M3_seed = M0_best[:3, :3] if M0_best.shape == (4, 3) else M0_best
    return fit_tilt_and_landmarks(cfg, sec, M3_seed, plane_seed, verbose=verbose)


def seed_from_corners(xenium_trapezoid_corners_um, top_edge, zstack_corners_um=None, scale=0.80):
    """*** TODO -- NOT IMPLEMENTED. ***

    Mode 3: 4 Xenium tissue-trapezoid corners (Xenium ALIGNED frame, um) + `top_edge` (which
    corner/edge is the trapezoid's short/slanted top -- fixes correspondence order/orientation
    against the z-stack's 4 corners) + optional 4 z-stack corners (would default to the canonical
    [0,0]-[700,700]um FOV square, matching how 833855's z-stack side was actually handled: never
    independently clicked, only assumed).

    Why not implemented now: the only historical precedent (833855) never actually exercised a
    real corner-to-corner correspondence -- the z-stack side was assumed canonical, not measured,
    and the *rotation* came from averaging the 4 clicked points' own edge angles (see
    `seed_from_center_rotation`'s docstring / the project history), not from matching to 4
    independent z-stack corners. Generalizing this into a real corners-of-trapezoid <->
    corners-of-zstack correspondence needs to be validated against real click data from more than
    one subject before it's trustworthy -- that data doesn't exist yet.
    """
    raise NotImplementedError(
        "seed_from_corners is not yet implemented -- see its docstring. Use "
        "seed_from_center_rotation + refine_from_seed instead (mode 2), which IS validated "
        "(833855). If you need corner-based pose input, compute center=centroid(corners) and "
        "rotation=mean of the 4 edge angles yourself for now, matching the math historically used "
        "for 833855, and pass those directly to seed_from_center_rotation.")
