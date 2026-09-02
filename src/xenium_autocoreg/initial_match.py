"""GT-free automatic initial-pose search at one anchor Xenium section: a full grid over
depth-slab x xy-seed x rotation, every combo certified directly via soma-print point matching,
ranked by certified count with a sanity filter on the implied affine (plausible isotropic scale +
rotation range + a minimum match rate) to reject spurious peaks.
"""
import numpy as np
import tifffile as tiff
from concurrent.futures import ProcessPoolExecutor
from . import somaprint as sp2, bigwarp
from .geometry import find_affine_transformation_2d, decompose_affine
from .populations import load_zstack_cells, load_xenium_cells
from .reference_ported import find_min_z_spread_rotation

SEED_OFFSETS = [(200, 200), (200, -200), (-200, 200), (-200, -200), (0, 0)]
ROTS = list(range(-30, 41, 5))
PLANES = list(range(60, 341, 20))
SLAB_HALF = 20
R_CAND = 100.0
REPORTER_MIN = 2
N_CANDIDATES = 300

# k_cz=k_xen symmetric (rather than assuming one modality is systematically denser than the
# other) and n_best well below k (rather than requiring near-total neighbor agreement) -- k and
# n_best are coupled, not independently tunable; a nearby setting can certify nothing at all even
# when the underlying pose is correct, so re-tune both together if match quality looks off, don't
# nudge one in isolation.
K = 30
N_BEST = 10

_CZ = _PL = _IDS_CZ = _XEN = _XC = None
_FOV_UM = None
_SXY = None


def make_seed(rot_deg, cz_centroid, target, sxy):
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = target - R @ (cz_centroid * sxy)
    return lambda p: (R @ (np.asarray(p) * sxy).T).T + t


def _aff(M, p):
    return (M @ np.hstack([p, np.ones((len(p), 1))]).T).T[:, :2]


def _in_convex_quad(points, quad):
    """points: (N,2), quad: (4,2) convex polygon vertices (either winding order). Returns a bool
    array -- avoids a matplotlib.path dependency inside worker processes for a simple affine-mapped
    rectangle (always convex)."""
    pos = neg = np.ones(len(points), bool)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        cross = edge[0] * (points[:, 1] - a[1]) - edge[1] * (points[:, 0] - a[0])
        pos = pos & (cross >= 0)
        neg = neg & (cross <= 0)
    return pos | neg


def _fov_overlap_count(M0_cz_to_xen, xen_xy_um, cz_extent_um):
    """How many of the candidate Xenium cells (um, aligned frame) fall inside the z-stack's
    physical FOV footprint. `M0_cz_to_xen` is the raw fit direction used in `_certify_at`
    (find_affine_transformation_2d(slab, _XEN) -- z-stack (x,y) um -> Xenium_aligned (x,y) um), so
    the z-stack's own FOV extent is mapped FORWARD through it directly, no inversion needed."""
    lo, hi = cz_extent_um
    corners_cz = np.array([[lo, lo], [lo, hi], [hi, hi], [hi, lo]], float)
    corners_xen = _aff(M0_cz_to_xen, corners_cz)
    return int(_in_convex_quad(xen_xy_um, corners_xen).sum())


def _pool_init(cz, pl, ids_cz, xen, xc, fov_um, sxy):
    global _CZ, _PL, _IDS_CZ, _XEN, _XC, _FOV_UM, _SXY
    _CZ, _PL, _IDS_CZ, _XEN, _XC, _FOV_UM, _SXY = cz, pl, ids_cz, xen, xc, fov_um, sxy


def _rank_combo(args):
    st, rot, off = args
    slab = _CZ[(_PL >= st) & (_PL < st + 2 * SLAB_HALF)]
    if len(slab) < 20:
        return (st, rot, off, 0, 0)
    seed = make_seed(rot, slab.mean(0), _XC + np.array(off), _SXY)
    res = sp2.register_full(slab, _XEN, seed, R_cand=R_CAND, k_cz=K, k_xen=K, n_best=N_BEST, max_rounds=3)
    ncert = 0
    if len(res["anchors"]) >= 6:
        M0 = find_affine_transformation_2d(slab[res["anchors"][:, 0]], _XEN[res["anchors"][:, 1]])
        r = sp2.match(slab, _XEN, lambda p: _aff(M0, p), R_cand=30.0, n_best=N_BEST, anchor_frac=0.8,
                      k_cz=K, k_xen=K, max_rounds=3)
        cert = r["accepted"]
        if len(cert) >= 9:
            cert = sp2.flow_filter(cert, _aff(M0, slab)[cert[:, 0]], _XEN)
        ncert = len(cert)
    return (st, rot, off, len(res["anchors"]), ncert)


def _certify_combo(args):
    return _certify_at(*args)


def _certify_at(st, rot, off):
    in_slab = (_PL >= st) & (_PL < st + 2 * SLAB_HALF)
    slab = _CZ[in_slab]
    ids_slab = _IDS_CZ[in_slab]
    pl_slab = _PL[in_slab]
    seed = make_seed(rot, slab.mean(0), _XC + np.array(off), _SXY)
    res = sp2.register_full(slab, _XEN, seed, R_cand=R_CAND, k_cz=K, k_xen=K, n_best=N_BEST, max_rounds=3)
    if len(res["anchors"]) < 6:
        return None
    M0 = find_affine_transformation_2d(slab[res["anchors"][:, 0]], _XEN[res["anchors"][:, 1]])
    r = sp2.match(slab, _XEN, lambda p: _aff(M0, p), R_cand=30.0, n_best=N_BEST, anchor_frac=0.8,
                  k_cz=K, k_xen=K, max_rounds=3)
    cert = r["accepted"]
    if len(cert) >= 9:
        cert = sp2.flow_filter(cert, _aff(M0, slab)[cert[:, 0]], _XEN)
    if len(cert) < 6:
        return None
    dec = decompose_affine(M0)
    s1, s2 = abs(dec["scale1"]), abs(dec["scale2"])
    scale = float(np.mean([s1, s2]))
    anisotropy = float(max(s1, s2) / max(min(s1, s2), 1e-6))
    shear = float(abs(dec["shear"]))
    rotation = float(dec["rotation_deg"])
    spread = float(min(np.ptp(slab[cert[:, 0], 0]), np.ptp(slab[cert[:, 0], 1])))

    # Match rate: what fraction of the AVAILABLE cells in the overlap actually got certified --
    # not just the raw certified count, which doesn't reveal whether a candidate is a dense,
    # confident lock or a small set of coincidental matches riding on an unrelated population.
    n_xen_fov = _fov_overlap_count(M0, _XEN, cz_extent_um=(0.0, _FOV_UM))
    n_cz_slab = len(slab)
    match_rate = float(len(cert) / max(1, min(n_cz_slab, n_xen_fov)))

    # Real tissue shrinkage/expansion between modalities should be ~isotropic -- checking only the
    # MEAN scale lets a genuinely anisotropic (distorted) candidate slip through as "sane"; check
    # anisotropy and shear explicitly, not just the average magnitude. The match-rate floor is a
    # deliberately conservative sanity check, not a fine discriminator -- meant to catch clearly
    # too-weak locks (a handful of coincidental matches on an otherwise-uncorrelated population).
    sane = (0.60 <= scale <= 0.95) and (-20 <= rotation <= 40) and spread > 150 \
        and anisotropy <= 1.15 and shear <= 0.08 and match_rate >= 0.05
    return dict(st=st, rot=rot, off=off, slab=slab, ids_slab=ids_slab, pl_slab=pl_slab, cert=cert,
               n_cert=len(cert), scale=round(scale, 3), rotation=round(rotation, 1),
               anisotropy=round(anisotropy, 3), shear=round(shear, 4), sane=sane,
               n_xen_fov=n_xen_fov, n_cz_slab=n_cz_slab, match_rate=round(match_rate, 4))


def search_anchor_section(cfg, sec, max_workers=14, verbose=True):
    """Full GT-free grid search at Xenium section `sec`. Returns a dict with the winning candidate
    (moving z-stack points, fixed ALIGNED-frame Xenium points, plane, affine) or None if nothing
    plausible was found."""
    ids_cz_all, cz_all_xy, cz_all_pl = load_zstack_cells(cfg)
    ids_x, xen_xy, used_reporter, n_total = load_xenium_cells(cfg, sec, aligned=True, min_count=REPORTER_MIN)
    xc = xen_xy.mean(0)
    if verbose:
        pop = "reporter+" if used_reporter else "ALL cells (no reporter population source for this subject)"
        print(f"[sec{sec}] Xenium {pop}: {len(xen_xy)}/{n_total} | z-stack cells: {len(cz_all_xy)}", flush=True)

    fov_um = cfg.zstack_xy_um * cfg.zstack_shape_px[-1]
    sxy = cfg.tissue_expansion_scale
    pool_args = (cz_all_xy, cz_all_pl, ids_cz_all, xen_xy, xc, fov_um, sxy)

    combos = [(st, rot, off) for st in PLANES for rot in ROTS for off in SEED_OFFSETS]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_pool_init, initargs=pool_args) as ex:
        out = list(ex.map(_rank_combo, combos, chunksize=8))
    out.sort(key=lambda r: -r[4])
    if verbose:
        print(f"[sec{sec}] top-5 raw: " + ", ".join(f"(st={o[0]},rot={o[1]},off={o[2]},cert={o[4]})"
              for o in out[:5]), flush=True)

    to_certify = [(st, rot, off) for st, rot, off, n_anc, n_cert in out[:N_CANDIDATES] if n_cert >= 6]
    if verbose:
        print(f"[sec{sec}] certifying top {len(to_certify)} candidates in parallel...", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_pool_init, initargs=pool_args) as ex:
        certified = list(ex.map(_certify_combo, to_certify, chunksize=4))
    candidates = []
    for (st, rot, off), c in zip(to_certify, certified):
        if c is not None:
            candidates.append(c)
            if verbose:
                print(f"  candidate st={st} rot={rot} off={off}: n_cert={c['n_cert']} "
                      f"match_rate={c['match_rate']:.1%} (pool cz={c['n_cz_slab']} xen={c['n_xen_fov']}) "
                      f"scale={c['scale']} anisotropy={c['anisotropy']} shear={c['shear']} "
                      f"rotation={c['rotation']} sane={c['sane']}", flush=True)

    sane_c = [c for c in candidates if c["sane"]]
    pool = sane_c if sane_c else candidates
    if not pool:
        if verbose:
            print(f"[sec{sec}] NO CONVERGENCE among top-{N_CANDIDATES} candidates", flush=True)
        return None
    # Rank by match rate, not raw certified count -- raw count doesn't reveal whether a candidate
    # is a dense, confident lock or a small set of coincidental matches on an unrelated population.
    best = max(pool, key=lambda c: c["match_rate"])

    # Real per-cell z (plane) of each certified z-stack ROI -- NOT the slab window's geometric
    # center. Feeds find_min_z_spread_rotation the actual z-spread of the matched ROIs (a constant
    # z per point would trivially fit a perfectly flat plane and always return R_3d=identity --
    # silently disabling the tilt fit).
    real_z = best["pl_slab"][best["cert"][:, 0]]
    moving = np.column_stack([best["slab"][best["cert"][:, 0]] / cfg.zstack_xy_um, real_z])
    fixed_aligned = np.column_stack([xen_xy[best["cert"][:, 1]] / cfg.xenium_xy_um, np.zeros(best["n_cert"])])
    M_aligned, plane = bigwarp.affine_from_landmarks(moving, fixed_aligned)
    R_3d, rotated_pts = find_min_z_spread_rotation(moving)
    z_spread_before = float(np.ptp(moving[:, 2]))
    z_spread_after = float(np.ptp(rotated_pts[:, 2]))

    if verbose:
        print(f"[sec{sec}] SELECTED (sane={best['sane']}): plane={plane} n_cert={best['n_cert']} "
              f"match_rate={best['match_rate']:.1%} scale={best['scale']} anisotropy={best['anisotropy']} "
              f"shear={best['shear']} rotation={best['rotation']}", flush=True)
        print(f"[sec{sec}] tilt fit: z-spread {z_spread_before:.1f} -> {z_spread_after:.1f} px "
              f"(R_3d diag={np.diag(R_3d).round(4).tolist()})", flush=True)

    # Iterative re-match: correct every z-stack cell's (x,y,z) for the just-fitted tilt (rotate
    # about the volume center), re-select the slab by the DE-TILTED z, re-run soma-print on the
    # de-tilted x,y, then re-fit R_3d from THAT improved cert set and repeat -- until a round adds
    # no more certified pairs (recovers cells the flat single-z-band assumption missed near the
    # tilt's edges). Each round only adopted if it certifies strictly more pairs than the previous
    # one, so this can never regress and is guaranteed to terminate. NOTE: `tilt_fit.
    # fit_tilt_and_landmarks` supersedes this loop with an accumulate+bijective-constrained version
    # -- prefer that for the final tilt/landmark set; this loop is kept for `search_anchor_section`'s
    # own self-contained pose search.
    zstack_shape = cfg.zstack_shape_px
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
    center = np.array(zstack_shape) / 2.0
    cz_px = cz_all_xy / cfg.zstack_xy_um
    pts_zyx_orig = np.column_stack([cz_all_pl, cz_px[:, 1], cz_px[:, 0]])
    st = best["st"]
    n_rounds = 0

    while True:
        R_vol = P @ R_3d @ P
        rot_zyx = (R_vol @ (pts_zyx_orig - center).T).T + center
        new_pl = rot_zyx[:, 0]
        new_xy_um = rot_zyx[:, [2, 1]] * cfg.zstack_xy_um

        in_slab2 = (new_pl >= st) & (new_pl < st + 2 * SLAB_HALF)
        slab2, pl2 = new_xy_um[in_slab2], new_pl[in_slab2]
        seed2 = make_seed(best["rot"], slab2.mean(0), xc + np.array(best["off"]), sxy)
        res2 = sp2.register_full(slab2, xen_xy, seed2, R_cand=R_CAND, k_cz=K, k_xen=K, n_best=N_BEST, max_rounds=3)
        cert2 = np.empty((0, 2), int)
        if len(res2["anchors"]) >= 6:
            M0b = find_affine_transformation_2d(slab2[res2["anchors"][:, 0]], xen_xy[res2["anchors"][:, 1]])
            r2 = sp2.match(slab2, xen_xy, lambda p: _aff(M0b, p), R_cand=30.0, n_best=N_BEST, anchor_frac=0.8,
                          k_cz=K, k_xen=K, max_rounds=3)
            cert2 = r2["accepted"]
            if len(cert2) >= 9:
                cert2 = sp2.flow_filter(cert2, _aff(M0b, slab2)[cert2[:, 0]], xen_xy)
        n_rounds += 1
        if verbose:
            print(f"[sec{sec}] tilt-refit round {n_rounds}: n_cert {best['n_cert']} -> {len(cert2)}", flush=True)
        if len(cert2) <= best["n_cert"]:
            if verbose:
                print(f"[sec{sec}] no further gain -- stopping after {n_rounds} round(s)", flush=True)
            break

        real_z2 = pl2[cert2[:, 0]]
        moving = np.column_stack([slab2[cert2[:, 0]] / cfg.zstack_xy_um, real_z2])
        fixed_aligned = np.column_stack([xen_xy[cert2[:, 1]] / cfg.xenium_xy_um, np.zeros(len(cert2))])
        M_aligned, plane = bigwarp.affine_from_landmarks(moving, fixed_aligned)
        # Re-fit R_3d from the IMPROVED cert set, in the ORIGINAL (untilted) frame -- compose with
        # the running rotation so it always maps original-volume points, not the already-rotated
        # ones. moving's (x,y) are in the de-tilted frame; convert back to original-frame (x,y,z)
        # before re-fitting so R_3d keeps meaning "original -> horizontal", not "twice-rotated".
        moving_zyx_detilted = np.column_stack([real_z2, moving[:, 1], moving[:, 0]])
        moving_zyx_orig = (R_vol.T @ (moving_zyx_detilted - center).T).T + center
        moving_orig = np.column_stack([moving_zyx_orig[:, 2], moving_zyx_orig[:, 1], moving_zyx_orig[:, 0]])
        R_3d_new, rotated_pts = find_min_z_spread_rotation(moving_orig)
        z_spread_before = float(np.ptp(moving_orig[:, 2]))
        z_spread_after = float(np.ptp(rotated_pts[:, 2]))
        best = dict(best, n_cert=len(cert2))
        R_3d = R_3d_new

    if verbose:
        print(f"[sec{sec}] final tilt: z-spread {z_spread_before:.1f} -> {z_spread_after:.1f} px "
              f"(R_3d diag={np.diag(R_3d).round(4).tolist()}) after {n_rounds} round(s)", flush=True)

    return dict(sec=sec, plane=plane, n_cert=best["n_cert"], match_rate=best["match_rate"],
               n_cz_slab=best["n_cz_slab"], n_xen_fov=best["n_xen_fov"], scale=best["scale"],
               anisotropy=best["anisotropy"], shear=best["shear"], rotation=best["rotation"],
               sane=best["sane"], used_reporter=used_reporter, moving=moving, fixed_aligned=fixed_aligned,
               M_aligned=M_aligned, R_3d=R_3d, z_spread_before=z_spread_before, z_spread_after=z_spread_after,
               n_tilt_rounds=n_rounds)
