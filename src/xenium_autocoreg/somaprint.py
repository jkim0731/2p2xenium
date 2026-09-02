"""2D soma-print matcher — the validated approach from `docs/xenium_coregistration_protocol.md`
(ported from the 2P↔HCR matcher, density-mismatch-aware).

NOT the RANSAC/ICP/Hungarian engine in `somaprint.py`, and NOT image-NCC. Pure geometric,
cell-centroid, iterative:
  warm-start (seed) → [soma_score (R_cand) → mutual_best → (rd0 local_flow) → within-round
  anchor_vote gate → re-evaluate accepted (no lock) → fit 2D TPS] × rounds.

Descriptor (§1a) is the raw displacement-vector set to neighbors — frame-dependent, so rotation is
handled by the seed/warm-start, not the descriptor. Density handled by radius/asymmetric-kNN
neighborhoods + mutual-best + anchor-vote (§4). All coordinates 2D xy, in µm.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator


# ───────── descriptor ─────────
def neighbor_indices(xy, radius=None, k=None):
    """Per-cell neighbor index arrays — radius-based (preferred under density mismatch) or k-NN."""
    tree = cKDTree(xy)
    if radius is not None:
        nb = tree.query_ball_point(xy, radius)
        return [np.array([j for j in nb[i] if j != i], int) for i in range(len(xy))]
    d, ii = tree.query(xy, k=min(k + 1, len(xy)))
    return [ii[i][ii[i] != i][:k] for i in range(len(xy))]


def descriptors(xy, nbr_idx):
    """Displacement-vector set per cell (relative to the cell)."""
    return [xy[nb] - xy[i] if len(nb) else np.zeros((0, 2)) for i, nb in enumerate(nbr_idx)]


def fixed_knn_descriptor(xy, k):
    """Fixed-k neighbor descriptor for the vectorized path. Returns idx (N,k) int and
    V (N,k,2) displacement vectors (zero-padded if fewer than k neighbors)."""
    n = len(xy)
    kk = min(k, n - 1)
    _, ii = cKDTree(xy).query(xy, k=kk + 1)
    if ii.ndim == 1:
        ii = ii[:, None]
    idx = ii[:, 1:kk + 1]                                   # drop self
    V = xy[idx] - xy[:, None, :]                            # (N,k,2)
    if kk < k:                                              # pad
        idx = np.pad(idx, ((0, 0), (0, k - kk)), constant_values=-1)
        V = np.pad(V, ((0, 0), (0, k - kk), (0, 0)))
    return idx, V


# ───────── soma-print score (vectorized) ─────────
def soma_score(cz_xy, xen_xy, cz_V, xen_V, R_cand, n_best):
    """(N_cz, M_xen) score matrix; pair score = mean of the n_best smallest distances between the
    two cells' displacement-vector sets (fixed-k arrays). Only pairs within R_cand are scored.
    Vectorized per cz cell over its candidates."""
    N, M = len(cz_xy), len(xen_xy)
    D = np.full((N, M), np.inf)
    cand = cKDTree(xen_xy).query_ball_point(cz_xy, R_cand)
    for i in range(N):
        js = cand[i]
        if not js:
            continue
        js = np.asarray(js)
        # dist between vi (kc,2) and each candidate's wj (nc,kx,2) -> (nc,kc,kx)
        dm = np.linalg.norm(cz_V[i][None, :, None, :] - xen_V[js][:, None, :, :], axis=3)
        flat = dm.reshape(len(js), -1)
        k = min(n_best, flat.shape[1])
        D[i, js] = np.partition(flat, k - 1, axis=1)[:, :k].mean(1)
    return D


def _best_neighbor_corr(i, j, cz_V, xen_V, cz_idx, xen_idx, n_best):
    """The n_best (cz_neighbor_cell, xen_neighbor_cell) correspondences implied by a pair (i,j)."""
    dm = np.linalg.norm(cz_V[i][:, None, :] - xen_V[j][None, :, :], axis=2)
    k = min(n_best, dm.size)
    flat = np.argpartition(dm.ravel(), k - 1)[:k]
    aa, bb = np.unravel_index(flat, dm.shape)
    return [(int(cz_idx[i][a]), int(xen_idx[j][b])) for a, b in zip(aa, bb)
            if cz_idx[i][a] >= 0 and xen_idx[j][b] >= 0]


def mutual_best(D):
    """Symmetric best-best pairs (density-robust)."""
    N, M = D.shape
    if not np.isfinite(D).any():
        return []
    bj = np.argmin(D, axis=1); bi = np.argmin(D, axis=0)
    pairs = [(i, int(bj[i])) for i in range(N) if np.isfinite(D[i, bj[i]]) and bi[bj[i]] == i]
    return pairs


# ───────── 2D TPS ─────────
def fit_tps_2d(src_xy, dst_xy, smoothing=1.0):
    return RBFInterpolator(np.asarray(src_xy), np.asarray(dst_xy),
                           kernel="thin_plate_spline", smoothing=smoothing)


def apply_tps_2d(tps, xy):
    return tps(np.asarray(xy))


def _affine_apply(Mx, P):
    return (Mx @ np.hstack([P, np.ones((len(P), 1))]).T).T[:, :2]


def fit_warp(model, src, dst, rd, smoothing):
    """Build a warp callable from matched (src→dst). `model`:
      'tps'        — thin-plate spline every round (local; does NOT spread from a clustered set);
      'affine'     — global 2D affine (per-axis scale+shear+rotation+translation); generalizes to the
                     whole FOV, so it EXPANDS the matched region across rounds;
      'affine_tps' — affine for the first 2 rounds (spread), then TPS (local refine)."""
    from .geometry import find_affine_transformation_2d
    if model == "affine" or (model == "affine_tps" and rd < 2):
        Mx = find_affine_transformation_2d(np.asarray(src), np.asarray(dst))
        return lambda P: _affine_apply(Mx, np.asarray(P))
    tps = fit_tps_2d(src, dst, smoothing)
    return lambda P: tps(np.asarray(P))


# ───────── the matcher ─────────
def match(cz_xy0, xen_xy, seed, R_cand=200.0, k_cz=15, k_xen=30,
          n_best=5, anchor_frac=0.8, max_rounds=5, converge_rel=0.02, local_flow_rd0=False,
          tps_smoothing=5.0, warp_model="affine_tps", **_ignore):
    """Run the iterative matcher between a CZ slab (cz_xy0, in CZ µm) and the Xenium top slice
    (xen_xy, µm), warm-started by `seed` (a callable cz_xy0 -> Xenium-frame xy, encoding sxy≈0.8 +
    translation + rotation). Returns a dict with accepted pairs, the final TPS, and quality stats.

    Descriptor: fixed asymmetric k-NN (k_cz on the sparser CZ side, k_xen on the denser Xenium side;
    §4.2) for a vectorized score. Density handled by mutual-best + within-round anchor-vote (§4)."""
    cz_pos = np.asarray(seed(cz_xy0), float)                      # warm-start into Xenium frame
    xen_idx, xen_V = fixed_knn_descriptor(xen_xy, k_xen)          # Xenium fixed across rounds
    accepted, warp = [], None
    history = []
    prev_n = 0
    for rd in range(max_rounds):
        if warp is not None:
            cz_pos = warp(np.asarray(seed(cz_xy0), float))
        cz_idx, cz_V = fixed_knn_descriptor(cz_pos, k_cz)
        D = soma_score(cz_pos, xen_xy, cz_V, xen_V, R_cand, n_best)
        pairs = mutual_best(D)
        if not pairs:
            break
        mutual_set = set(pairs)
        if rd == 0 and local_flow_rd0:
            pairs = _local_flow_filter(pairs, cz_pos, xen_xy)
        # within-round anchor-vote gate
        kept = []
        for (i, j) in pairs:
            corr = _best_neighbor_corr(i, j, cz_V, xen_V, cz_idx, xen_idx, n_best)
            if not corr:
                continue
            frac = np.mean([(a, b) in mutual_set for a, b in corr])
            if frac >= anchor_frac:
                kept.append((i, j))
        accepted = kept                                           # re-evaluate, do NOT lock
        history.append(dict(round=rd, n_mutual=len(pairs), n_accepted=len(accepted)))
        if len(accepted) >= 3:
            # fit the warp in the *seed* frame so it composes with the warm-start each round
            src = np.asarray(seed(cz_xy0))[[p[0] for p in accepted]]
            dst = xen_xy[[p[1] for p in accepted]]
            warp = fit_warp(warp_model, src, dst, rd, tps_smoothing)
        if rd > 0 and prev_n > 0 and abs(len(accepted) - prev_n) / prev_n < converge_rel:
            break
        prev_n = max(len(accepted), 1)
    return dict(accepted=np.array(accepted, int) if accepted else np.zeros((0, 2), int),
                tps=warp, cz_pos=cz_pos, n_accepted=len(accepted), history=history)


def flow_filter(accepted, src_warmstart, xen_xy, k=8, tol_um=45.0):
    """Local optical-flow consistency filter (post-hoc). A correct match's warm-start→match
    displacement agrees with its accepted neighbours' median displacement; false positives are
    outliers and get dropped. Validated to beat a stricter anchor-vote gate (more recall at 100%
    precision): pair the lenient 3/5 gate with this filter. `src_warmstart` = seed(cz)[accepted_src],
    `accepted` = (K,2) [src,dst] indices. Returns the filtered (K',2)."""
    acc = np.asarray(accepted)
    if len(acc) < k + 1:
        return acc
    src = np.asarray(src_warmstart); dst = xen_xy[acc[:, 1]]; flow = dst - src
    _, nn = cKDTree(src).query(src, k=k + 1)
    keep = [t for t in range(len(acc))
            if np.linalg.norm(flow[t] - np.median(flow[nn[t][1:]], axis=0)) < tol_um]
    return acc[keep]


def _local_flow_filter(pairs, cz_pos, xen_xy, k=8, tol_um=40.0):
    """Round-0 outlier reject: a pair's displacement should agree with its neighbors' displacements."""
    if len(pairs) < k + 1:
        return pairs
    src = cz_pos[[p[0] for p in pairs]]; dst = xen_xy[[p[1] for p in pairs]]
    flow = dst - src
    tree = cKDTree(src); _, nn = tree.query(src, k=min(k + 1, len(src)))
    keep = []
    for t, (i, j) in enumerate(pairs):
        med = np.median(flow[nn[t][1:]], axis=0)
        if np.linalg.norm(flow[t] - med) < tol_um:
            keep.append((i, j))
    return keep


# ───────── GT-free quality (§6) ─────────
def quality(result, cz_xy0, xen_xy, seed):
    """GT-free quality of a run: accepted-set size + spatial spread + TPS leave-one-out residual."""
    acc = result["accepted"]
    if len(acc) < 3:
        return dict(n=len(acc), spread=0.0, loo_um=np.inf, score=float(len(acc)))
    dst = xen_xy[acc[:, 1]]
    spread = float(np.sqrt(np.linalg.det(np.cov(dst.T)) + 1e-9))   # ~area of accepted spread
    src_seed = np.asarray(seed(cz_xy0))[acc[:, 0]]
    loo = []
    for t in range(len(acc)):
        m = np.ones(len(acc), bool); m[t] = False
        try:
            tps = fit_tps_2d(src_seed[m], dst[m], 5.0)
            loo.append(np.linalg.norm(tps(src_seed[t:t + 1])[0] - dst[t]))
        except Exception:
            pass
    loo_um = float(np.median(loo)) if loo else np.inf
    return dict(n=len(acc), spread=spread, loo_um=loo_um, score=float(len(acc)))


# ───────── whole-FOV expansion ─────────
def expand_by_affine_nn(anchors, cz_xy, xen_xy, nn_radius_um=12.0, flow_tol_um=30.0):
    """Expand a clustered set of reliable anchors to the WHOLE FOV. The soma-print matcher only
    fires where the local pattern is distinctive (a minority of the FOV), but those anchors
    determine a globally accurate 2D affine. So: fit a global affine from the anchors, then assign
    every CZ cell to its nearest Xenium cell within `nn_radius_um` of its affine-predicted
    position, and apply the local optical-flow filter.

    anchors : (K,2) [cz_idx, xen_idx]   cz_xy : (N,2) µm (CZ slab, ORIGINAL frame)   xen_xy : (M,2) µm
    Returns (P,2) [cz_idx, xen_idx] expanded matches, and the 3x3 affine (CZ→Xenium)."""
    from .geometry import find_affine_transformation_2d
    anchors = np.asarray(anchors)
    if len(anchors) < 3:
        return anchors, None
    M = find_affine_transformation_2d(cz_xy[anchors[:, 0]], xen_xy[anchors[:, 1]])   # CZ µm -> Xenium µm
    pred = (M @ np.hstack([cz_xy, np.ones((len(cz_xy), 1))]).T).T[:, :2]
    dnn, idx = cKDTree(xen_xy).query(pred, k=1)
    keep = dnn < nn_radius_um
    pairs = np.column_stack([np.where(keep)[0], idx[keep]])
    pairs = flow_filter(pairs, pred[pairs[:, 0]], xen_xy, tol_um=flow_tol_um)
    return pairs, M


def register_full(cz_xy, xen_xy, seed, anchor_frac=0.6, flow=True, **match_kw):
    """End-to-end whole-FOV registration (the validated recipe):
      1. soma-print `match` (lenient gate) → mutual-best + anchor-vote anchors,
      2. local optical-flow filter → reliable anchors (false positives removed),
      3. `expand_by_affine_nn` → global affine + nearest-neighbour expansion + flow filter.
    Returns dict(anchors, matches, affine). `cz_xy`=(N,2) CZ slab µm, `seed`=warm-start callable."""
    r = match(cz_xy, xen_xy, seed, anchor_frac=anchor_frac, **match_kw)
    anc = r["accepted"]
    if flow and len(anc) >= 9:
        anc = flow_filter(anc, np.asarray(seed(cz_xy))[anc[:, 0]], xen_xy)
    matches, M = expand_by_affine_nn(anc, cz_xy, xen_xy)
    return dict(anchors=anc, matches=matches, affine=M)
