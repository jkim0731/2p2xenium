"""Soma-print cell matching (Wang et al., TRU-FACT) — ported from the developer's
`20260507 SomaPrint3D.ipynb`, plus the radius-based rotation-invariant Soma-print *descriptor*
(the paper's neighbor-graph signature) which the dev pipeline left as an optional `features` slot.

Pipeline: `soma_print_descriptor` (per-cell local-constellation signature, contrast-invariant) →
`register_and_match` (descriptor-seeded RANSAC global rigid init → ICP → Hungarian assignment on a
spatial + descriptor cost, with a per-pair Soma-print score 0–100).

Convention: src = ex-vivo (moving, e.g. Xenium); dst = in-vivo (FIXED, e.g. czstack). Points are
(N,3) [z,y,x] in µm — use z=0 for the 2D-per-slab mode.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist


# ───────────────────────── containers ─────────────────────────
@dataclass
class RegistrationResult:
    affine_matrix: np.ndarray; rms_history: List[float]; n_inliers: int
    converged: bool; notes: str
    def transform(self, pts):
        h = np.hstack([pts, np.ones((len(pts), 1))]); return (self.affine_matrix @ h.T).T[:, :3]


@dataclass
class MatchingResult:
    matched_pairs: np.ndarray; match_distances: np.ndarray; match_costs: np.ndarray
    match_scores: np.ndarray; unmatched_src: np.ndarray; unmatched_dst: np.ndarray
    score_matrix: np.ndarray; best_match_scores: np.ndarray; second_best_scores: np.ndarray


# ───────────────────── Soma-print descriptor (NEW) ─────────────────────
def soma_print_descriptor(pts, radius=75.0, r_bins=4, a_bins=12):
    """Per-cell rotation-invariant local-constellation signature ('Soma-print').
    For each cell, histogram its neighbors WITHIN `radius` µm over (radial × angular) bins, then take
    the angular FFT magnitude per radial ring → invariant to in-plane rotation. Radius-based
    neighborhood (loosened from fixed-k), per the 50–100 µm guidance. Returns (N, F), L2-normalized.
    `pts` is (N,2)[y,x] or (N,3)[z,y,x] (xy used)."""
    xy = np.asarray(pts, float)[:, -2:]
    tree = cKDTree(xy)
    nbrs = tree.query_ball_point(xy, radius)
    desc = np.zeros((len(xy), r_bins, a_bins))
    for i, nb in enumerate(nbrs):
        for j in nb:
            if j == i:
                continue
            dy, dx = xy[j] - xy[i]
            r = np.hypot(dy, dx)
            if r <= 1e-6 or r > radius:
                continue
            ri = min(int(r / radius * r_bins), r_bins - 1)
            ai = int((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * a_bins) % a_bins
            desc[i, ri, ai] += 1.0
    F = np.abs(np.fft.rfft(desc, axis=2)).reshape(len(xy), -1)   # rotation-invariant
    n = np.linalg.norm(F, axis=1, keepdims=True); n[n == 0] = 1.0
    return F / n


# ───────────────────────── registration ─────────────────────────
def _umeyama_rigid(src, dst):
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = ((dst - mu_d).T @ (src - mu_s)) / n
    U, _, Vt = np.linalg.svd(cov)
    D = np.eye(d); D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    T = np.eye(d + 1); T[:d, :d] = R; T[:d, d] = mu_d - R @ mu_s
    return T


def _apply(pts, T):
    return (T @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]


def _ransac_init(src, dst, n_iterations, sample_size, inlier_dist, rng,
                 src_feat=None, dst_feat=None):
    dst_tree = KDTree(dst); best_T = np.eye(4); best_count = -1
    use_feat = src_feat is not None and dst_feat is not None
    if use_feat:
        std = np.vstack([src_feat, dst_feat]).std(0); std[std < 1e-9] = 1.0
        feat_tree = KDTree(dst_feat / std)
        k_nn = min(5, len(dst))
        _, nn = feat_tree.query(src_feat / std, k=k_nn, workers=-1)
        cand_src = np.repeat(np.arange(len(src)), k_nn); cand_dst = nn.ravel()
        n_cands = len(cand_src)
    for _ in range(n_iterations):
        if use_feat and n_cands >= sample_size:
            chosen = rng.choice(n_cands, sample_size, replace=False)
            s_idx, d_idx = cand_src[chosen], cand_dst[chosen]
            if len(set(s_idx)) < sample_size or len(set(d_idx)) < sample_size:
                s_idx = rng.choice(len(src), sample_size, replace=False)
                d_idx = rng.choice(len(dst), sample_size, replace=False)
        else:
            s_idx = rng.choice(len(src), sample_size, replace=False)
            d_idx = rng.choice(len(dst), sample_size, replace=False)
        try:
            T_cand = _umeyama_rigid(src[s_idx], dst[d_idx])
        except np.linalg.LinAlgError:
            continue
        dists, _ = dst_tree.query(_apply(src, T_cand), k=1, workers=-1)
        c = int((dists < inlier_dist).sum())
        if c > best_count:
            best_count, best_T = c, T_cand
    return best_T


def _icp(src, dst, T_init, max_dist, n_iters, tol):
    dst_tree = KDTree(dst); src_cur = _apply(src, T_init); T_acc = T_init.copy()
    rms_history: List[float] = []; n_inliers_final = 0; converged = False
    for it in range(n_iters):
        dists, nn = dst_tree.query(src_cur, k=1, workers=-1)
        mask = dists < max_dist; n_inliers_final = int(mask.sum())
        if n_inliers_final < 4:
            break
        T_step = _umeyama_rigid(src_cur[mask], dst[nn[mask]])
        rms_history.append(float(np.sqrt(np.mean(dists[mask] ** 2))))
        src_cur = _apply(src_cur, T_step); T_acc = T_step @ T_acc
        if it > 0 and abs(rms_history[-2] - rms_history[-1]) < tol:
            converged = True; break
    return T_acc, rms_history, n_inliers_final, converged


def register_somas(src, dst, ransac_iterations=2000, ransac_sample_size=4,
                   ransac_inlier_dist_um=25.0, max_correspondence_dist_um=30.0,
                   n_icp_iterations=200, convergence_tol=1e-5, random_seed=42,
                   src_features=None, dst_features=None):
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    if len(src) < 4 or len(dst) < 4:
        raise ValueError("Both clouds need ≥4 points.")
    rng = np.random.default_rng(random_seed)
    T_init = _ransac_init(src, dst, ransac_iterations, ransac_sample_size,
                          ransac_inlier_dist_um, rng, src_features, dst_features)
    T_final, rms, n_in, conv = _icp(src, dst, T_init, max_correspondence_dist_um,
                                    n_icp_iterations, convergence_tol)
    notes = f"{'converged' if conv else 'no-converge'} | RMS={rms[-1]:.3f}µm | inliers={n_in}" if rms \
        else f"no ICP steps | inliers={n_in}"
    return RegistrationResult(T_final, rms, n_in, conv, notes)


# ───────────────────────── matching ─────────────────────────
def _build_cost_and_score(src, dst, max_match_dist_um, src_features, dst_features, feature_weight):
    N, M = len(src), len(dst); INF = 1e9
    D = cdist(src, dst)
    C = np.full((N, M), INF); within = D < max_match_dist_um
    C[within] = D[within] / max_match_dist_um
    max_valid = 1.0
    if src_features is not None and dst_features is not None:
        sf, df = np.asarray(src_features, float), np.asarray(dst_features, float)
        std = np.vstack([sf, df]).std(0); std[std < 1e-9] = 1.0
        Fdim = sf.shape[1]
        featD = cdist(sf / std, df / std, metric="cityblock") / Fdim
        C[within] += feature_weight * featD[within]; max_valid += feature_weight
    scores = np.zeros((N, M)); valid = C < INF
    scores[valid] = np.clip((1.0 - C[valid] / max_valid) * 100.0, 0, 100)
    return C, D, scores


def _solve_assignment(C, D, scores, cost_threshold):
    N, M = C.shape; INF = 1e9; pad = max(N, M)
    Cp = np.full((pad, pad), cost_threshold + 1e-6); Cp[:N, :M] = C
    ri, ci = linear_sum_assignment(Cp)
    sp, dp, co, di, sc = [], [], [], [], []
    for r, c in zip(ri, ci):
        if r >= N or c >= M or C[r, c] >= INF or C[r, c] > cost_threshold:
            continue
        sp.append(r); dp.append(c); co.append(float(C[r, c])); di.append(float(D[r, c])); sc.append(float(scores[r, c]))
    pairs = np.column_stack([sp, dp]) if sp else np.zeros((0, 2), int)
    ms, md = set(sp), set(dp)
    ss = np.sort(scores, axis=0)[::-1, :] if N else np.zeros((1, M))
    return MatchingResult(pairs, np.array(di), np.array(co), np.array(sc),
                          np.array([i for i in range(N) if i not in ms], int),
                          np.array([j for j in range(M) if j not in md], int),
                          scores, ss[0, :] if N else np.zeros(M),
                          ss[1, :] if N >= 2 else np.zeros(M))


def match_somas(src, dst, registration=None, max_match_dist_um=15.0, src_features=None,
                dst_features=None, feature_weight=0.3, cost_threshold=1.0):
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    src_m = registration.transform(src) if registration is not None else src
    C, D, scores = _build_cost_and_score(src_m, dst, max_match_dist_um, src_features, dst_features, feature_weight)
    return _solve_assignment(C, D, scores, cost_threshold)


def register_and_match(src, dst, ransac_iterations=2000, ransac_sample_size=4,
                       ransac_inlier_dist_um=25.0, max_correspondence_dist_um=30.0,
                       n_icp_iterations=200, convergence_tol=1e-5, random_seed=42,
                       max_match_dist_um=15.0, src_features=None, dst_features=None,
                       feature_weight=0.3, cost_threshold=1.0):
    """Convenience: register (descriptor-seeded RANSAC + ICP) then match. Returns (reg, match)."""
    reg = register_somas(src, dst, ransac_iterations, ransac_sample_size, ransac_inlier_dist_um,
                         max_correspondence_dist_um, n_icp_iterations, convergence_tol, random_seed,
                         src_features, dst_features)
    match = match_somas(src, dst, reg, max_match_dist_um, src_features, dst_features,
                        feature_weight, cost_threshold)
    return reg, match
