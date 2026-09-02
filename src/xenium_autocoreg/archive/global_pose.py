"""Phase 1 — automated slice-1 global pose search (the step that replaces manual BigWarp
landmark placement).

Idea (grounded in code/docs/gt_analysis.md): the Xenium->z-stack affine is almost fully
constrained — scale ~0.77 (xen->z px, includes the ~18% tissue shrink), shear ~0 — so the only
unknowns for slice 1 are (rotation, plane, tx, ty).  We rescale the large Xenium feature image by
the scale prior (its features then sit at the size/scale they appear in the z-stack), then for each
(rotation, plane) we slide the 512x512 z-band template over the rescaled+rotated section with
normalized cross-correlation (`skimage.feature.match_template`), which gives the translation and a
score in closed form.  Best (rotation, plane, translation) -> a 3x3 affine in the GT convention
(Xenium row,col -> z-stack row,col), directly comparable to `Affine matrices/*.npy`.

All geometry is built from explicit 3x3 matrices to avoid skimage.rotate sign ambiguity.
"""
import numpy as np
from skimage.transform import rescale, warp
from skimage.feature import match_template

from . import XENIUM_S2_UM, ZSTACK_XY_UM


# ---- explicit homogeneous matrices in (x=col, y=row) ----
def _T(tx, ty):  return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1.]])
def _S(s):       return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1.]])
def _R(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
_P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.]])   # swap (x,y)<->(row,col)


def tissue_centroid(xen_feat):
    """Centroid (row,col) of the Xenium section tissue — the automatic translation anchor, since the
    2P FOV lands near-central (gt_analysis: 0.11–0.21 of tissue radius). Use as `anchor_rc`."""
    from skimage.filters import threshold_otsu
    from skimage.filters import gaussian
    a = gaussian(xen_feat.astype(np.float32), 4)
    m = a > threshold_otsu(a) * 0.5
    ys, xs = np.where(m)
    if len(xs) < 50:
        ys, xs = np.where(xen_feat > 0)
    return (float(ys.mean()), float(xs.mean()))


def band_projection(vol, plane, half=8, how="mean"):
    lo, hi = max(0, plane - half), min(vol.shape[0], plane + half + 1)
    sub = vol[lo:hi].astype(np.float32)
    return sub.mean(0) if how == "mean" else sub.max(0)


def _norm(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) if hi > lo else a * 0


def feature_map(a, sigma=0):
    """Normalized (optionally Gaussian-blurred 'cell-density') feature — blurring suppresses
    high-frequency modality differences and emphasizes shared coarse structure."""
    from scipy.ndimage import gaussian_filter
    f = _norm(a)
    return _norm(gaussian_filter(f, sigma)) if sigma else f


def _coverage(fg, th, tw):
    """Fraction of foreground in every top-left th×tw window (match_template index convention)."""
    P = np.zeros((fg.shape[0] + 1, fg.shape[1] + 1))
    P[1:, 1:] = np.cumsum(np.cumsum(fg, 0), 1)
    H, W = fg.shape
    I, J = np.mgrid[:H - th + 1, :W - tw + 1]
    s = P[I + th, J + tw] - P[I, J + tw] - P[I + th, J] + P[I, J]
    return s / (th * tw)


def search_pose(xen_feat, vol, scale=0.77, rot_range=(-12, 30, 2),
                plane_range=(50, 142, 4), band_half=8, how="mean",
                downsample=2, refine=True, feat_sigma=3, min_cover=0.4,
                anchor_rc=None, anchor_radius_px=120):
    """If `anchor_rc=(row,col)` is given (an operator's rough click marking where the z-FOV centre
    lands in the original Xenium image), the translation search is restricted to within
    `anchor_radius_px` (z-stack px) of it — the semi-automatic mode."""
    """Coarse (then optionally fine) global pose search.

    Parameters
    ----------
    xen_feat : (H, W) Xenium slice-1 feature image (native ~0.85 um/px), e.g. Neurons or protein.
    vol      : (Z, 512, 512) z-stack volume, single channel (e.g. GCaMP for cells, Dextran vessels).
    scale    : xen->z pixel scale prior (~0.77).  rot_range/plane_range = (start, stop, step).

    Returns dict: M (3x3 xen row,col -> z row,col), rotation, plane, translation (row,col in z px),
    score (NCC), and the search grid best-per-(rot,plane).
    """
    xen = feature_map(xen_feat, feat_sigma)
    # bring Xenium to z-stack pixel scale, then to the coarse grid
    xs = rescale(xen, scale, anti_aliasing=True, preserve_range=True)
    if downsample > 1:
        xs = rescale(xs, 1.0 / downsample, anti_aliasing=True, preserve_range=True)
    H, W = xs.shape
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ds = downsample

    rots = np.arange(*rot_range)
    planes = np.arange(*plane_range)
    grid = np.full((len(rots), len(planes)), -np.inf)

    # precompute coarse templates per plane
    templates = {}
    for p in planes:
        T = feature_map(band_projection(vol, int(p), band_half, how), feat_sigma)
        if ds > 1:
            T = rescale(T, 1.0 / ds, anti_aliasing=True, preserve_range=True)
        templates[int(p)] = T

    Tshape = next(iter(templates.values())).shape
    a_scaled = (_S(scale / ds) @ np.array([anchor_rc[1], anchor_rc[0], 1.0])
                if anchor_rc is not None else None)
    rad_c = anchor_radius_px / ds

    best = dict(score=-np.inf)
    for ri, deg in enumerate(rots):
        # sign: with the row/col swap, _R(-deg) makes the final matrix rotation == +deg, so the
        # reported `deg` matches decompose_affine(M)['rotation_deg'] (GT matrix rotations are +2..+27).
        Rc = _T(cx, cy) @ _R(-deg) @ _T(-cx, -cy)         # rotate-about-centre (x,y)
        Xr = warp(xs, np.linalg.inv(Rc), output_shape=xs.shape,
                  preserve_range=True, order=1)
        amask = None
        if a_scaled is not None:                          # restrict translation around the anchor
            a_xr = Rc @ a_scaled                          # FOV centre's expected (x,y) in Xr
            exp_r, exp_c = a_xr[1] - Tshape[0] / 2, a_xr[0] - Tshape[1] / 2
            oh, ow = Xr.shape[0] - Tshape[0] + 1, Xr.shape[1] - Tshape[1] + 1
            if oh > 0 and ow > 0:
                I, J = np.mgrid[:oh, :ow]
                amask = (I - exp_r) ** 2 + (J - exp_c) ** 2 <= rad_c ** 2
        for pj, p in enumerate(planes):
            T = templates[int(p)]
            if T.shape[0] >= Xr.shape[0] or T.shape[1] >= Xr.shape[1]:
                continue
            resp = match_template(Xr, T)
            # mask out background-dominated positions (kills the skimage constant-region artifact)
            cov = _coverage((Xr > 0.05).astype(np.float64), T.shape[0], T.shape[1])
            resp = np.where(cov >= min_cover, resp, -np.inf)
            if amask is not None:
                resp = np.where(amask, resp, -np.inf)
            if not np.isfinite(resp).any():
                continue
            idx = np.unravel_index(np.argmax(resp), resp.shape)
            sc = float(resp[idx])
            grid[ri, pj] = sc
            if sc > best["score"]:
                prow, pcol = idx[0] * 1.0, idx[1] * 1.0      # T top-left in Xr (coarse px)
                best = dict(score=sc, deg=float(deg), plane=int(p),
                            prow=prow, pcol=pcol, Rc=Rc, H=H, W=W)

    M = _assemble(best, scale, ds)
    from .geometry import decompose_affine
    out = dict(M=M, rotation=decompose_affine(M)["rotation_deg"], plane=best["plane"],
               score=best["score"], translation=(M[0, 2], M[1, 2]),
               grid=grid, rots=rots, planes=planes, feat_sigma=feat_sigma, min_cover=min_cover,
               anchor_rc=anchor_rc, anchor_radius_px=anchor_radius_px)

    if refine:
        out = _refine(out, xen_feat, vol, scale, band_half, how)
    return out


def _assemble(best, scale, ds):
    """Compose the explicit forward chain xen_orig -> z (row,col)."""
    cx, cy = (best["W"] - 1) / 2.0, (best["H"] - 1) / 2.0
    # X_theta -> z(template) translation (coarse px): T(x,y) ~ Xr(pcol+x, prow+y) => Xr->T = -(pcol,prow)
    Ttr = _T(-best["pcol"], -best["prow"])
    M_coarse_xy = Ttr @ best["Rc"] @ _S(1.0)              # in coarse, scaled-pixel space
    # undo the coarse downsample (coarse px -> z full px) and fold the xen->z scale:
    #   xen_orig --S(scale)--> scaled --S(1/ds)--> coarse --[M_coarse_xy]--> z_coarse --S(ds)--> z
    M_xy = _S(ds) @ M_coarse_xy @ _S(scale / ds)
    return _P @ M_xy @ _P                                  # -> (row,col)


def _refine(out, xen_feat, vol, scale, band_half, how):
    """Fine search at full resolution around the coarse optimum."""
    r0, p0 = out["rotation"], out["plane"]
    fine = search_pose(xen_feat, vol, scale=scale,
                       rot_range=(r0 - 2.5, r0 + 2.51, 0.5),
                       plane_range=(max(0, p0 - 6), p0 + 7, 2),
                       band_half=band_half, how=how, downsample=2, refine=False,
                       feat_sigma=out.get("feat_sigma", 3), min_cover=out.get("min_cover", 0.4),
                       anchor_rc=out.get("anchor_rc"), anchor_radius_px=out.get("anchor_radius_px", 120))
    return fine if fine["score"] >= out["score"] else out


# ------------------------------------------------------------ comparison to GT
def pose_error(M, M_gt, image_shape, n=400):
    """RMS displacement (z-stack px) between two affines over a grid of Xenium points,
    plus the implied micron error.  image_shape = Xenium (H,W)."""
    H, W = image_shape
    ys = np.linspace(0, H - 1, int(np.sqrt(n)))
    xs = np.linspace(0, W - 1, int(np.sqrt(n)))
    gy, gx = np.meshgrid(ys, xs)
    pts = np.vstack([gy.ravel(), gx.ravel(), np.ones(gy.size)])      # (row,col,1)
    a = M @ pts; b = M_gt @ pts
    d = np.hypot(a[0] - b[0], a[1] - b[1])
    return dict(rms_px=float(np.sqrt((d**2).mean())), med_px=float(np.median(d)),
                rms_um=float(np.sqrt((d**2).mean()) * ZSTACK_XY_UM))
