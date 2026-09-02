"""Piecewise (2.5D) tile cross-correlation — the image-based nonrigid step.

For each tile of the fixed (affine-warped Xenium) image, search a (z, y, x) window of the
moving z-stack volume for the best normalized-cross-correlation match.  Returns matched
tile-centre correspondences (z,y,x) in both frames + correlation scores, which are then fed
to a thin-plate spline (tps.py).  This is the fast, FFT/integral-image vectorized version
from alignment_fucntions.tile_based_warping (identical results, ~100x faster than the
triple-loop original).
"""
import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm


def tile_based_warping(img, vol, z_base, margin=5, tile_size=(64, 64),
                       overlap=0.4, max_shift=(10, 20, 20), progress=True):
    """
    Parameters
    ----------
    img : (H, W) fixed 2D image (e.g. affine-warped Xenium Neurons or binary masks).
    vol : (Z, H, W) moving z-stack volume (same channel/feature as `img`).
    z_base : centre plane of the z search.
    tile_size, overlap, margin : tiling grid.
    max_shift : (dz, dy, dx) half-window of the search.

    Returns
    -------
    zyx_img  : (n_tiles, 3) tile centres in the fixed image (z=0 column).
    zyx_vol  : (n_tiles, 3) matched (z, y, x) in the moving volume.
    best_corrs, base_corrs : per-tile NCC of the best match and of the no-shift baseline.
    """
    tile_size = np.asarray(tile_size)
    step = (tile_size * (1 - overlap)).astype(int)
    y_size, x_size = img.shape
    ni = int((y_size - 2 * margin - tile_size[0]) / step[0]) + 1
    nj = int((x_size - 2 * margin - tile_size[1]) / step[1]) + 1
    ii, jj = np.mgrid[:ni, :nj]
    tile_starts = np.column_stack([(margin + ii.ravel() * step[0]),
                                   (margin + jj.ravel() * step[1])])
    tile_starts = (tile_starts - tile_starts.mean(0) - tile_size / 2
                   + np.array(img.shape) / 2).astype(int)
    tile_centers = tile_starts + tile_size // 2
    n = len(tile_starts)

    ty, tx = tile_size
    n_pix = int(ty * tx)
    mz, my, mx = max_shift
    z_range = np.arange(-mz, mz + 1)
    H, W = img.shape

    best_corrs = np.full(n, -np.inf)
    best_shifts = np.zeros((n, 3), int); best_shifts[:, 0] = z_base
    base_corrs = np.empty(n)
    img_f = img.astype(np.float64)

    it = tqdm(tile_starts) if progress else tile_starts
    for t, (y0, x0) in enumerate(it):
        fixed = img_f[y0:y0 + ty, x0:x0 + tx]
        f_cent = fixed - fixed.mean(); f_std = f_cent.std()
        mov0 = vol[z_base, y0:y0 + ty, x0:x0 + tx].astype(np.float64)
        base_corrs[t] = np.corrcoef(mov0.ravel(), fixed.ravel())[0, 1]
        best_corrs[t] = base_corrs[t]
        if f_std == 0:
            continue
        kernel = f_cent[::-1, ::-1]
        sy0, sy1 = max(0, y0 - my), min(H, y0 + ty + my)
        sx0, sx1 = max(0, x0 - mx), min(W, x0 + tx + mx)
        for z in z_range:
            zz = z_base + z
            if zz < 0 or zz >= vol.shape[0]:
                continue
            region = vol[zz, sy0:sy1, sx0:sx1].astype(np.float64)
            cc = fftconvolve(region, kernel, mode="valid")
            cum = np.cumsum(np.cumsum(region, 0), 1)
            cum2 = np.cumsum(np.cumsum(region ** 2, 0), 1)
            rh, rw = region.shape
            P = np.zeros((rh + 1, rw + 1)); P[1:, 1:] = cum
            P2 = np.zeros((rh + 1, rw + 1)); P2[1:, 1:] = cum2
            vy, vx = cc.shape
            IY, IX = np.mgrid[:vy, :vx]; BY, BX = IY + ty, IX + tx
            s1 = P[BY, BX] - P[IY, BX] - P[BY, IX] + P[IY, IX]
            s2 = P2[BY, BX] - P2[IY, BX] - P2[BY, IX] + P2[IY, IX]
            t_std = np.sqrt(np.maximum(s2 / n_pix - (s1 / n_pix) ** 2, 0))
            denom = n_pix * t_std * f_std
            with np.errstate(divide="ignore", invalid="ignore"):
                cmap = np.where(denom > 0, cc / denom, -2.0)
            pos = np.unravel_index(cmap.argmax(), cmap.shape)
            if cmap[pos] > best_corrs[t]:
                best_corrs[t] = cmap[pos]
                best_shifts[t] = (zz, sy0 + pos[0] - y0, sx0 + pos[1] - x0)

    zyx_img = np.hstack((np.zeros((n, 1)), tile_centers))
    zyx_vol = zyx_img + best_shifts
    return zyx_img, zyx_vol, best_corrs.tolist(), base_corrs.tolist()
