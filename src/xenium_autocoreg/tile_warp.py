"""Parallelized tile_based_warping -- same algorithm and outputs as coreg.tile_warp.tile_based_warping
(the per-tile (z,y,x) NCC search is embarrassingly parallel: each tile only reads img/vol, which are
fork-shared read-only, so ProcessPoolExecutor gives a near-linear speedup with no result difference).
Kept as a separate function (not a drop-in replacement of the shared coreg module) until verified;
see verify_parallel_tilewarp.py for the exact-match check against the original serial function."""
import numpy as np
from scipy.signal import fftconvolve
from .resources import pool_map

_IMG_F = _VOL = _TILE = _MAX_SHIFT = _ZBASE = None


def _pool_init(img_f, vol, tile_size, max_shift, z_base):
    global _IMG_F, _VOL, _TILE, _MAX_SHIFT, _ZBASE
    _IMG_F, _VOL, _TILE, _MAX_SHIFT, _ZBASE = img_f, vol, tile_size, max_shift, z_base


def _one_tile(y0x0):
    y0, x0 = y0x0
    ty, tx = _TILE
    n_pix = int(ty * tx)
    mz, my, mx = _MAX_SHIFT
    H, W = _IMG_F.shape
    z_base = _ZBASE

    fixed = _IMG_F[y0:y0 + ty, x0:x0 + tx]
    f_cent = fixed - fixed.mean(); f_std = f_cent.std()
    mov0 = _VOL[z_base, y0:y0 + ty, x0:x0 + tx].astype(np.float64)
    base_corr = np.corrcoef(mov0.ravel(), fixed.ravel())[0, 1]
    best_corr = base_corr
    best_shift = (z_base, 0, 0)
    if f_std == 0:
        return best_corr, base_corr, best_shift

    kernel = f_cent[::-1, ::-1]
    sy0, sy1 = max(0, y0 - my), min(H, y0 + ty + my)
    sx0, sx1 = max(0, x0 - mx), min(W, x0 + tx + mx)
    for z in range(-mz, mz + 1):
        zz = z_base + z
        if zz < 0 or zz >= _VOL.shape[0]:
            continue
        region = _VOL[zz, sy0:sy1, sx0:sx1].astype(np.float64)
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
        if cmap[pos] > best_corr:
            best_corr = cmap[pos]
            best_shift = (zz, sy0 + pos[0] - y0, sx0 + pos[1] - x0)
    return best_corr, base_corr, best_shift


def tile_based_warping_parallel(img, vol, z_base, margin=5, tile_size=(64, 64),
                                overlap=0.4, max_shift=(10, 20, 20), num_cpus=None, reserve_cpus=0,
                                **_ignore):
    """Same signature/outputs as coreg.tile_warp.tile_based_warping (drops the `progress` kwarg;
    absorbed by **_ignore for call-compatibility), parallelized over tiles via ProcessPoolExecutor
    (or run serially if `num_cpus` resolves to that -- see `resources.resolve_num_cpus`)."""
    tile_size_arr = np.asarray(tile_size)
    step = (tile_size_arr * (1 - overlap)).astype(int)
    y_size, x_size = img.shape
    ni = int((y_size - 2 * margin - tile_size_arr[0]) / step[0]) + 1
    nj = int((x_size - 2 * margin - tile_size_arr[1]) / step[1]) + 1
    ii, jj = np.mgrid[:ni, :nj]
    tile_starts = np.column_stack([(margin + ii.ravel() * step[0]),
                                   (margin + jj.ravel() * step[1])])
    tile_starts = (tile_starts - tile_starts.mean(0) - tile_size_arr / 2
                   + np.array(img.shape) / 2).astype(int)
    tile_centers = tile_starts + tile_size_arr // 2
    n = len(tile_starts)

    img_f = img.astype(np.float64)
    out = pool_map(_one_tile, [tuple(p) for p in tile_starts], num_cpus, reserve_cpus=reserve_cpus,
                  initializer=_pool_init,
                  initargs=(img_f, vol, tuple(tile_size), tuple(max_shift), z_base), chunksize=4)

    best_corrs = np.array([o[0] for o in out])
    base_corrs = np.array([o[1] for o in out])
    best_shifts = np.array([o[2] for o in out], int)

    zyx_img = np.hstack((np.zeros((n, 1)), tile_centers))
    zyx_vol = zyx_img + best_shifts
    return zyx_img, zyx_vol, best_corrs.tolist(), base_corrs.tolist()
