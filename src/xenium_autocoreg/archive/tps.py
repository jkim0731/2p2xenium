"""Thin-plate-spline warp of a z-stack plane into Xenium space from tile correspondences
(the RBF step in alignment_matching / xenium_automatic_coreg).

Mirrors BigWarp's TPS: fit RBF(thin_plate) interpolators X,Y -> (x,y,z) in z-stack space
from the matched tile centres, then resample the volume onto the Xenium grid.
"""
import numpy as np
from scipy.interpolate import Rbf


def fit_tps(xyz_xenium, xyz_zstack):
    """Fit thin-plate RBF interpolators (x_xen, y_xen) -> x/y/z in z-stack space.
    `xyz_*` are (N,3) [x,y,z]; the z of xenium is ignored (planar source)."""
    zero = np.zeros_like(xyz_xenium[:, 0])
    fx = Rbf(xyz_xenium[:, 0], xyz_xenium[:, 1], zero, xyz_zstack[:, 0], function="thin_plate")
    fy = Rbf(xyz_xenium[:, 0], xyz_xenium[:, 1], zero, xyz_zstack[:, 1], function="thin_plate")
    fz = Rbf(xyz_xenium[:, 0], xyz_xenium[:, 1], zero, xyz_zstack[:, 2], function="thin_plate")
    return fx, fy, fz


def warp_plane(volume, fx, fy, fz, out_shape=None, order=0):
    """Resample `volume` (Z,Y,X) onto the Xenium grid using fitted TPS interpolators.
    order=0 (nearest) preserves label ids; order=1 for intensity (done by caller upstream)."""
    H, W = out_shape or volume.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W]
    xn = np.clip(fx(xx, yy, np.zeros_like(xx)), 0, volume.shape[2] - 1).astype(int)
    yn = np.clip(fy(xx, yy, np.zeros_like(xx)), 0, volume.shape[1] - 1).astype(int)
    zn = np.clip(fz(xx, yy, np.zeros_like(xx)), 0, volume.shape[0] - 1).astype(int)
    out = np.zeros((H, W), dtype=volume.dtype)
    out[yy, xx] = volume[zn, yn, xn]
    return out


def tps_warp_from_landmarks(volume, zyx_xenium, zyx_zstack, best_corrs,
                            corr_threshold=0.2, out_shape=None):
    """Convenience: filter tile correspondences by correlation, fit TPS, warp the volume.
    `zyx_*` are (N,3) [z,y,x] tile correspondences from tile_based_warping."""
    best = np.asarray(best_corrs)
    keep = best > corr_threshold
    xyz_xen = zyx_xenium[keep][:, ::-1]   # -> x,y,z
    xyz_z = zyx_zstack[keep][:, ::-1]
    fx, fy, fz = fit_tps(xyz_xen, xyz_z)
    return warp_plane(volume, fx, fy, fz, out_shape)
