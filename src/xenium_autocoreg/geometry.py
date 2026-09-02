"""2D affine + 3D tilt/rotation geometry (curated & de-duplicated from
alignment_fucntions.py and the alignment notebooks).

Conventions (match the ground-truth pipeline exactly):
  * Landmark/point arrays are (x, y) or (x, y, z).  Image/volume indices are (row, col)
    i.e. (y, x) / (z, y, x).  The affine is built in (row, col) space, so callers swap
    point columns with `pts[:, 1::-1]` before `find_affine_transformation_2d` (see bigwarp.py).
  * The 2D affine maps Xenium pixels -> z-stack pixels and is stored as a 3x3 homogeneous
    matrix (the `Affine matrices/section_N_affine_matrix.npy` ground-truth format).
"""
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates

# permutation [x,y,z] <-> [z,y,x]
_P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)


# ---------------------------------------------------------------- 2D affine
def find_affine_transformation_2d(points1, points2):
    """Least-squares 3x3 affine mapping homogeneous points1 -> points2 (each (N,2))."""
    h1 = np.hstack([points1, np.ones((points1.shape[0], 1))])
    h2 = np.hstack([points2, np.ones((points2.shape[0], 1))])
    M, *_ = np.linalg.lstsq(h1, h2, rcond=None)
    return M.T


def apply_affine_based_on_reference_2d(image, affine_matrix, output_shape, order=0):
    """Warp `image` into `output_shape` via inverse-mapping of a 3x3 affine
    (order=0 for label masks, order=1 for intensity)."""
    coords = np.indices(output_shape).reshape(2, -1)
    coords = np.vstack([coords, np.ones(coords.shape[1])])
    new = np.linalg.inv(affine_matrix) @ coords
    out = map_coordinates(image, new[:2], order=order, mode="nearest")
    return out.reshape(output_shape)


def decompose_affine(M):
    """SVD decomposition of a 3x3 (or 2x2) affine. Returns dict with rotation (deg),
    principal scales s1>=s2, shear, area scale (det), translation."""
    A = np.asarray(M)[:2, :2]
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    rot = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    Rr = np.array([[np.cos(np.radians(rot)), -np.sin(np.radians(rot))],
                   [np.sin(np.radians(rot)),  np.cos(np.radians(rot))]])
    K = Rr.T @ A
    shear = K[0, 1] / K[0, 0] if K[0, 0] else np.nan
    t = np.asarray(M)[:2, 2] if np.asarray(M).shape[1] >= 3 else np.zeros(2)
    return dict(rotation_deg=rot, scale1=float(S[0]), scale2=float(S[1]),
                shear=float(shear), area=float(np.linalg.det(A)), translation=t)


# ------------------------------------------------------------ 3D tilt/rotation
def find_min_z_spread_rotation(points):
    """Fit a plane z=a*x+b*y+c to (N,3) [x,y,z] points and return the rotation R
    (in xyz space, Ry@Rx) that levels it, plus the rotated points. Used to flatten a
    tilted Xenium section against the z-stack planes before tile-warping."""
    points = np.asarray(points, float)
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    a, b, c = np.linalg.lstsq(A, points[:, 2], rcond=None)[0]
    ry = np.arctan2(a, 1.0)
    rx = -np.arctan2(b, np.sqrt(1 + a**2))
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R = Ry @ Rx
    return R, points @ R.T


def rotate_volume(volume, R, order=1):
    """Rotate a (Z,Y,X) volume about its centre by R (given in xyz point space)."""
    R_vol = _P @ R @ _P
    center = np.array(volume.shape) / 2.0
    R_inv = R_vol.T
    return affine_transform(volume, R_inv, offset=center - R_inv @ center, order=order)


def make_2d_to_3d_mapper(R, affine_matrix, z_base, volume_shape):
    """Map points from the ORIGINAL 2D Xenium image to their aligned (x,y,z) location
    in the ORIGINAL z-stack volume, inverting the rotate+affine+plane pipeline.
    Returns mapper(x, y) -> (x3, y3, z3). See alignment_fucntions.make_2d_to_3d_mapper."""
    R_vol = _P @ R @ _P
    R_inv = R_vol.T
    center = np.array(volume_shape, float) / 2.0
    A = np.asarray(affine_matrix, float)

    def mapper(x, y):
        x = np.atleast_1d(np.asarray(x, float)); y = np.atleast_1d(np.asarray(y, float))
        aligned = A @ np.vstack([y, x, np.ones_like(x)])
        row_a, col_a = aligned[0] / aligned[2], aligned[1] / aligned[2]
        o = np.vstack([np.full_like(row_a, float(z_base)), row_a, col_a])
        orig = R_inv @ (o - center[:, None]) + center[:, None]
        out = np.vstack([orig[2], orig[1], orig[0]]).T
        return out[0] if out.shape[0] == 1 else out

    return mapper
