"""Section-to-section alignment primitives (tissue contour + rigid ICP), curated from
alignment_fucntions.py. Used to register all Xenium sections to a reference section
(default section 4) before z-stack registration.  The interactive GUI lives in the
original `interactive_section_alignment.py`."""
import numpy as np
from scipy.spatial.distance import cdist
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours
from skimage.transform import EuclideanTransform
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation


def get_tissue_contour(image_norm, downsample=4):
    """Segment tissue from dark background (mean over channels, Otsu) and return the
    longest contour (full-res coords) + the tissue mask."""
    combined = np.mean(image_norm, axis=0)[::downsample, ::downsample]
    sm = gaussian(combined, sigma=5)
    mask = sm > threshold_otsu(sm) * 0.5
    mask = binary_fill_holes(mask)
    mask = binary_erosion(binary_dilation(mask, iterations=3), iterations=3)
    contour = max(find_contours(mask.astype(float), 0.5), key=len) * downsample
    return contour, mask


def icp_rigid(src_points, dst_points, max_iterations=100, tolerance=1e-6):
    """Rigid (rotation+translation) ICP aligning src->dst (both (N,2) in (row,col)).
    Returns (EuclideanTransform, final_error, cumulative_R, cumulative_t)."""
    src = src_points.copy()
    dst_sub = (dst_points[np.random.choice(len(dst_points), 2000, replace=False)]
               if len(dst_points) > 2000 else dst_points)
    prev = np.inf
    cR, ct = np.eye(2), np.zeros(2)
    for it in range(max_iterations):
        if len(src) > 2000:
            idx = np.random.choice(len(src), 2000, replace=False); src_sub = src[idx]
        else:
            src_sub = src
        d = cdist(src_sub, dst_sub)
        nn = np.argmin(d, axis=1); nearest = dst_sub[nn]
        dist = np.min(d, axis=1)
        good = dist < 3 * np.median(dist)
        if good.sum() < 10:
            break
        sg, dg = src_sub[good], nearest[good]
        sc, dc = sg.mean(0), dg.mean(0)
        H = (sg - sc).T @ (dg - dc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = dc - R @ sc
        src = (R @ src.T).T + t
        cR = R @ cR; ct = R @ ct + t
        err = np.mean(dist[good])
        if abs(prev - err) < tolerance:
            break
        prev = err
    angle = np.arctan2(cR[1, 0], cR[0, 0])
    tf = EuclideanTransform(rotation=-angle, translation=(ct[1], ct[0]))
    return tf, prev, cR, ct
