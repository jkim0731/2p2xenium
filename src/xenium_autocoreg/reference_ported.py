"""3D geometry and tile-correlation helpers used throughout the pipeline: fitting/undoing the
z-stack tilt, mapping 2D Xenium points into the original 3D z-stack volume, mask-matching IoU, and
centroid utilities.

`find_affine_transformation_2d` / `apply_affine_based_on_reference_2d` live in `geometry.py`, not
here."""
import numpy as np
from scipy import sparse
from scipy.signal import fftconvolve
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import matplotlib.pyplot as plt


def normalize_to_uint8(img, plow=1, phigh=99):
    """Normalize image to uint8 using percentile-based contrast stretching."""
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, [plow, phigh])
    return np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)


def calculate_iou(mask1, mask2):
    """Intersection over Union of two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_centroid(mask):
    """Centroid (x, y) of a binary mask."""
    y, x = np.where(mask)
    if len(x) == 0:
        return (np.nan, np.nan)
    return np.mean(x), np.mean(y)


def find_mask_matches_fast(labels1, labels2, threshold=0.5):
    """Greedy best-IoU matching between two label images (0=background)."""
    n1 = int(labels1.max())
    n2 = int(labels2.max())
    if n1 == 0 or n2 == 0:
        return [], list(range(n1)), list(range(n2)), np.zeros((n1, n2))

    l1 = labels1.ravel()
    l2 = labels2.ravel()
    overlap = sparse.coo_matrix(
        (np.ones(l1.size, dtype=np.uint32), (l1, l2)), shape=(n1 + 1, n2 + 1)
    ).toarray()

    intersection = overlap[1:, 1:].astype(np.float64)
    sizes1 = overlap[1:, :].sum(axis=1).astype(np.float64)
    sizes2 = overlap[:, 1:].sum(axis=0).astype(np.float64)
    union = sizes1[:, None] + sizes2[None, :] - intersection
    iou_matrix = np.zeros_like(intersection)
    valid = union > 0
    iou_matrix[valid] = intersection[valid] / union[valid]

    best_j = iou_matrix.argmax(axis=1)
    best_iou = iou_matrix[np.arange(n1), best_j]

    matched1, matched2, matches = set(), set(), []
    for i in range(n1):
        if best_iou[i] > threshold:
            j = int(best_j[i])
            matches.append((i, j))
            matched1.add(i)
            matched2.add(j)

    unmatched1 = [i for i in range(n1) if i not in matched1]
    unmatched2 = [j for j in range(n2) if j not in matched2]
    return matches, unmatched1, unmatched2, iou_matrix


def tile_based_warping(img, vol, z_base, margin=5, tile_size=(64, 64), overlap=0.4, max_shift=(10, 20, 20)):
    """Tiled FFT cross-correlation of a 2D image against a 3D volume around z_base -- the exact
    reference algorithm (ported verbatim, only the tqdm progress bar is left as-is)."""
    tile_size = np.asarray(tile_size)
    step = (tile_size * (1 - overlap)).astype(int)
    y_size, x_size = img.shape

    ni = int((y_size - 2 * margin - tile_size[0]) / step[0]) + 1
    nj = int((x_size - 2 * margin - tile_size[1]) / step[1]) + 1
    ii, jj = np.mgrid[:ni, :nj]
    tile_starts = np.column_stack([(margin + ii.ravel() * step[0]), (margin + jj.ravel() * step[1])])
    tile_starts = (tile_starts - tile_starts.mean(axis=0) - tile_size / 2 + np.array(img.shape) / 2).astype(int)
    tile_centers = tile_starts + tile_size // 2
    n_tiles = len(tile_starts)

    ty, tx = tile_size
    n_pix = int(ty * tx)
    max_sz, max_sy, max_sx = max_shift
    z_range = np.arange(-max_sz, max_sz + 1)
    img_h, img_w = img.shape

    best_corrs = np.full(n_tiles, -np.inf)
    best_shifts = np.zeros((n_tiles, 3), dtype=int)
    best_shifts[:, 0] = z_base
    base_corrs = np.empty(n_tiles)
    img_f64 = img.astype(np.float64)

    for t, (y0, x0) in enumerate(tqdm(tile_starts, disable=True)):
        fixed = img_f64[y0:y0 + ty, x0:x0 + tx]
        f_mean = fixed.mean()
        f_cent = fixed - f_mean
        f_std = f_cent.std()

        mov0 = vol[z_base, y0:y0 + ty, x0:x0 + tx].astype(np.float64)
        base_corrs[t] = np.corrcoef(mov0.ravel(), fixed.ravel())[0, 1]
        best_corrs[t] = base_corrs[t]

        if f_std == 0:
            continue

        kernel = f_cent[::-1, ::-1]
        sy0 = max(0, y0 - max_sy)
        sy1 = min(img_h, y0 + ty + max_sy)
        sx0 = max(0, x0 - max_sx)
        sx1 = min(img_w, x0 + tx + max_sx)

        for z in z_range:
            region = vol[z_base + z, sy0:sy1, sx0:sx1].astype(np.float64)
            cc = fftconvolve(region, kernel, mode="valid")
            cum = np.cumsum(np.cumsum(region, axis=0), axis=1)
            cum2 = np.cumsum(np.cumsum(region ** 2, axis=0), axis=1)
            rh, rw = region.shape
            P = np.zeros((rh + 1, rw + 1)); P[1:, 1:] = cum
            P2 = np.zeros((rh + 1, rw + 1)); P2[1:, 1:] = cum2
            vy, vx = cc.shape
            IY, IX = np.mgrid[:vy, :vx]
            BY, BX = IY + ty, IX + tx
            s1 = P[BY, BX] - P[IY, BX] - P[BY, IX] + P[IY, IX]
            s2 = P2[BY, BX] - P2[IY, BX] - P2[BY, IX] + P2[IY, IX]
            t_std = np.sqrt(np.maximum(s2 / n_pix - (s1 / n_pix) ** 2, 0))
            denom = n_pix * t_std * f_std
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_map = np.where(denom > 0, cc / denom, -2.0)
            pos = np.unravel_index(corr_map.argmax(), corr_map.shape)
            if corr_map[pos] > best_corrs[t]:
                best_corrs[t] = corr_map[pos]
                best_shifts[t] = (z_base + z, sy0 + pos[0] - y0, sx0 + pos[1] - x0)

    zyx_img = np.hstack((np.zeros((n_tiles, 1)), tile_centers))
    zyx_vol = zyx_img + best_shifts
    return zyx_img, zyx_vol, best_corrs.tolist(), base_corrs.tolist()


def find_min_z_spread_rotation(points):
    """Ported verbatim from alignment_fucntions.py. Fits a plane z = a*x + b*y + c (least squares)
    through 3D points [x,y,z] and returns the rotation (about X and Y only, closed-form -- no
    optimization) that makes that plane horizontal, i.e. minimizes the z-spread of `points` after
    rotation. Used to recover how the Xenium cutting plane is tilted relative to the z-stack's z
    axis, from the z-values of the initial landmark-matched z-stack ROIs."""
    points = np.asarray(points, dtype=float)
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    coeffs, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    a, b, c = coeffs
    ry = np.arctan2(a, 1.0)
    rx = -np.arctan2(b, np.sqrt(1 + a ** 2))
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R = Ry @ Rx
    rotated_points = points @ R.T
    return R, rotated_points


def rotate_volume(volume, R, order=1):
    """Ported verbatim. Rotates a (Z,Y,X) volume about its center by R (given in [x,y,z] point
    space -- permuted internally to the volume's [z,y,x] index order)."""
    from scipy.ndimage import affine_transform
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    R_vol = P @ R @ P
    center = np.array(volume.shape) / 2.0
    R_vol_inv = R_vol.T
    offset = center - R_vol_inv @ center
    return affine_transform(volume, R_vol_inv, offset=offset, order=order)


def get_2d_to_3d_transform_func(R, affine_matrix, z_base, volume_shape):
    """Ported verbatim from alignment_fucntions.py. Maps (x,y) points from the aligned 2D plane
    back to (x,y,z) in the ORIGINAL 3D z-stack volume, given the 2D affine and a 3D rotation R
    (identity when no per-section tilted-plane fit exists -- see transform_xenium_points.py)."""
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)
    R_vol = P @ R @ P
    center = np.array(volume_shape) / 2.0

    def transform_points(points_2d):
        points_2d = np.atleast_2d(points_2d)
        N = points_2d.shape[0]
        homog_2d = np.vstack([points_2d[:, 0], points_2d[:, 1], np.ones(N)])
        orig_2d_homog = affine_matrix @ homog_2d
        orig_x = orig_2d_homog[0, :] / orig_2d_homog[2, :]
        orig_y = orig_2d_homog[1, :] / orig_2d_homog[2, :]
        coords_rotated_vol = np.vstack([np.full(N, z_base), orig_y, orig_x])
        centered_coords = coords_rotated_vol - center[:, np.newaxis]
        orig_coords_vol = (R_vol.T @ centered_coords) + center[:, np.newaxis]
        z_orig = orig_coords_vol[0, :]
        y_orig = orig_coords_vol[1, :]
        x_orig = orig_coords_vol[2, :]
        return np.vstack([x_orig, y_orig, z_orig]).T

    return transform_points


def plot_alignment_qc(img1, img2, title_1="Image 1", title_2="Image 2", title_overlay="Alignment QC"):
    """Ported verbatim -- 3-panel red/cyan/overlay QC figure. Caller handles savefig/close."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    img_r = normalize_to_uint8(img1)
    img_c = normalize_to_uint8(img2)
    img_rgb_1 = np.stack([img_r, np.zeros_like(img_r), np.zeros_like(img_r)], axis=-1)
    img_rgb_2 = np.stack([np.zeros_like(img_c), img_c, img_c], axis=-1)
    img_overlay = np.stack([img_r, img_c, img_c], axis=-1)
    axes[0].imshow(img_rgb_1); axes[0].set_title(title_1); axes[0].axis("off")
    axes[1].imshow(img_rgb_2); axes[1].set_title(title_2); axes[1].axis("off")
    axes[2].imshow(img_overlay); axes[2].set_title(title_overlay); axes[2].axis("off")
    plt.tight_layout()
    return fig
