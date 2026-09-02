"""Ported from transform_xenium_points.ipynb: maps every Xenium cell centroid (not just matched
ones) into 3D z-stack coordinates two ways, matching the reference's own "nr"/"r" convention:
  - non-rigid (TPS, "_nr"): the section's fitted thin-plate-spline (the same interpolator
    propagate_format.py fits per section, refit here from the saved
    post_affine_warping/section_N.csv landmark table).
  - rigid (affine + 3D rotation, "_r"): `get_2d_to_3d_transform_func`, using the section's 2D
    affine (Affine matrices/section_N_affine_matrix.npy) and the 3D tilt-correction rotation R
    fit once at the anchor section (`initial_match.search_anchor_section`'s `find_min_z_spread_
    rotation` on the initial landmarks' z-stack z-values, saved by propagate_format.py as
    `Affine matrices/section_N_rotation_3d.npy` -- identical for every section, matching the
    reference's own convention of computing R_3d once where real landmarks exist and inheriting it
    unchanged for every auto-propagated section). Falls back to identity if that file is missing.

`zstack_mask_properties.csv` (regionprops on the z-stack segmentation volume) is generated here
since nothing in the reference codebase produces it, and it's a directly-derivable cache file, not
missing input data."""
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import Rbf
from skimage.measure import regionprops_table
from .reference_ported import get_2d_to_3d_transform_func


def write_zstack_mask_properties(cfg, out_path):
    seg = tiff.imread(cfg.zstack_segmented_tif)
    props = regionprops_table(seg, properties=("label", "centroid"))
    df = pd.DataFrame({
        "label": props["label"],
        "centroid_z_um": props["centroid-0"] * 1.0,     # z-step is 1um/plane
        "centroid_y_um": props["centroid-1"] * cfg.zstack_xy_um,
        "centroid_x_um": props["centroid-2"] * cfg.zstack_xy_um,
    })
    df.to_csv(out_path, index=False)
    return df


def map_section_to_3d(cfg, sec, out_dir, zstack_shape):
    out_dir = Path(out_dir)
    paw_path = out_dir / "post_affine_warping" / f"section_{sec}.csv"
    table = pd.read_csv(paw_path, header=None)
    # columns: 0 id, 1 valid, 2 z_zst, 3 y_zst, 4 x_zst, 5 z_xen, 6 y_xen, 7 x_xen(+jitter), 8 corr
    xyz_zst = np.column_stack([table[4], table[3], table[2]]).astype(float)   # x,y,z
    xyz_xen = np.column_stack([table[7], table[6], table[5]]).astype(float)

    zeros = np.zeros_like(xyz_xen[:, 0])
    interp_x = Rbf(xyz_xen[:, 0], xyz_xen[:, 1], zeros, xyz_zst[:, 0], function="thin_plate")
    interp_y = Rbf(xyz_xen[:, 0], xyz_xen[:, 1], zeros, xyz_zst[:, 1], function="thin_plate")
    interp_z = Rbf(xyz_xen[:, 0], xyz_xen[:, 1], zeros, xyz_zst[:, 2], function="thin_plate")

    masks = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"))
    props = regionprops_table(masks, properties=("label", "centroid"))
    cx, cy = props["centroid-1"], props["centroid-0"]   # x=col, y=row
    cz = np.zeros_like(cx)

    x3 = interp_x(cx, cy, cz)
    y3 = interp_y(cx, cy, cz)
    z3 = interp_z(cx, cy, cz)

    M = np.load(out_dir / "Affine matrices" / f"section_{sec}_affine_matrix.npy")
    M3, z_base = M[:3, :3], M[3, 2]
    rot_path = out_dir / "Affine matrices" / f"section_{sec}_rotation_3d.npy"
    R_3d = np.load(rot_path) if rot_path.exists() else np.eye(3)
    rigid_map = get_2d_to_3d_transform_func(R_3d, np.linalg.inv(M3), z_base, zstack_shape)
    xyz_r = rigid_map(np.column_stack([cx, cy]))

    df = pd.DataFrame({"mask_id": props["label"], "x_xenium_aligned_px": cx, "y_xenium_aligned_px": cy,
                       "x_zstack_nr": x3, "y_zstack_nr": y3, "z_zstack_nr": z3,
                       "x_zstack_r": xyz_r[:, 0], "y_zstack_r": xyz_r[:, 1], "z_zstack_r": xyz_r[:, 2],
                       "section": sec})
    df.to_csv(out_dir / "mapped_3d_coordinates" / f"section_{sec}_3d_centroids.csv", index=False)
    return df


def _map_one(args):
    cfg, sec, out_dir, zstack_shape = args
    paw_path = Path(out_dir) / "post_affine_warping" / f"section_{sec}.csv"
    if not paw_path.exists():
        print(f"[transform_xenium_points] section {sec}: no post_affine_warping table, skipping", flush=True)
        return None
    print(f"[transform_xenium_points] section {sec}", flush=True)
    return map_section_to_3d(cfg, sec, out_dir, zstack_shape)


def run_transform_xenium_points(cfg, out_dir, sections, max_workers=14):
    """Each section maps independently (its own post_affine_warping table + affine) --
    parallelized across sections."""
    out_dir = Path(out_dir)
    (out_dir / "mapped_3d_coordinates").mkdir(parents=True, exist_ok=True)

    zprops_path = out_dir / "ophys-z-stacks_segmentation_masks" / f"{cfg.subject_id}_zstack_mask_properties.csv"
    zprops_path.parent.mkdir(parents=True, exist_ok=True)
    write_zstack_mask_properties(cfg, zprops_path)

    with tiff.TiffFile(cfg.zstack_registered_tif) as tf:
        zstack_shape = (len(tf.pages),) + tf.pages[0].shape

    args = [(cfg, sec, out_dir, zstack_shape) for sec in sections]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        all_dfs = [d for d in ex.map(_map_one, args) if d is not None]

    if all_dfs:
        total = pd.concat(all_dfs, axis=0).reset_index(drop=True)
        total.to_csv(out_dir / "mapped_3d_coordinates" / "all_sections_3d_centroids.csv", index=False)
        return total
    return None
