"""Read/write the BigWarp ground-truth formats: the z-stack<->Xenium
`*_warp_project.json` thin-plate-spline landmark files and the
`Affine matrices/section_N_affine_matrix.npy` files.

Landmark/affine convention (verified against the GT-producing notebooks):
  * movingPoints = z-stack (x, y, plane); fixedPoints = Xenium (x, y, 0).
  * The 2D affine is fit in (row, col) space, i.e. on `pts[:, 1::-1]`, and maps
    Xenium pixels -> z-stack pixels:  M = find_affine_transformation_2d(xen_rc, z_rc).
"""
import json
import numpy as np
from pathlib import Path
from .geometry import find_affine_transformation_2d


def read_landmarks(json_path, min_active=3):
    """Return (moving_xyz, fixed_xyz, names) for ACTIVE landmarks, or None if < min_active.
    moving = z-stack (x,y,plane); fixed = Xenium (x,y,z)."""
    d = json.load(open(json_path))
    lm = d["Transform"]["landmarks"]
    act = lm.get("active", [])
    if not act or int(np.sum(act)) < min_active:
        return None
    mov = np.array([p for i, p in enumerate(lm["movingPoints"]) if act[i]], float)
    fix = np.array([p for i, p in enumerate(lm["fixedPoints"]) if act[i]], float)
    names = [n for i, n in enumerate(lm.get("names", [])) if act[i]]
    return mov, fix, names


def affine_from_landmarks(moving_xyz, fixed_xyz):
    """Fit the GT-convention 3x3 affine (Xenium px -> z-stack px) and the section plane.
    Returns (M_3x3, plane). Inputs are (N,3) [x,y,z] arrays."""
    z_rc = moving_xyz[:, 1::-1]          # (x,y)->(row,col)
    xen_rc = fixed_xyz[:, 1::-1]
    M = find_affine_transformation_2d(xen_rc, z_rc)
    plane = int(np.median(moving_xyz[:, 2]))
    return M, plane


def save_affine(path, M, z_base=None):
    """Save a (3,3) affine, or (4,3) with a trailing [0,0,z_base] row (the automatic
    pipeline's format) when z_base is given."""
    M = np.asarray(M)[:3, :3]
    arr = M if z_base is None else np.concatenate([M, [[0, 0, z_base]]], axis=0)
    np.save(path, arr)
    return arr


def load_affine(path):
    """Return (M_3x3, z_base or nan)."""
    a = np.load(path)
    return a[:3, :3], (float(a[3, 2]) if a.shape[0] == 4 else np.nan)


def write_warp_json(template_path, out_path, M, z_base, image_shape,
                    xenium_uri=None):
    """Write a BigWarp warp_project.json whose landmarks encode the affine `M`
    (Xenium->z-stack) as four corner correspondences at plane `z_base` — the trick the
    automatic notebook uses to round-trip an affine back into BigWarp. `image_shape`=(H,W)."""
    d = json.load(open(template_path))
    H, W = image_shape
    Minv = np.linalg.inv(M)
    moving = [[0, 0, z_base], [0, W, z_base], [H, 0, z_base], [H, W, z_base]]
    fixed = [list(Minv[:2, :2] @ np.array(p[1::-1]) + Minv[:2, 2]) + [0] for p in moving]
    lm = d["Transform"]["landmarks"]
    lm["movingPoints"] = moving
    lm["fixedPoints"] = fixed
    lm["active"] = [True] * len(moving)
    lm["names"] = [f"pt-{i}" for i in range(len(moving))]
    if xenium_uri:
        d["Sources"]["2"]["uri"] = xenium_uri
    json.dump(d, open(out_path, "w"))
    return out_path
