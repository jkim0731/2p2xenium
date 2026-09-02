"""
Unified CLI entry point. Replaces the old `run_subject.py`/`run_full_pipeline.py`/`pipeline.py`
(now in `archive/`) with one command that wires the validated stage order:

    pose_seed (mode 1/2/3) -> chain_refine.run_chain (all sections) ->
    fine_registration.register_section (per section) -> copy_zstack_assets ->
    cell_centroids -> transform_xenium_points -> concatenated total CSV

Usage:
    xenium-autocoreg <subject_id> <out_dir> --pose-mode auto
    xenium-autocoreg <subject_id> <out_dir> --pose-mode center-rotation --center-um X,Y --rotation-deg R [--anchor-sec N]
    xenium-autocoreg <subject_id> <out_dir> --pose-mode corners --pose-json seed.json   # NOT YET IMPLEMENTED

`--pose-json PATH` is an alternative to inline flags for mode 2/3, pointing at a JSON file:
    {"center_um": [x, y], "rotation_deg": r, "scale": 0.8}   (mode 2; "scale" optional --
    defaults to the subject's own SubjectConfig.tissue_expansion_scale if omitted)
    {"xenium_trapezoid_corners_um": [[x,y],[x,y],[x,y],[x,y]], "top_edge": 0}      (mode 3, TODO)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops_table

from .config import resolve_subject
from .anchor import select_anchor_section
from .chain_refine import run_chain
from .fine_registration import register_section
from .copy_zstack_assets import copy_zstack_assets
from .transform_xenium_points import run_transform_xenium_points
from .reference_ported import rotate_volume
from . import pose_seed as ps


def _centroids_one(args):
    cfg, out_dir, sec = args
    cdir = Path(out_dir) / "cell_centroids"
    xen_path = cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"
    if xen_path.exists():
        m = np.squeeze(tiff.imread(xen_path))
        props = regionprops_table(m, properties=("label", "centroid"))
        pd.DataFrame({"mask_id": props["label"], "centroid_y": props["centroid-0"],
                     "centroid_x": props["centroid-1"]}).to_csv(
            cdir / f"section_{sec}_xenium_centroids.csv", index=False)
    ophys_path = Path(out_dir) / "warped_zstacks" / f"section_{sec}_zstack_warped_masks_plane.tif"
    if ophys_path.exists():
        m = tiff.imread(ophys_path)
        props = regionprops_table(m, properties=("label", "centroid"))
        pd.DataFrame({"mask_id": props["label"], "centroid_y": props["centroid-0"],
                     "centroid_x": props["centroid-1"]}).to_csv(
            cdir / f"section_{sec}_ophys_centroids.csv", index=False)


def write_cell_centroids(cfg, out_dir, sections, max_workers=14):
    (Path(out_dir) / "cell_centroids").mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_centroids_one, [(cfg, out_dir, s) for s in sections]))


def run_subject(subject_id, out_dir, pose_mode="auto", anchor_sec=None, center_um=None,
                rotation_deg=None, scale=None, verbose=True):
    def log(msg):
        if verbose:
            print(msg, flush=True)

    cfg = resolve_subject(subject_id)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if anchor_sec is None:
        anchor_sec, _ = select_anchor_section(cfg, verbose=verbose)

    log(f"[{subject_id}] pose-seed mode={pose_mode}, anchor section={anchor_sec}")
    if pose_mode == "auto":
        seed_result = ps.seed_from_auto_search(cfg, anchor_sec, verbose=verbose)
    elif pose_mode == "center-rotation":
        seed = ps.seed_from_center_rotation(center_um, rotation_deg, scale, cfg=cfg)
        seed_result = ps.refine_from_seed(cfg, anchor_sec, seed, verbose=verbose)
    elif pose_mode == "corners":
        raise NotImplementedError("pose-mode 'corners' is not yet implemented -- see pose_seed.seed_from_corners")
    else:
        raise ValueError(f"unknown pose_mode {pose_mode!r}")

    anchor_M, anchor_z, R_3d = seed_result["M3"], seed_result["plane"], seed_result["R_3d"]
    log(f"[{subject_id}] anchor pose: plane={anchor_z}, n_landmarks={seed_result['n_landmarks']}, "
        f"tilt={seed_result['tilt_deg']:.3f} deg")

    log(f"[{subject_id}] stage: raw z-stack assets")
    copy_zstack_assets(cfg, out_dir)

    log(f"[{subject_id}] stage: chain_refine coarse propagation across {len(cfg.sections)} sections")
    chain_dir = out_dir / "_chain_internal"
    chain_results = run_chain(cfg, anchor_sec, anchor_M, anchor_z, chain_dir, sections=cfg.sections, R_3d=R_3d)
    sections = [r["sec"] for r in chain_results]
    log(f"[{subject_id}] chain covers sections: {sections}")

    zstack_r = tiff.imread(cfg.zstack_registered_tif)
    if zstack_r.ndim == 4:
        zstack_r = zstack_r[:, 0]
    zstack_masks_r = tiff.imread(cfg.zstack_segmented_tif)
    if not np.allclose(R_3d, np.eye(3)):
        zstack_r = rotate_volume(zstack_r, R_3d, order=1)
        zstack_masks_r = rotate_volume(zstack_masks_r, R_3d, order=0)

    log(f"[{subject_id}] stage: fine registration (mask-TPS + probability matching) per section")
    summary = []
    for rec in chain_results:
        sec = rec["sec"]
        M = np.load(chain_dir / f"section_{sec}_affine.npy")
        summary.append(register_section(cfg, sec, M, rec["z_base"], R_3d, zstack_r, zstack_masks_r, out_dir))
    json.dump(summary, open(out_dir / "propagation_summary.json", "w"), indent=2)

    log(f"[{subject_id}] stage: concatenated cell_matching_probability totals")
    cmp_dir = out_dir / "cell_matching_probability"
    tables = [pd.read_csv(cmp_dir / f"section_{s}_matching_results.csv") for s in sections]
    pd.concat(tables, axis=0).reset_index(drop=True).to_csv(
        cmp_dir / f"mouse_{subject_id}_total_matching_results.csv", index=False)

    log(f"[{subject_id}] stage: cell_centroids")
    write_cell_centroids(cfg, out_dir, sections)

    log(f"[{subject_id}] stage: transform_xenium_points")
    run_transform_xenium_points(cfg, out_dir, sections)

    log(f"[{subject_id}] DONE -> {out_dir}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="xenium-autocoreg")
    p.add_argument("subject_id", type=int)
    p.add_argument("out_dir")
    p.add_argument("--pose-mode", choices=["auto", "center-rotation", "corners"], default="auto")
    p.add_argument("--anchor-sec", type=int, default=None)
    p.add_argument("--center-um", default=None, help="X,Y (comma-separated) for --pose-mode center-rotation")
    p.add_argument("--rotation-deg", type=float, default=None)
    p.add_argument("--scale", type=float, default=None,
                  help="Xenium-to-z-stack scale prior; defaults to the subject's own config value if omitted")
    p.add_argument("--pose-json", default=None, help="JSON file alternative to the inline flags above")
    args = p.parse_args(argv)

    center_um, rotation_deg, scale = None, args.rotation_deg, args.scale
    if args.center_um:
        center_um = tuple(float(v) for v in args.center_um.split(","))
    if args.pose_json:
        data = json.load(open(args.pose_json))
        center_um = tuple(data["center_um"]) if "center_um" in data else center_um
        rotation_deg = data.get("rotation_deg", rotation_deg)
        scale = data.get("scale", scale)

    run_subject(args.subject_id, args.out_dir, pose_mode=args.pose_mode, anchor_sec=args.anchor_sec,
               center_um=center_um, rotation_deg=rotation_deg, scale=scale)


if __name__ == "__main__":
    sys.exit(main())
