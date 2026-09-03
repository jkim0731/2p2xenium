"""
Unified CLI entry point. Replaces the old `run_subject.py`/`run_full_pipeline.py`/`pipeline.py`
(now in `archive/`) with one command that wires the validated stage order:

    pose_seed (mode 1/2/3) -> chain_refine.run_chain (all sections) ->
    fine_registration.register_section (per section) -> copy_zstack_assets ->
    cell_centroids -> transform_xenium_points -> concatenated total CSV

This module is data-layout-agnostic: `run_subject` takes a `SubjectConfig` directly (built by hand,
or loaded from a plain JSON file via `config.subject_config_from_json`) -- it does not resolve
anything from a mounted-asset naming convention itself. For a specific lab's asset layout (e.g. a
CodeOcean capsule), write your own resolver that builds a `SubjectConfig` and either call
`run_subject(cfg, ...)` directly from Python, or write it out as a JSON file for this CLI -- see
https://github.com/AllenNeuralDynamics/ophys-xenium-autocoreg for a reference implementation.

Usage:
    xenium-autocoreg <config.json> <out_dir> --pose-mode auto [--anchor-sec N]
    xenium-autocoreg <config.json> <out_dir> --pose-mode center-rotation --center-um X,Y --rotation-deg R [--scale S]
    xenium-autocoreg <config.json> <out_dir> --pose-mode corners --xenium-trapezoid-corners-um X1,Y1,X2,Y2,X3,Y3,X4,Y4 --top-edge N   # NOT YET IMPLEMENTED upstream (pose_seed.seed_from_corners) -- parameters are wired through for forward-compatibility

`<config.json>` matches `SubjectConfig`'s own field names (see `config.subject_config_from_json`):
    {"subject_id": 816462, "aligned_dir": "...", "zstack_registered_tif": "...",
     "zstack_segmented_tif": "...", "zstack_xy_um": 1.367, "reporter_zarr_root": "..."}

`--pose-json PATH` is an alternative to inline flags for mode 2/3, pointing at a JSON file:
    {"center_um": [x, y], "rotation_deg": r, "scale": s}   (mode 2; "scale" optional --
    defaults to the subject's own SubjectConfig.zstack_scale_to_Xenium if omitted)
    {"xenium_trapezoid_corners_um": [[x,y],[x,y],[x,y],[x,y]], "top_edge": 0,
     "zstack_corners_um": [[x,y],[x,y],[x,y],[x,y]], "scale": s}   (mode 3, NOT YET IMPLEMENTED --
    "zstack_corners_um" and "scale" optional)

`--num-cpus N` controls worker-process count for every parallelized stage (see
`resources.resolve_num_cpus`): blank/0/N > this machine's CPU count -> auto (every available
core); N=1 -> serial, no multiprocessing at all (useful for debugging, or a resource-constrained
sandbox where spawning many worker processes gets silently killed).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops_table

from .config import subject_config_from_json
from .anchor import select_anchor_section
from .chain_refine import run_chain
from .fine_registration import register_section
from .copy_zstack_assets import copy_zstack_assets
from .transform_xenium_points import run_transform_xenium_points
from .reference_ported import rotate_volume
from .resources import pool_map
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


def write_cell_centroids(cfg, out_dir, sections, num_cpus=None):
    (Path(out_dir) / "cell_centroids").mkdir(parents=True, exist_ok=True)
    pool_map(_centroids_one, [(cfg, out_dir, s) for s in sections], num_cpus)


def run_subject(cfg, out_dir, pose_mode="auto", anchor_sec=None, center_um=None,
                rotation_deg=None, scale=None, xenium_trapezoid_corners_um=None, top_edge=None,
                zstack_corners_um=None, num_cpus=None, verbose=True):
    """Run the full pipeline for one subject, given its `SubjectConfig` (`cfg`) -- see the module
    docstring for how to obtain one (directly, or via `config.subject_config_from_json`).

    `pose_mode` selects which of the 3 pose-seeding protocols seeds the anchor section's initial
    pose (see `pose_seed`'s module docstring for the algorithm):
        "auto"            -- fully automatic; only `anchor_sec` is relevant (optional -- blank
                             auto-selects via `anchor.select_anchor_section`).
        "center-rotation" -- needs `center_um` + `rotation_deg` (`scale` optional).
        "corners"         -- needs `xenium_trapezoid_corners_um` + `top_edge` (`zstack_corners_um`
                             + `scale` optional). NOT YET IMPLEMENTED upstream
                             (`pose_seed.seed_from_corners` raises `NotImplementedError`) -- these
                             parameters are wired through now so no caller changes will be needed
                             once it is.

    `num_cpus`: worker-process count for every parallelized stage (the auto pose-grid search, the
    per-section fine-registration tile correlation, cell-centroid extraction, and 3D point
    mapping). See `resources.resolve_num_cpus`: None/0/a value exceeding this machine's CPU count
    -> auto (every available core); 1 -> serial, no multiprocessing at all.
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    subject_id = cfg.subject_id
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if anchor_sec is None:
        anchor_sec, _ = select_anchor_section(cfg, verbose=verbose)

    log(f"[{subject_id}] pose-seed mode={pose_mode}, anchor section={anchor_sec}")
    if pose_mode == "auto":
        seed_result = ps.seed_from_auto_search(cfg, anchor_sec, num_cpus=num_cpus, verbose=verbose)
    elif pose_mode == "center-rotation":
        seed = ps.seed_from_center_rotation(center_um, rotation_deg, scale, cfg=cfg)
        seed_result = ps.refine_from_seed(cfg, anchor_sec, seed, verbose=verbose)
    elif pose_mode == "corners":
        # NOT YET IMPLEMENTED upstream -- see pose_seed.seed_from_corners's docstring. Parameters
        # are validated + passed through here so this call site needs no changes once it is
        # implemented (expected to return the same dict(M3, R_3d, plane, ...) shape as the other
        # two modes, since that's what's unpacked immediately below).
        seed_result = ps.seed_from_corners(xenium_trapezoid_corners_um, top_edge,
                                           zstack_corners_um=zstack_corners_um, scale=scale)
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
        summary.append(register_section(cfg, sec, M, rec["z_base"], R_3d, zstack_r, zstack_masks_r,
                                        out_dir, num_cpus=num_cpus))
    json.dump(summary, open(out_dir / "propagation_summary.json", "w"), indent=2)

    log(f"[{subject_id}] stage: concatenated cell_matching_probability totals")
    cmp_dir = out_dir / "cell_matching_probability"
    tables = [pd.read_csv(cmp_dir / f"section_{s}_matching_results.csv") for s in sections]
    pd.concat(tables, axis=0).reset_index(drop=True).to_csv(
        cmp_dir / f"mouse_{subject_id}_total_matching_results.csv", index=False)

    log(f"[{subject_id}] stage: cell_centroids")
    write_cell_centroids(cfg, out_dir, sections, num_cpus=num_cpus)

    log(f"[{subject_id}] stage: transform_xenium_points")
    run_transform_xenium_points(cfg, out_dir, sections, num_cpus=num_cpus)

    log(f"[{subject_id}] DONE -> {out_dir}")
    return summary


def _parse_points(s, n):
    """Parse a flat comma-separated string of `2*n` floats into a list of `n` (x, y) tuples."""
    vals = [float(v) for v in s.split(",")]
    if len(vals) != 2 * n:
        raise ValueError(f"expected {2 * n} comma-separated numbers ({n} x,y pairs), got {len(vals)}")
    return [(vals[i], vals[i + 1]) for i in range(0, 2 * n, 2)]


_COORD_FLAGS = ("--center-um", "--xenium-trapezoid-corners-um", "--zstack-corners-um")


def _fix_negative_coord_tokens(argv):
    """Rewrite `--flag value` to `--flag=value` for the comma-joined coordinate flags, so a value
    starting with a minus sign (e.g. `-50,-100`) isn't misparsed by argparse as a new option --
    argparse's own negative-number heuristic only recognizes a bare `-123`/`-1.5`, not a
    comma-separated list, so `--center-um -50,-100` would otherwise fail with 'expected one
    argument' before this module's own validation ever runs."""
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _COORD_FLAGS and i + 1 < len(argv):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def main(argv=None):
    argv = _fix_negative_coord_tokens(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="xenium-autocoreg")
    p.add_argument("config", help="path to a SubjectConfig JSON file (see config.subject_config_from_json)")
    p.add_argument("out_dir")
    p.add_argument("--pose-mode", choices=["auto", "center-rotation", "corners"], default="auto")
    p.add_argument("--anchor-sec", type=int, default=None)
    p.add_argument("--center-um", default=None, help="X,Y (comma-separated) for --pose-mode center-rotation")
    p.add_argument("--rotation-deg", type=float, default=None)
    p.add_argument("--scale", type=float, default=None,
                  help="z-stack-to-Xenium scale factor (center-rotation or corners); defaults to "
                       "the subject's own config value if omitted")
    p.add_argument("--xenium-trapezoid-corners-um", default=None,
                  help="X1,Y1,X2,Y2,X3,Y3,X4,Y4 (comma-separated, 4 x,y pairs) for --pose-mode corners")
    p.add_argument("--top-edge", type=int, default=None,
                  help="which corner/edge (0-3) is the trapezoid's short/slanted top, for --pose-mode corners")
    p.add_argument("--zstack-corners-um", default=None,
                  help="X1,Y1,X2,Y2,X3,Y3,X4,Y4 (comma-separated, 4 x,y pairs), OPTIONAL for "
                       "--pose-mode corners; defaults to the z-stack's own canonical FOV rectangle")
    p.add_argument("--pose-json", default=None, help="JSON file alternative to the inline flags above")
    p.add_argument("--num-cpus", type=int, default=None,
                  help="worker-process count for every parallelized stage. Blank/0/a value "
                       "exceeding this machine's CPU count = auto (every available core); "
                       "1 = serial, no multiprocessing at all. See resources.resolve_num_cpus.")
    args = p.parse_args(argv)

    center_um = _parse_points(args.center_um, 1)[0] if args.center_um else None
    rotation_deg, scale = args.rotation_deg, args.scale
    xenium_trapezoid_corners_um = (_parse_points(args.xenium_trapezoid_corners_um, 4)
                                   if args.xenium_trapezoid_corners_um else None)
    top_edge = args.top_edge
    zstack_corners_um = (_parse_points(args.zstack_corners_um, 4)
                        if args.zstack_corners_um else None)

    if args.pose_json:
        data = json.load(open(args.pose_json))
        center_um = tuple(data["center_um"]) if "center_um" in data else center_um
        rotation_deg = data.get("rotation_deg", rotation_deg)
        scale = data.get("scale", scale)
        if "xenium_trapezoid_corners_um" in data:
            xenium_trapezoid_corners_um = [tuple(pt) for pt in data["xenium_trapezoid_corners_um"]]
        top_edge = data.get("top_edge", top_edge)
        if "zstack_corners_um" in data:
            zstack_corners_um = [tuple(pt) for pt in data["zstack_corners_um"]]

    cfg = subject_config_from_json(args.config)
    run_subject(cfg, args.out_dir, pose_mode=args.pose_mode, anchor_sec=args.anchor_sec,
               center_um=center_um, rotation_deg=rotation_deg, scale=scale,
               xenium_trapezoid_corners_um=xenium_trapezoid_corners_um, top_edge=top_edge,
               zstack_corners_um=zstack_corners_um, num_cpus=args.num_cpus)


if __name__ == "__main__":
    sys.exit(main())
