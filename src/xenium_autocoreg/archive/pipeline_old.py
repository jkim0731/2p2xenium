"""Top-level orchestration: resolve a subject's data, find an anchor section and its initial
landmarks (GT-free soma-print search), chain-refine across all sections, run fine registration +
cell matching on a QC subset, and generate the 3 QC figures. See README.md for the full picture."""
import json
import numpy as np
from pathlib import Path

from .config import resolve_subject
from .anchor import select_anchor_section
from .initial_match import search_anchor_section
from .chain_refine import run_chain
from .fine_register import fine_register_sections
from . import qc as qc_mod


def _pick_qc_sections(available, anchor_sec, n=6):
    available = sorted(available)
    if len(available) <= n:
        return available
    idx = np.linspace(0, len(available) - 1, n).round().astype(int)
    secs = sorted(set(available[i] for i in idx))
    if anchor_sec not in secs:
        secs = sorted(set(secs) | {anchor_sec})
    return secs


def run_subject(subject_id, out_dir, anchor_section=None, n_qc_sections=6, max_workers=14):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolve_subject(subject_id)
    print(f"\n########## subject {subject_id} ##########", flush=True)
    print(f"aligned dir: {cfg.aligned_dir}", flush=True)
    print(f"z-stack (registered): {cfg.zstack_registered_tif}", flush=True)
    print(f"z-stack (segmented): {cfg.zstack_segmented_tif}", flush=True)
    print(f"reporter zarr: {cfg.reporter_zarr_root or '(none -- using all-cells fallback)'}", flush=True)
    print(f"manual GT: {cfg.manual_gt_dir or '(none)'}", flush=True)
    print(f"{len(cfg.sections)} sections available: {cfg.sections}", flush=True)

    if anchor_section is None:
        anchor_section, section_counts = select_anchor_section(cfg)
    else:
        section_counts = None

    print(f"\n--- stage 1: initial landmark search (anchor section {anchor_section}) ---", flush=True)
    anchor_result = search_anchor_section(cfg, anchor_section, max_workers=max_workers)
    if anchor_result is None:
        raise RuntimeError(f"subject {subject_id}: initial search failed to converge at section "
                           f"{anchor_section} -- try a different anchor_section")

    init_dir = out_dir / "initial_match"
    init_dir.mkdir(exist_ok=True)
    np.save(init_dir / f"section_{anchor_section}_affine.npy", anchor_result["M_aligned"])
    json.dump(dict(sec=anchor_section, plane=anchor_result["plane"], n_cert=anchor_result["n_cert"],
                   match_rate=anchor_result["match_rate"], n_cz_slab=anchor_result["n_cz_slab"],
                   n_xen_fov=anchor_result["n_xen_fov"], scale=anchor_result["scale"],
                   anisotropy=anchor_result["anisotropy"], shear=anchor_result["shear"],
                   rotation=anchor_result["rotation"], sane=anchor_result["sane"],
                   used_reporter=anchor_result["used_reporter"]),
              open(init_dir / f"section_{anchor_section}_result.json", "w"), indent=2)
    np.save(init_dir / f"section_{anchor_section}_moving.npy", anchor_result["moving"])
    np.save(init_dir / f"section_{anchor_section}_fixed_aligned.npy", anchor_result["fixed_aligned"])
    np.save(init_dir / f"section_{anchor_section}_rotation_3d.npy", anchor_result["R_3d"])

    print(f"\n--- stage 2: chain-refined propagation across all sections ---", flush=True)
    chain_dir = out_dir / "chained_refined"
    chain_results = run_chain(cfg, anchor_section, anchor_result["M_aligned"], anchor_result["plane"], chain_dir,
                              R_3d=anchor_result["R_3d"])
    covered = [r["sec"] for r in chain_results]

    print(f"\n--- stage 3: fine registration + cell matching (QC subset) ---", flush=True)
    qc_sections = _pick_qc_sections(covered, anchor_section, n=n_qc_sections)
    fine_dir = out_dir / "fine_registered"
    fine_results = fine_register_sections(cfg, qc_sections, chain_dir, fine_dir)

    print(f"\n--- stage 4: QC figures ---", flush=True)
    qc_mod.qc_initial_match(cfg, anchor_result, out_dir / "QC1_initial_match.png")
    qc_mod.qc_propagation(cfg, chain_dir, qc_sections, out_dir / "QC2_propagation.png")
    qc_mod.qc_fine_registration(fine_dir, qc_sections, out_dir / "QC3_fine_registration.png")

    summary = dict(subject_id=subject_id, anchor_section=anchor_section,
                   anchor_result={k: v for k, v in anchor_result.items()
                                 if k not in ("moving", "fixed_aligned", "M_aligned", "R_3d")},
                   sections_covered=covered, qc_sections=qc_sections,
                   fine_registration=fine_results,
                   used_reporter=anchor_result["used_reporter"],
                   manual_gt_available=cfg.manual_gt_dir is not None)
    json.dump(summary, open(out_dir / "pipeline_summary.json", "w"), indent=2)
    print(f"\n[{subject_id}] DONE. wrote {out_dir}/pipeline_summary.json", flush=True)
    return summary
