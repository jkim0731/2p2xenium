"""Per-subject path resolution for the automatic z-stack <-> Xenium coregistration pipeline.

Generalizes the subject-specific hardcoded paths used during development (816462) to any subject
that has: (1) a `Xenium-ophys-coregistered_{subject}_*` aligned-frame Xenium directory (produced by
`interactive_section_alignment.py` / `prepare_data_and_align.py` -- provides *_aligned.tif and
confirmed_section_transforms.json), and (2) an `ophys-z-stacks_{subject}_segmented*` asset (hard
requirement -- provides both the segmentation and, via `zstack_data.tif`, the raw intensity volume
if no separate `_registered*` asset is mounted; the two are verified bit-identical on 827543, so
`_registered*` is a nice-to-have, not a second hard requirement).

Manual ground truth (`Xenium_ophys_{subject}_coregistered/`) and the raw processed Xenium zarr
(`Xenium_{subject}_*_processed/`, for reporter transcript counts) are OPTIONAL -- most subjects run
through this pipeline won't have either. When the zarr is missing, `populations.py` falls back to
using all segmented Xenium cells (see its docstring for why that's a reasonable substitute, not a
hack: reporter+ at the validated min_count=2 threshold already captures ~94% of all cells on 816462).
"""
import glob
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_ROOT = Path("/data")


def _latest_glob(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


@dataclass
class SubjectConfig:
    subject_id: int
    aligned_dir: Path                    # Xenium-ophys-coregistered_{sub}_*/  (images, masks, transforms)
    zstack_registered_tif: Path          # raw intensity volume, (Z,512,512), the 700x700-GCaMP stack
    zstack_segmented_tif: Path           # matching segmentation label volume
    reporter_zarr_root: Optional[Path]   # Xenium_{sub}_*_processed/  (None -> fall back to all-cells)
    manual_gt_dir: Optional[Path]        # Xenium_ophys_{sub}_coregistered/  (None -> no GT comparison)
    zstack_xy_um: float                  # per-subject z-stack um/px -- NOT always 700/512 (833855's
                                          # `multiplane-ophys` asset is a genuinely smaller 512x512um
                                          # FOV at the same 512x512px, i.e. 1.0um/px, not 1.367um/px;
                                          # verified via roi_groups_metadata.json's sizeXY*objectiveResolution)

    @property
    def sections(self):
        """All section numbers present in the aligned-frame directory."""
        paths = (self.aligned_dir / "Xenium_images").glob("section_*_aligned.tif")
        secs = set()
        for p in paths:
            stem = p.stem  # e.g. section_10_aligned or section_10_Neurons_aligned
            parts = stem.split("_")
            if parts[0] == "section":
                secs.add(int(parts[1]))
        return sorted(secs)


def _700x700_stacks(root):
    """All 700x700 acquisitions, GCaMP-only AND GCaMP+Dextran -- channel_0_ref_0 is the GCaMP
    channel in either case (verified for 816462: corr 0.47 vs 0.07 against a known GCaMP-only
    reference at the same plane, and visually round nuclei vs. vasculature), so a Dextran-paired
    acquisition is not excluded, just compared on equal footing via its own channel_0."""
    stacks = [p for p in glob.glob(f"{root}/*") if Path(p).is_dir()]
    return sorted(s for s in stacks if "700x700" in s)


def _n_rois(seg_tif_path):
    import tifffile as tiff
    return int(np.unique(tiff.imread(seg_tif_path)).size) - 1   # exclude background (0)


def _zstack_xy_um_from_roi_metadata(roi_metadata_path):
    """um/px from a ScanImage roi_groups_metadata.json: FOV_um = sizeXY[0] * objectiveResolution,
    so um/px = FOV_um / pixelResolutionXY[0]. Verified against the known 700um/400um stacks (both
    give objectiveResolution=157, and sizeXY*157 reproduces 700.0/400.0 exactly)."""
    import json
    d = json.load(open(roi_metadata_path))
    imaging_group = d["RoiGroups"]["imagingRoiGroup"] if "RoiGroups" in d else d["imagingRoiGroup"]
    roi = imaging_group["rois"]
    roi = roi[0] if isinstance(roi, list) else roi
    sf = roi["scanfields"]
    sf = sf[0] if isinstance(sf, list) else sf
    size_x = sf["sizeXY"][0]
    px_x = sf["pixelResolutionXY"][0]
    obj_res = 157.0  # SI.objectiveResolution -- constant across every acquisition checked so far
    return size_x * obj_res / px_x


def _find_zstack_pair_multiplane(subject_id: int):
    """Fallback for the newer `multiplane-ophys_{subject}_..._cortical-zstack-*` asset naming
    (e.g. 833855) -- a different convention from `ophys-z-stacks_{subject}_segmented*`, AND
    sometimes a genuinely different physical FOV (833855 originally only had a 512x512um asset,
    not the standard 700x700um -- since resolved by a proper 700x700 acquisition, but this still
    prefers "700x700" in the path when more than one multiplane asset is mounted, matching the
    `_700x700_stacks` convention used for the older naming, rather than relying on sort order),
    so the um/px scale must always be read from this stack's own metadata, never assumed to be
    the global 700/512 default."""
    seg_dirs = sorted(glob.glob(str(DATA_ROOT / f"multiplane-ophys_{subject_id}_*_cortical-zstack-segmentation_*")))
    seg_dirs_700 = [d for d in seg_dirs if "700x700" in d]
    seg_dirs = seg_dirs_700 if seg_dirs_700 else seg_dirs
    seg_tif = None
    for d in seg_dirs:
        t = _latest_glob(f"{d}/channel_0_ref_0/segmentation_masks.tif")
        if t:
            seg_tif = t
    if seg_tif is None:
        raise FileNotFoundError(f"no multiplane-ophys_{subject_id}_*_cortical-zstack-segmentation_* "
                                f"segmentation_masks.tif found under {DATA_ROOT}")

    reg_dirs = sorted(glob.glob(str(DATA_ROOT / f"multiplane-ophys_{subject_id}_*_cortical-zstack-registration_*")))
    reg_dirs_700 = [d for d in reg_dirs if "700x700" in d]
    reg_dirs = reg_dirs_700 if reg_dirs_700 else reg_dirs
    reg_tif = None
    roi_meta = None
    for d in reg_dirs:
        t = _latest_glob(f"{d}/cortical_zstack_0/channel_0_ref_0/*_2xREG.tif")
        if t:
            reg_tif = t
            roi_meta = _latest_glob(f"{d}/cortical_zstack_0/roi_groups_metadata.json")
    if reg_tif is None:
        raise FileNotFoundError(f"no multiplane-ophys_{subject_id}_*_cortical-zstack-registration_* "
                                f"*_2xREG.tif found under {DATA_ROOT}")
    zstack_xy_um = _zstack_xy_um_from_roi_metadata(roi_meta) if roi_meta else 700.0 / 512.0
    return Path(reg_tif), Path(seg_tif), zstack_xy_um


def _find_zstack_pair(subject_id: int):
    """Pick the 700x700 acquisition (GCaMP-only OR GCaMP+Dextran -- channel_0_ref_0 is GCaMP in
    both) with the MOST segmented ROIs in its channel_0 segmentation, when more than one
    acquisition exists (not just the alphabetically/date-first one, and not excluding Dextran-
    paired acquisitions -- their channel_0 is equally valid GCaMP data). Raw intensity: prefer a
    separate `_registered` asset matching the same acquisition-name stem; if none is mounted, fall
    back to `zstack_data.tif` co-located with the segmentation -- verified bit-identical to the
    separately-mounted registered `_2xREG.tif` on 827543 (both are the same upstream registered
    volume, just packaged differently), so this is not a lesser substitute.

    Falls back to `_find_zstack_pair_multiplane` for the newer `multiplane-ophys_*` naming
    convention (e.g. 833855) if no `ophys-z-stacks_*` asset is mounted."""
    seg_dirs = glob.glob(str(DATA_ROOT / f"ophys-z-stacks_{subject_id}_segmented*"))
    if not seg_dirs:
        return _find_zstack_pair_multiplane(subject_id)

    seg_stacks = []
    for d in seg_dirs:
        seg_stacks += _700x700_stacks(d)
    if not seg_stacks:
        raise FileNotFoundError(f"no 700x700 segmented stack for subject {subject_id}")

    seg_tifs = {}
    for stack in seg_stacks:
        t = _latest_glob(f"{stack}/channel_0_ref_0/segmentation_masks.tif")
        if t:
            seg_tifs[stack] = t
    if not seg_tifs:
        raise FileNotFoundError(f"no segmentation_masks.tif (channel_0_ref_0) under any 700x700 "
                                f"stack for subject {subject_id}")

    best_stack = max(seg_tifs, key=lambda s: _n_rois(seg_tifs[s]))
    seg_tif = seg_tifs[best_stack]

    stem_name = Path(best_stack).name.split("_segmented")[0]     # e.g. ophys-z-stack-700x700x450-GCaMP_2025-12-12_15-13
    reg_dirs = glob.glob(str(DATA_ROOT / f"ophys-z-stacks_{subject_id}_registered*"))
    reg_tif = None
    for d in reg_dirs:
        reg_tif = _latest_glob(f"{d}/{stem_name}_registered_*/channel_0_ref_0/*_2xREG.tif")
        if reg_tif:
            break
    if reg_tif is None:
        reg_tif = _latest_glob(f"{best_stack}/channel_0_ref_0/zstack_data.tif")
    if reg_tif is None:
        raise FileNotFoundError(f"no raw intensity source (registered *_2xREG.tif or co-located "
                                f"zstack_data.tif) for {best_stack}")
    return Path(reg_tif), Path(seg_tif), 700.0 / 512.0


def resolve_subject(subject_id: int) -> SubjectConfig:
    aligned = _latest_glob(str(DATA_ROOT / f"Xenium-ophys-coregistered_{subject_id}_*"))
    if aligned is None:
        raise FileNotFoundError(f"no Xenium-ophys-coregistered_{subject_id}_* aligned-frame directory "
                                f"under {DATA_ROOT} -- this is a hard requirement")
    zstack_reg, zstack_seg, zstack_xy_um = _find_zstack_pair(subject_id)

    reporter_root = _latest_glob(str(DATA_ROOT / f"Xenium_{subject_id}_*_processed"))
    manual_gt = _latest_glob(str(DATA_ROOT / f"Xenium_ophys_{subject_id}_coregistered"))

    return SubjectConfig(
        subject_id=subject_id,
        aligned_dir=Path(aligned),
        zstack_registered_tif=zstack_reg,
        zstack_segmented_tif=zstack_seg,
        reporter_zarr_root=Path(reporter_root) if reporter_root else None,
        manual_gt_dir=Path(manual_gt) if manual_gt else None,
        zstack_xy_um=zstack_xy_um,
    )


if __name__ == "__main__":
    import sys
    cfg = resolve_subject(int(sys.argv[1]))
    print(cfg)
    print(f"{len(cfg.sections)} sections: {cfg.sections}")
