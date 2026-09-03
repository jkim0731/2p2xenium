"""Per-subject configuration for the coregistration pipeline.

`SubjectConfig` is a plain dataclass -- construct it directly with your own paths and acquisition
parameters for any data layout. `resolve_subject` is an OPTIONAL convenience resolver for one
specific lab's asset-naming convention (see its own docstring); it is not part of the required
interface and can be skipped entirely if you build `SubjectConfig` yourself.

A subject needs, at minimum: an "aligned frame" Xenium directory (each section already
section-to-section aligned, providing `Xenium_images/section_N_Neurons_aligned.tif` and
`Xenium_segmentation_masks/section_N_Masks_aligned.tif`) and a registered + segmented z-stack
volume pair (same shape, same physical field of view). A reporter-transcript population source is
optional; without one the pipeline falls back to using every segmented Xenium cell.
"""
import glob
import json
import numpy as np
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import DEFAULT_XENIUM_PX_UM, DEFAULT_Z_STEP_UM, DEFAULT_ZSTACK_SCALE_TO_XENIUM

DATA_ROOT = Path(os.environ.get("XENIUM_AUTOCOREG_DATA_ROOT", "/data"))


def _latest_glob(pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


@dataclass
class SubjectConfig:
    subject_id: int
    aligned_dir: Path                    # aligned-frame Xenium directory (images, masks, transforms)
    zstack_registered_tif: Path          # raw intensity volume, (Z, H, W)
    zstack_segmented_tif: Path           # matching segmentation label volume, same shape
    zstack_xy_um: float                  # this z-stack's lateral pixel size (um/px) -- always read
                                          # from the acquisition's own metadata; never assume a
                                          # fixed value, different acquisitions can differ
    reporter_zarr_root: Optional[Path] = None    # optional reporter-transcript population source;
                                                  # None -> fall back to all segmented cells
    z_step_um: float = DEFAULT_Z_STEP_UM             # z-stack axial resolution (um/plane)
    zstack_scale_to_Xenium: float = DEFAULT_ZSTACK_SCALE_TO_XENIUM  # multiply a z-stack point's
                                                                    # um coords by this to land in
                                                                    # the Xenium-aligned frame (see README)
    xenium_xy_um: float = DEFAULT_XENIUM_PX_UM       # Xenium morphology-image pixel size (um/px)
    section_spacing_um: float = 15.0                 # nominal physical spacing between consecutive
                                                      # Xenium sections -- the pipeline estimates the
                                                      # real per-section plane step empirically as it
                                                      # propagates, so this default is only a coarse
                                                      # prior, not load-bearing; override with your
                                                      # own acquisition's real value when known
    _zstack_shape_px: Optional[tuple] = field(default=None, repr=False, compare=False)

    @property
    def zstack_shape_px(self):
        """(Z, H, W) of the z-stack volume, read once from the segmentation file's header (no
        pixel data loaded) rather than assumed."""
        if self._zstack_shape_px is None:
            import tifffile as tiff
            with tiff.TiffFile(self.zstack_segmented_tif) as tf:
                self._zstack_shape_px = (len(tf.pages),) + tf.pages[0].shape
        return self._zstack_shape_px

    @property
    def zstack_fov_um(self):
        """Physical (H, W) field of view of the z-stack, in microns."""
        _, h, w = self.zstack_shape_px
        return (h * self.zstack_xy_um, w * self.zstack_xy_um)

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


# ---------------------------------------------------------------------------------------------
# Below: an OPTIONAL resolver for one lab's specific Code-Ocean-style data-asset naming
# convention. If your data isn't laid out this way, ignore this section entirely and construct
# `SubjectConfig` yourself.
# ---------------------------------------------------------------------------------------------

def _candidate_stacks(root, fov_tag=None):
    """Subdirectories of `root`, optionally filtered to ones whose name contains `fov_tag`."""
    stacks = [p for p in glob.glob(f"{root}/*") if Path(p).is_dir()]
    return sorted(s for s in stacks if fov_tag is None or fov_tag in s)


def _n_rois(seg_tif_path):
    import tifffile as tiff
    return int(np.unique(tiff.imread(seg_tif_path)).size) - 1   # exclude background (0)


def _zstack_xy_um_from_roi_metadata(roi_metadata_path, objective_resolution=157.0):
    """um/px from a ScanImage roi_groups_metadata.json: FOV_um = sizeXY[0] * objectiveResolution,
    so um/px = FOV_um / pixelResolutionXY[0]."""
    d = json.load(open(roi_metadata_path))
    imaging_group = d["RoiGroups"]["imagingRoiGroup"] if "RoiGroups" in d else d["imagingRoiGroup"]
    roi = imaging_group["rois"]
    roi = roi[0] if isinstance(roi, list) else roi
    sf = roi["scanfields"]
    sf = sf[0] if isinstance(sf, list) else sf
    size_x = sf["sizeXY"][0]
    px_x = sf["pixelResolutionXY"][0]
    return size_x * objective_resolution / px_x


def _find_zstack_pair_multiplane(subject_id: int, fov_tag="700x700"):
    """Resolver for a `multiplane-ophys_{subject}_*_cortical-zstack-{segmentation,registration}_*`
    asset layout, reading pixel size from the acquisition's own metadata rather than assuming one."""
    seg_dirs = glob.glob(str(DATA_ROOT / f"multiplane-ophys_{subject_id}_*_cortical-zstack-segmentation_*"))
    seg_dirs_tagged = [d for d in seg_dirs if fov_tag in d]
    seg_dirs = seg_dirs_tagged if seg_dirs_tagged else seg_dirs
    seg_tif = None
    for d in sorted(seg_dirs):
        t = _latest_glob(f"{d}/channel_0_ref_0/segmentation_masks.tif")
        if t:
            seg_tif = t
    if seg_tif is None:
        raise FileNotFoundError(f"no multiplane-ophys_{subject_id}_*_cortical-zstack-segmentation_* "
                                f"segmentation_masks.tif found under {DATA_ROOT}")

    reg_dirs = glob.glob(str(DATA_ROOT / f"multiplane-ophys_{subject_id}_*_cortical-zstack-registration_*"))
    reg_dirs_tagged = [d for d in reg_dirs if fov_tag in d]
    reg_dirs = reg_dirs_tagged if reg_dirs_tagged else reg_dirs
    reg_tif, roi_meta = None, None
    for d in sorted(reg_dirs):
        t = _latest_glob(f"{d}/cortical_zstack_0/channel_0_ref_0/*_2xREG.tif")
        if t:
            reg_tif = t
            roi_meta = _latest_glob(f"{d}/cortical_zstack_0/roi_groups_metadata.json")
    if reg_tif is None:
        raise FileNotFoundError(f"no multiplane-ophys_{subject_id}_*_cortical-zstack-registration_* "
                                f"*_2xREG.tif found under {DATA_ROOT}")
    if roi_meta is None:
        raise FileNotFoundError(f"no roi_groups_metadata.json alongside {reg_tif} -- cannot "
                                f"determine this acquisition's um/px without it")
    return Path(reg_tif), Path(seg_tif), _zstack_xy_um_from_roi_metadata(roi_meta)


def _find_zstack_pair(subject_id: int, fov_tag="700x700"):
    """Picks the acquisition (matching `fov_tag` when more than one is mounted) with the most
    segmented ROIs, preferring a separate registered-intensity asset and falling back to a
    co-located raw-data file if none is mounted. Falls back to `_find_zstack_pair_multiplane` for
    a different asset-naming convention if the primary one isn't found."""
    seg_dirs = glob.glob(str(DATA_ROOT / f"ophys-z-stacks_{subject_id}_segmented*"))
    if not seg_dirs:
        return _find_zstack_pair_multiplane(subject_id, fov_tag)

    seg_stacks = []
    for d in seg_dirs:
        seg_stacks += _candidate_stacks(d, fov_tag)
    if not seg_stacks:
        raise FileNotFoundError(f"no {fov_tag} segmented stack for subject {subject_id}")

    seg_tifs = {}
    for stack in seg_stacks:
        t = _latest_glob(f"{stack}/channel_0_ref_0/segmentation_masks.tif")
        if t:
            seg_tifs[stack] = t
    if not seg_tifs:
        raise FileNotFoundError(f"no segmentation_masks.tif (channel_0_ref_0) under any {fov_tag} "
                                f"stack for subject {subject_id}")

    best_stack = max(seg_tifs, key=lambda s: _n_rois(seg_tifs[s]))
    seg_tif = seg_tifs[best_stack]

    stem_name = Path(best_stack).name.split("_segmented")[0]
    reg_dir = None
    reg_dirs = glob.glob(str(DATA_ROOT / f"ophys-z-stacks_{subject_id}_registered*"))
    reg_tif = None
    for d in reg_dirs:
        found_dir = _latest_glob(f"{d}/{stem_name}_registered_*")
        t = _latest_glob(f"{found_dir}/channel_0_ref_0/*_2xREG.tif") if found_dir else None
        if t:
            reg_tif, reg_dir = t, found_dir
            break
    if reg_tif is None:
        reg_tif = _latest_glob(f"{best_stack}/channel_0_ref_0/zstack_data.tif")
    if reg_tif is None:
        raise FileNotFoundError(f"no raw intensity source (registered *_2xREG.tif or co-located "
                                f"zstack_data.tif) for {best_stack}")

    # roi_groups_metadata.json's location varies by acquisition pipeline version -- check every
    # plausible spot (registered-dir root, registered-dir/channel_0_ref_0, segmented-dir/channel_0_ref_0)
    # before giving up.
    roi_meta = None
    for candidate in (
        f"{reg_dir}/roi_groups_metadata.json" if reg_dir else None,
        f"{reg_dir}/channel_0_ref_0/roi_groups_metadata.json" if reg_dir else None,
        f"{best_stack}/channel_0_ref_0/roi_groups_metadata.json",
        f"{best_stack}/roi_groups_metadata.json",
    ):
        if candidate and _latest_glob(candidate):
            roi_meta = _latest_glob(candidate)
            break
    if roi_meta is not None:
        zstack_xy_um = _zstack_xy_um_from_roi_metadata(roi_meta)
    else:
        zstack_xy_um = None  # caller decides whether/how to fall back -- see resolve_subject
    return Path(reg_tif), Path(seg_tif), zstack_xy_um


def resolve_subject(subject_id: int, fov_tag: str = "700x700",
                    fallback_fov_um: Optional[float] = 700.0,
                    fallback_native_px: Optional[int] = 512) -> SubjectConfig:
    """Convenience resolver for one lab's Code-Ocean-style mounted-asset naming convention. For
    any other data layout, construct `SubjectConfig` directly instead.

    `fallback_fov_um`/`fallback_native_px`: used ONLY when no `roi_groups_metadata.json` can be
    found for this acquisition (some older assets don't have one mounted) -- an explicit, visible,
    overridable nominal value rather than a silent assumption. Pass `fallback_fov_um=None` to
    require real metadata and raise instead."""
    aligned = _latest_glob(str(DATA_ROOT / f"Xenium-ophys-coregistered_{subject_id}_*"))
    if aligned is None:
        raise FileNotFoundError(f"no Xenium-ophys-coregistered_{subject_id}_* aligned-frame directory "
                                f"under {DATA_ROOT}")
    zstack_reg, zstack_seg, zstack_xy_um = _find_zstack_pair(subject_id, fov_tag)
    if zstack_xy_um is None:
        if fallback_fov_um is None:
            raise FileNotFoundError(f"no roi_groups_metadata.json found for subject {subject_id}'s "
                                    f"z-stack, and no fallback_fov_um given -- cannot determine um/px")
        print(f"[{subject_id}] WARNING: no roi_groups_metadata.json found -- assuming a nominal "
              f"{fallback_fov_um}um FOV over {fallback_native_px}px (pass fallback_fov_um= to "
              f"resolve_subject to change this).", flush=True)
        zstack_xy_um = fallback_fov_um / fallback_native_px
    reporter_root = _latest_glob(str(DATA_ROOT / f"Xenium_{subject_id}_*_processed"))

    return SubjectConfig(
        subject_id=subject_id,
        aligned_dir=Path(aligned),
        zstack_registered_tif=zstack_reg,
        zstack_segmented_tif=zstack_seg,
        zstack_xy_um=zstack_xy_um,
        reporter_zarr_root=Path(reporter_root) if reporter_root else None,
    )


if __name__ == "__main__":
    import sys
    cfg = resolve_subject(int(sys.argv[1]))
    print(cfg)
    print(f"{len(cfg.sections)} sections: {cfg.sections}")
