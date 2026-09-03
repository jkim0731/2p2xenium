"""Per-subject configuration for the coregistration pipeline.

`SubjectConfig` is a plain dataclass -- construct it directly with your own paths and acquisition
parameters for any data layout, or load one from a JSON file with `subject_config_from_json`. This
module is intentionally data-layout-agnostic: it does not know about any particular lab's
data-asset naming convention. If you need to resolve `SubjectConfig`'s fields from a specific
mounted-asset layout (e.g. a CodeOcean capsule with a lab-specific asset-naming convention), write
that resolver in your own pipeline/capsule instead -- see
https://github.com/AllenNeuralDynamics/ophys-xenium-autocoreg for a reference implementation (a
CodeOcean capsule wrapping this package).

A subject needs, at minimum: an "aligned frame" Xenium directory (each section already
section-to-section aligned, providing `Xenium_images/section_N_Neurons_aligned.tif` and
`Xenium_segmentation_masks/section_N_Masks_aligned.tif`) and a registered + segmented z-stack
volume pair (same shape, same physical field of view). A reporter-transcript population source is
optional; without one the pipeline falls back to using every segmented Xenium cell.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import DEFAULT_XENIUM_PX_UM, DEFAULT_Z_STEP_UM, DEFAULT_ZSTACK_SCALE_TO_XENIUM


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


# Fields subject_config_from_json coerces to Path if present (matches the dataclass's own
# Path-typed fields above).
_PATH_FIELDS = ("aligned_dir", "zstack_registered_tif", "zstack_segmented_tif", "reporter_zarr_root")
_REQUIRED_FIELDS = ("subject_id", "aligned_dir", "zstack_registered_tif", "zstack_segmented_tif",
                    "zstack_xy_um")


def subject_config_from_json(path) -> SubjectConfig:
    """Build a `SubjectConfig` from a JSON file whose keys match the dataclass's own field names,
    e.g.:
        {
          "subject_id": 816462,
          "aligned_dir": "/data/aligned_816462",
          "zstack_registered_tif": "/data/zstack_816462_registered.tif",
          "zstack_segmented_tif": "/data/zstack_816462_segmented.tif",
          "zstack_xy_um": 1.367,
          "reporter_zarr_root": "/data/xenium_816462_processed"
        }
    Only subject_id/aligned_dir/zstack_registered_tif/zstack_segmented_tif/zstack_xy_um are
    required; any other `SubjectConfig` field present in the JSON overrides its dataclass default.

    This is a generic, layout-agnostic loader -- it does not glob or resolve anything from a
    mounted-asset naming convention; build the JSON yourself (or construct `SubjectConfig` directly
    in Python) from whatever data layout you have.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"{path}: missing required SubjectConfig field(s) {missing}")
    kwargs = dict(data)
    for f in _PATH_FIELDS:
        if kwargs.get(f) is not None:
            kwargs[f] = Path(kwargs[f])
    return SubjectConfig(**kwargs)


def zstack_xy_um_from_roi_metadata(roi_metadata_path, objective_resolution: float = 157.0) -> float:
    """um/px from a ScanImage `roi_groups_metadata.json`: FOV_um = sizeXY[0] * objectiveResolution,
    so um/px = FOV_um / pixelResolutionXY[0]. A general ScanImage-acquisition utility -- not tied to
    any particular data-asset layout."""
    d = json.loads(Path(roi_metadata_path).read_text())
    imaging_group = d["RoiGroups"]["imagingRoiGroup"] if "RoiGroups" in d else d["imagingRoiGroup"]
    roi = imaging_group["rois"]
    roi = roi[0] if isinstance(roi, list) else roi
    sf = roi["scanfields"]
    sf = sf[0] if isinstance(sf, list) else sf
    size_x = sf["sizeXY"][0]
    px_x = sf["pixelResolutionXY"][0]
    return size_x * objective_resolution / px_x
