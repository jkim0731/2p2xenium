"""Copies the raw z-stack assets actually used for this subject's coregistration into the
reference-format `ophys-z-stacks/` and `ophys-z-stacks_segmentation_masks/` directories -- a plain
file copy (no normalization/channel-swap like the reference's own `load_assets_prepare_data.ipynb`
data-prep step), so dtype/values are byte-identical to source. `ophys-z-stacks/` gets both the
GCaMP channel (always -- this is the channel actually used for matching) and the Dextran channel
if the SAME acquisition directory has one; `ophys-z-stacks_segmentation_masks/` only gets GCaMP
(masks + a computed outline, since no separate outline asset exists for the mounted segmentation)."""
import shutil
import numpy as np
import tifffile as tiff
from pathlib import Path
from skimage.segmentation import find_boundaries


def _find_dextran_companion(gcamp_reg_tif: Path):
    """The GCaMP 2xREG.tif always lives at .../channel_0_ref_0/<name>_2xREG.tif -- check whether
    the SAME acquisition also has a channel_1_ref_1 (Dextran) sibling."""
    channel0_dir = gcamp_reg_tif.parent
    if channel0_dir.name != "channel_0_ref_0":
        return None
    channel1_dir = channel0_dir.parent / "channel_1_ref_1"
    if not channel1_dir.is_dir():
        return None
    candidates = list(channel1_dir.glob("*_2xREG.tif"))
    return candidates[0] if candidates else None


def copy_zstack_assets(cfg, out_dir):
    out_dir = Path(out_dir)
    zstack_dir = out_dir / "ophys-z-stacks"
    zstack_seg_dir = out_dir / "ophys-z-stacks_segmentation_masks"
    zstack_dir.mkdir(parents=True, exist_ok=True)
    zstack_seg_dir.mkdir(parents=True, exist_ok=True)

    gcamp_reg = Path(cfg.zstack_registered_tif)
    dest_gcamp = zstack_dir / f"{gcamp_reg.stem}.tif"
    shutil.copy2(gcamp_reg, dest_gcamp)
    with tiff.TiffFile(gcamp_reg) as tf:
        print(f"[copy_zstack_assets] GCaMP intensity: {gcamp_reg.name} "
              f"dtype={tf.pages[0].dtype} shape=({len(tf.pages)},{tf.pages[0].shape})", flush=True)

    dextran_reg = _find_dextran_companion(gcamp_reg)
    if dextran_reg is not None:
        dest_dextran = zstack_dir / f"{dextran_reg.stem}.tif"
        shutil.copy2(dextran_reg, dest_dextran)
        with tiff.TiffFile(dextran_reg) as tf:
            print(f"[copy_zstack_assets] Dextran intensity: {dextran_reg.name} "
                  f"dtype={tf.pages[0].dtype} shape=({len(tf.pages)},{tf.pages[0].shape})", flush=True)
    else:
        print("[copy_zstack_assets] no Dextran channel alongside the GCaMP acquisition used -- "
              "skipping (none available, not silently substituting)", flush=True)

    gcamp_seg = Path(cfg.zstack_segmented_tif)
    dest_masks = zstack_seg_dir / f"{gcamp_seg.parent.parent.name}_masks.tif"
    shutil.copy2(gcamp_seg, dest_masks)
    with tiff.TiffFile(gcamp_seg) as tf:
        print(f"[copy_zstack_assets] GCaMP segmentation masks: dtype={tf.pages[0].dtype} "
              f"shape=({len(tf.pages)},{tf.pages[0].shape})", flush=True)

    # no separate outline asset is mounted for the segmentation -- compute one (per-plane inner
    # boundary of the label volume, matching the "outline" semantics used elsewhere in this
    # pipeline for the Xenium side) rather than skip it.
    masks_vol = tiff.imread(gcamp_seg)
    outline_vol = np.zeros_like(masks_vol, dtype=np.uint8)
    for z in range(masks_vol.shape[0]):
        outline_vol[z] = find_boundaries(masks_vol[z], mode="inner").astype(np.uint8)
    dest_outline = zstack_seg_dir / f"{gcamp_seg.parent.parent.name}_masks_outline.tif"
    tiff.imwrite(dest_outline, outline_vol)
    print(f"[copy_zstack_assets] GCaMP segmentation outline (computed, no separate asset existed): "
          f"dtype={outline_vol.dtype} shape={outline_vol.shape}", flush=True)

    return dict(gcamp_intensity=str(dest_gcamp),
               dextran_intensity=str(dest_dextran) if dextran_reg is not None else None,
               gcamp_masks=str(dest_masks), gcamp_outline=str(dest_outline))
