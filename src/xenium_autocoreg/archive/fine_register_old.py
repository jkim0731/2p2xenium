"""Per-section fine registration + cell matching -- the real production code
(coreg.tile_warp/tps/cell_matching, coreg.geometry.apply_affine_based_on_reference_2d), called
exactly as capsule-8111671-coregistration-codes/alignment_matching.ipynb does (see the
coreg-aligned-native-pipeline project memory for the exact call sequence and the frame gotcha:
the final IoU comparison happens on the z-stack's own (512,512) canvas, not Xenium's native grid)."""
import json
import numpy as np
import tifffile as tiff
from pathlib import Path
from coreg.geometry import apply_affine_based_on_reference_2d
from coreg.tile_warp import tile_based_warping
from coreg.tps import tps_warp_from_landmarks
from coreg.cell_matching import match_table


def fine_register_section(cfg, sec, M, plane, zstack_vol, zstack_seg, out_dir):
    xen_neurons = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_images" / f"section_{sec}_Neurons_aligned.tif"))
    xen_masks = np.squeeze(tiff.imread(cfg.aligned_dir / "Xenium_segmentation_masks" / f"section_{sec}_Masks_aligned.tif"))

    xen_neurons_t = apply_affine_based_on_reference_2d(xen_neurons.astype(np.float32), M, (512, 512), order=0)
    xen_masks_t = apply_affine_based_on_reference_2d(xen_masks, M, (512, 512), order=0)

    zyx_img, zyx_vol, best_corrs, base_corrs = tile_based_warping(
        xen_neurons_t, zstack_vol, z_base=plane, margin=5, tile_size=(64, 64),
        overlap=0.4, max_shift=(20, 20, 20), progress=False)
    n_good = int((np.array(best_corrs) > 0.2).sum())
    zstack_warped_masks = tps_warp_from_landmarks(zstack_seg, zyx_img, zyx_vol, best_corrs, corr_threshold=0.2)

    df = match_table(xen_masks_t, zstack_warped_masks, threshold=0.2)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"section_{sec}_cell_matching.csv", index=False)
    tiff.imwrite(out_dir / f"section_{sec}_xenium_masks_transformed.tif", xen_masks_t.astype(np.uint32))
    tiff.imwrite(out_dir / f"section_{sec}_zstack_warped_masks.tif", zstack_warped_masks.astype(np.uint32))

    rec = dict(sec=sec, plane=plane, n_tiles=len(best_corrs), n_tiles_good=n_good, n_cell_matches=int(len(df)))
    print(f"[sec{sec}] fine-register: {n_good}/{len(best_corrs)} tiles ok, {len(df)} cell matches", flush=True)
    return rec


def fine_register_sections(cfg, sections, chain_dir, out_dir):
    print("loading z-stack intensity + segmentation volumes...", flush=True)
    zstack_vol = tiff.imread(cfg.zstack_registered_tif)
    zstack_seg = tiff.imread(cfg.zstack_segmented_tif)
    chain_dir = Path(chain_dir); out_dir = Path(out_dir)

    results = []
    for sec in sections:
        M = np.load(chain_dir / f"section_{sec}_affine.npy")
        plane = json.load(open(chain_dir / f"section_{sec}_result.json"))["z_base"]
        results.append(fine_register_section(cfg, sec, M, plane, zstack_vol, zstack_seg, out_dir))

    json.dump(results, open(out_dir / "fine_registration_summary.json", "w"), indent=2)
    return results
