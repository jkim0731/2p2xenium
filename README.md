# xenium-autocoreg (2p2xenium)

GT-free coregistration of an **in-vivo 2-photon cortical z-stack** to **Xenium spatial
transcriptomics** sections of the same mouse cortex, so the same physical cell can be found in
both modalities.

- Initial protocol development by Omid Zobeiri @ Allen Institute: Xenium slice alignment, tilt fitting,
chain-refine propagation, fine registration, and 3D point mapping
- This repo added automatic initial landmark searching, enabling the full co-registration process automatic.

## Install

```bash
pip install -e .
# with tests:
pip install -e ".[test]"
```

## The process

1. **Pose seeding** (find a rough starting pose -- center, rotation, scale -- for one anchor
   Xenium section): 3 modes, see [`src/xenium_autocoreg/pose_seed.py`](src/xenium_autocoreg/pose_seed.py).
   - `auto` -- fully automatic blind rotation x position x depth pose grid
     (`initial_match.search_anchor_section`). No human input, but can fail outright if the true
     pose sits outside the searched position window (this happened for one real subject; see
     "Known limitations").
   - `center-rotation` -- a human provides a rough center (Xenium-aligned frame, um) and rotation
     (deg), eyeballed from a confocal/vasculature image or however already available. Only a
     **depth sweep** runs automatically from there -- no position/rotation grid. Validated: this
     recovered the strongest match of any subject tested, on a case where `auto` failed
     completely.
   - `corners` -- **not yet implemented** (see the docstring on `pose_seed.seed_from_corners` for
     exactly why, and what real click data is needed before it can be).
2. **Tilt fitting** (`tilt_fit.fit_tilt_and_landmarks`): iteratively match landmarks in a 60um
   z-stack slab around the seeded pose, accumulate new ones (bijective -- no z-stack or Xenium
   cell reused across landmarks), fit a 3D tilt correction `R_3d` from the *full* accumulated set,
   de-tilt the z-stack population, re-match in the newly-revealed slab, repeat until a round finds
   no new landmarks. `R_3d` is then held **fixed** for the rest of the pipeline -- the z-stack
   volume is rotated once, not per section.
3. **Chain-refine propagation** (`chain_refine.run_chain`): starting from the anchor's converged
   pose, walk outward section-by-section (forward and backward), seeding each section from the
   *previous* section's own converged result via image-intensity tile correlation, composing an
   SVD-scale-clipped correction affine each round (`chain_refine.clip_affine_scale` -- prevents a
   weak-signal section drifting into an unphysical anisotropic-scale affine that would otherwise
   propagate to every section downstream). The seed step is scaled by the actual **gap in Xenium
   section numbers** between consecutive processed sections (some subjects are missing sections),
   using a running mean of observed plane-step-per-section.
5. **Fine registration** (`fine_registration.register_section`): fine mask-based tile correlation
   (tight window, binary cell masks, not raw intensity) -> thin-plate-spline control points -> warp
   the z-stack into the Xenium-affine-transformed frame -> the probability-filtered cell-matching
   metric (kNN spatial-shift null model -> Mahalanobis distance -> empirical p-value;
   `valid = iou>0.2 & p<0.05`).
6. **3D point mapping** (`transform_xenium_points.run_transform_xenium_points`): maps every Xenium
   cell centroid (not just matched ones) into 3D z-stack coordinates, both non-rigid (TPS) and
   rigid (affine + tilt).

## Required input data (per subject)

- **Xenium, ALIGNED frame** (sections already registered to each other -- do NOT use the raw,
  un-aligned files):
  - `Xenium_images/section_N_Neurons_aligned.tif`
  - `Xenium_segmentation_masks/section_N_{Masks,Masks_outline}_aligned.tif`
- **z-stack, registered + segmented**: a single-channel (or channel-0-of-multichannel) intensity
  volume and its matching label-mask segmentation, same shape, same physical FOV.
- Optional: a reporter-transcript Xenium zarr (used to restrict the population to reporter+ cells
  when available; falls back to all segmented cells otherwise).

`config.SubjectConfig`/`resolve_subject` currently resolve these from this project's own
CodeOcean-capsule data-asset conventions (`/data/...`) -- porting to a different data layout means
either mounting data the same way or writing your own `SubjectConfig` construction.

## Run

```bash
xenium-autocoreg 823049 /path/to/out --pose-mode auto
xenium-autocoreg 833855 /path/to/out --pose-mode center-rotation --center-um 1363.15,1535.68 --rotation-deg 9.61
xenium-autocoreg <id> /path/to/out --pose-mode center-rotation --pose-json seed.json
```
`seed.json` for `center-rotation`:
```json
{"center_um": [1363.149, 1535.675], "rotation_deg": 9.610, "scale": 0.80}
```

## Output structure

```
<out_dir>/
├── Affine matrices/
│   ├── section_N_affine_matrix.npy      # (4,3): 3x3 affine (Xenium row,col -> z-stack row,col)
│   │                                     #  stacked with a 4th row [0,0,z_base]
│   └── section_N_rotation_3d.npy        # (3,3) tilt rotation R_3d -- identical across every
│                                         #  section for a subject (fit once at the anchor, held fixed)
├── Xenium_affine_transformed/
│   ├── section_N_Neurons_transformed.tif        # Xenium intensity, affine-warped into z-stack canvas
│   ├── section_N_Masks_transformed.tif          # Xenium label masks, same warp
│   └── section_N_Masks_outline_transformed.tif  # Xenium mask outlines, same warp
├── warped_zstacks/
│   ├── section_N_zstack_warped_plane.tif        # z-stack intensity, TPS-warped into the shared frame
│   └── section_N_zstack_warped_masks_plane.tif  # z-stack label masks, same TPS warp
├── post_affine_warping/
│   └── section_N.csv                    # 9-col, no header: id,valid(corr>0.2),z/y/x_zstack,
│                                         #  z/y/x_xenium(+jitter),corr -- the fine tile-correlation points
├── cell_matching_probability/
│   ├── section_N_matching_results.csv   # mask_id_xenium,mask_id_cz,iou,iou_neghbors,
│   │                                     #  mahal_pvalues,mahal_qvalues,empirical_pvalues,section,valid
│   ├── section_N_iou_data.npz           # raw null-model arrays
│   └── mouse_{subject}_total_matching_results.csv  # all sections concatenated
├── cell_centroids/
│   ├── section_N_xenium_centroids.csv   # mask_id,centroid_y,centroid_x (Xenium aligned-frame px)
│   └── section_N_ophys_centroids.csv    # mask_id,centroid_y,centroid_x (z-stack warped-plane px)
├── mapped_3d_coordinates/
│   ├── section_N_3d_centroids.csv       # every Xenium cell -> 3D z-stack coords, non-rigid (_nr) + rigid (_r)
│   └── all_sections_3d_centroids.csv    # all sections concatenated
├── ophys-z-stacks/                       # raw intensity volume(s), copied as-is (self-contained results)
├── ophys-z-stacks_segmentation_masks/    # raw segmentation + a computed outline + regionprops cache
├── QC/
│   ├── xenium_affine_zstack/section_N_xenium_affine_zstack.png   # 2x3: z-stack/Xenium/overlap, intensity + masks
│   ├── cell_matching/section_N_cell_matching__.{png,svg}         # 3-panel: contours + valid-match overlay
│   └── initial_match/section_N_initial_match_wide.png            # anchor-only: landmark search vs. final registration
├── propagation_summary.json              # per-section [{"sec","z_base","n_tiles","n_tiles_ok","n_valid_matches"}, ...]
└── _chain_internal/                      # chain_refine's own working files (harmless scratch)
```

## QC figures

- **`QC/xenium_affine_zstack/section_N_xenium_affine_zstack.png`** -- one per section: a 2x3 panel
  (z-stack / Xenium / overlap, intensity on top, masks on bottom) showing the final registration.
- **`QC/cell_matching/section_N_cell_matching__.png`** -- one per section: 3 panels (Xenium cell
  contours, z-stack cell contours, overlay of only the statistically-valid matched pairs).
- **`QC/initial_match/section_N_initial_match_wide.png`** -- **anchor section only**: a 2x3 panel,
  top row = the initial landmark search (full-section context + a 1.1x-FOV zoomed
  point-correspondence view + filled-mask overlap, all BEFORE tilt/TPS), bottom row = the same
  region AFTER the fitted tilt + fine mask-TPS warp. Generated by
  [`qc/initial_match.py`](src/xenium_autocoreg/qc/initial_match.py)'s `plot_initial_match_qc`.

## Tests

```bash
pytest
```
All tests are synthetic/fast and need no mounted subject data -- they exercise the geometry
primitives, the SVD scale-clip, soma-print point matching, and cell-mask IoU logic directly. One
(`test_pose_seed.py`) is anchored to the real, validated numbers from the one subject where
`center-rotation` mode was actually used in production.


## Known limitations

- The SVD scale-clip in `chain_refine.clip_affine_scale` bounds scale but not shear -- a
  weak-correlation section can still show shear-driven cell-shape distortion.
- `auto` pose-seeding can fail outright (see `pose_seed.seed_from_auto_search`'s docstring) if the
  true pose sits outside the searched position window; use `center-rotation` when it does.
- `corners` pose-seeding is not implemented -- see `pose_seed.seed_from_corners`.
- Data resolution (`config.resolve_subject`) is tied to this project's own data-asset layout, not
  yet parameterized via environment variables the way a fully portable package would be.
