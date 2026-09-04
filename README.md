# xenium-autocoreg (2p2xenium)

Automatic coregistration of an **in-vivo 2-photon structural z-stack** to **Xenium spatial transcriptomics** sections of the same mouse cortex, so the same physical cell can be found in
both modalities.

- Initial protocol development by Omid Zobeiri @ Allen Institute: Xenium slice alignment, tilt fitting,
chain-refine propagation, fine registration, and 3D point mapping
- This repo added automatic initial landmark searching using the Soma-print point-cloud descriptor
  ([Wang et al., 2026](https://www.biorxiv.org/content/10.64898/2026.04.28.719500v1)), enabling the
  full co-registration process to run automatically.

## Install

```bash
pip install -e .
# with tests:
pip install -e ".[test]"
```

## The process

1. **Anchor-section selection** (`anchor.select_anchor_section`): pick the one Xenium section to
   run the (expensive) initial pose search on. Counts cells per section (whichever population
   `populations.load_xenium_cells` returns -- reporter+ or all-cells) and picks the first section
   (in section-number order) that's both cell-rich (>= `ANCHOR_MIN_CELLS`, default 1000) and past
   a density plateau (>= `ANCHOR_PLATEAU_FRAC`, default 0.80, of the densest section's count) --
   falling back to the single densest section if none clears that bar. The idea: an early, sparse
   section (e.g. right at a tissue edge) is a weak foundation for the one search the rest of the
   chain propagates from, so skip past the density ramp-up first.
2. **Pose seeding** (find a rough starting pose -- center, rotation, scale -- for the anchor
   section): 3 modes, see [`src/xenium_autocoreg/pose_seed.py`](src/xenium_autocoreg/pose_seed.py).
   Both implemented modes certify candidate landmark correspondences with the Soma-print
   point-cloud descriptor ([Wang et al.,
   2026](https://www.biorxiv.org/content/10.64898/2026.04.28.719500v1); `somaprint.py`) -- a
   rotation-invariant per-cell neighbor-constellation signature that lets a candidate pose be
   scored by how many bijective, mutually-consistent cell pairs it certifies, not just raw overlap.
   - `auto` -- fully automatic blind rotation x position x depth pose grid
     (`initial_match.search_anchor_section`). No human input, but can fail outright if the true
     pose sits outside the searched position window (see "Known limitations").
   - `center-rotation` -- a human provides a rough center (Xenium-aligned frame, um) and rotation
     (deg), eyeballed from a confocal/vasculature image or however already available. Only a
     **depth sweep** runs automatically from there -- no position/rotation grid. Use this when
     `auto` fails.
   - `corners` -- **not yet implemented** (see the docstring on `pose_seed.seed_from_corners` for
     exactly why, and what real click data is needed before it can be).
3. **Tilt fitting** (`tilt_fit.fit_tilt_and_landmarks`): iteratively match landmarks in a z-stack
   slab around the seeded pose, accumulate new ones (bijective -- no z-stack or Xenium cell reused
   across landmarks), fit a 3D tilt correction `R_3d` from the *full* accumulated set, de-tilt the
   z-stack population, re-match in the newly-revealed slab, repeat until a round finds no new
   landmarks. `R_3d` is then held **fixed** for the rest of the pipeline -- the z-stack volume is
   rotated once, not per section.
4. **Chain-refine propagation** (`chain_refine.run_chain`): starting from the anchor's converged
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

  "Aligned" here means section-to-section: consecutive Xenium sections have already been
  registered to each other (so the same tissue landmark sits at the same pixel across sections N,
  N+1, N-1, ...) by whatever upstream process produced these files -- this pipeline does not do
  that alignment itself, it only consumes it. This is a different, prior step from what this
  pipeline itself produces (`Xenium_affine_transformed/`, which further warps the aligned frame
  into the z-stack's own frame). If your data isn't already section-to-section aligned, align it
  first; feeding in raw, un-aligned per-section files will silently produce a wrong pose (each
  section would need its own independent search, not the one shared anchor+chain this pipeline
  assumes).
- **z-stack, registered + segmented**: a single-channel intensity
  volume and its matching label-mask segmentation, same shape, same physical FOV.
- Optional: a reporter-transcript Xenium population source (used to restrict matching to
  reporter+ cells when available; falls back to all segmented cells otherwise).

## Configuration -- exposed acquisition parameters

Every physical/acquisition parameter lives on `SubjectConfig` (`config.py`), with a package-level
default you can override per subject -- nothing in the algorithm hardcodes a specific field of
view, pixel count, or resolution:

| field | meaning | default |
|---|---|---|
| `zstack_xy_um` | z-stack lateral pixel size (um/px) | *(required, no default -- read from acquisition metadata)* |
| `z_step_um` | z-stack axial resolution (um/plane) | `1.0` |
| `xenium_xy_um` | Xenium morphology-image pixel size (um/px) | `0.2125 * 3.9996` |
| `zstack_scale_to_Xenium` | multiply a z-stack point's um coordinates by this to land in the Xenium-aligned frame's um scale (accounts for tissue processing shrink/expansion between the two modalities) | `0.80` |
| `section_spacing_um` | nominal physical spacing between Xenium sections -- a coarse prior only; the pipeline estimates the real per-section plane step empirically as it propagates, so this is not load-bearing | `15.0` |
| `zstack_shape_px` | `(Z, H, W)` of the z-stack volume | read from the segmentation file's header |
| `zstack_fov_um` | physical `(H, W)` field of view | derived from `zstack_shape_px * zstack_xy_um` |

Construct `SubjectConfig` directly for your own data layout:
```python
from xenium_autocoreg.config import SubjectConfig
cfg = SubjectConfig(
    subject_id=..., aligned_dir=..., zstack_registered_tif=..., zstack_segmented_tif=...,
    zstack_xy_um=1.234,               # required -- read from your acquisition's own metadata
    zstack_scale_to_Xenium=0.82,      # override if your tissue prep differs from the default
)
```
or load one from a plain JSON file with the same field names via `config.subject_config_from_json`
(see the `xenium-autocoreg` CLI below, which takes exactly this JSON as its first argument).

This package does not implement any lab-specific data-asset resolver (e.g. globbing a CodeOcean
capsule's mounted-asset naming convention into a `SubjectConfig`) -- write that resolver in your
own pipeline/capsule instead. See
[`ophys-xenium-autocoreg`](https://github.com/AllenNeuralDynamics/ophys-xenium-autocoreg) for a
reference implementation (a CodeOcean capsule wrapping this package, with its own
`subject_resolver.py`).

## Run

```bash
xenium-autocoreg <config.json> /path/to/out --pose-mode auto
xenium-autocoreg <config.json> /path/to/out --pose-mode center-rotation --center-um 1200.0,1500.0 --rotation-deg 10.0
xenium-autocoreg <config.json> /path/to/out --pose-mode center-rotation --pose-json seed.json
```
`<config.json>` matches `SubjectConfig`'s own field names (see `config.subject_config_from_json`
above) -- e.g.:
```json
{"subject_id": 816462, "aligned_dir": "/data/aligned_816462",
 "zstack_registered_tif": "/data/zstack_816462_registered.tif",
 "zstack_segmented_tif": "/data/zstack_816462_segmented.tif", "zstack_xy_um": 1.367}
```
`seed.json` for `center-rotation`:
```json
{"center_um": [1200.0, 1500.0], "rotation_deg": 10.0, "scale": 0.80}
```
(`"scale"` is optional -- defaults to the subject's own `zstack_scale_to_Xenium` if omitted.)

`--num-cpus N` controls worker-process count for every parallelized stage (the auto pose-grid
search, the per-section fine-registration tile correlation, cell-centroid extraction, and 3D point
mapping) -- see `resources.resolve_num_cpus`: blank/`0`/`N` greater than this machine's CPU count
= auto (every available core); `N=1` = serial, no multiprocessing at all (useful for debugging, or
a resource-constrained environment where spawning many worker processes gets silently killed).

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
│   ├── cell_matching/section_N_cell_matching.png                 # 3-panel: contours + valid-match overlay
│   └── initial_match/section_N_initial_match_wide.png            # anchor-only: landmark search vs. final registration
├── propagation_summary.json              # per-section [{"sec","z_base","n_tiles","n_tiles_ok","n_valid_matches"}, ...]
└── _chain_internal/                      # chain_refine's own working files (harmless scratch)
```

## QC figures

- **`QC/xenium_affine_zstack/section_N_xenium_affine_zstack.png`** -- one per section: a 2x3 panel
  (z-stack / Xenium / overlap, intensity on top, masks on bottom) showing the final registration.
- **`QC/cell_matching/section_N_cell_matching.png`** -- one per section: 3 panels (Xenium cell
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
primitives, the SVD scale-clip, soma-print point matching, and cell-mask IoU logic directly.

### Real-data test fixture

A ~89MB real-data fixture (5 Xenium sections + a grid-aligned z-stack crop around one subject's
validated anchor) is available on this repo's [Releases](../../releases) page rather than
committed into the repository -- see the release notes / bundled `DATASET.md` for exactly where
it's from (source Code Ocean data asset IDs), how it was generated (anchor pose search, tilt
fitting, and the float16/label-remap downcasting, empirically validated to not change this
pipeline's matching output), and its `validated_test_cases`, where BOTH pose-seeding modes
(`auto` and `center-rotation`) are run directly against the released files and converge to the
identical result -- a real regression target, not just example data.

**To use it:**
```bash
# download and unzip the asset from this repo's Releases page, then:
cd test_data
python -c "
import json
from pathlib import Path
from xenium_autocoreg.config import SubjectConfig
from xenium_autocoreg import pose_seed as ps

meta = json.load(open('metadata.json'))
cfg = SubjectConfig(
    subject_id=meta['subject_id'],
    aligned_dir=Path('aligned'),
    zstack_registered_tif=Path('zstack/zstack_registered_cropped.tif'),
    zstack_segmented_tif=Path('zstack/zstack_segmented_cropped.tif'),
    zstack_xy_um=meta['zstack_xy_um'],
)
result = ps.seed_from_auto_search(cfg, meta['anchor_sec'])
print(result['plane'], result['n_landmarks'], result['tilt_deg'])
# compare against metadata.json's validated_test_cases.mode1_auto_search.result
"
```
Note: `run_subject`/the `xenium-autocoreg` CLI now take an arbitrary `SubjectConfig` (or a JSON
file matching its fields, via `config.subject_config_from_json`) directly, so you can also drive
the fixture through the full CLI/`run_subject` path -- write the fixture's fields out as a
`config.json` (or call `run_subject(cfg, ...)` directly with the `SubjectConfig` built above)
instead of only the lower-level stage functions shown here.
`metadata.json`'s `z_base_in_cropped_volume` (NOT `z_base_original`) is the correct `z_base` for
this cropped volume. Either pose-seeding mode should reproduce the exact
`plane`/`n_landmarks`/`tilt_deg` recorded in `metadata.json`'s `validated_test_cases` -- if it
doesn't, that's a real regression.

## Known limitations

- The SVD scale-clip in `chain_refine.clip_affine_scale` bounds scale but not shear -- a
  weak-correlation section can still show shear-driven cell-shape distortion.
- `auto` pose-seeding can fail outright (see `pose_seed.seed_from_auto_search`'s docstring) if the
  true pose sits outside the searched position window; use `center-rotation` when it does.
- A third, corner-based pose-seeding mode is a documented TODO -- see `pose_seed.seed_from_corners`
  for exactly why it's not implemented, and what real click data is needed before it can be. It is
  not exposed via `run_subject`/the CLI/`--pose-json` (only `auto` and `center-rotation` are).
- This package does not resolve `SubjectConfig` from any mounted-asset naming convention itself
  (`config.subject_config_from_json` is a generic "JSON matching the dataclass fields" loader, not
  a data-asset resolver) -- write a resolver in your own pipeline/capsule for that. See
  [`ophys-xenium-autocoreg`](https://github.com/AllenNeuralDynamics/ophys-xenium-autocoreg)'s
  `subject_resolver.py` for a reference implementation (including its own fallback
  field-of-view handling for acquisitions with no metadata file).
