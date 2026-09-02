"""Anchor-section selection: pick the Xenium section to run the (expensive) GT-free initial search
on. Generalizes the 816462 recipe (coreg-match-recipe project memory) -- the first section that's
both cell-rich (>= anchor_min_cells) and past a density plateau (>= anchor_plateau_frac * max count)
-- from reporter+ counts to whatever population `populations.load_xenium_cells` returns (reporter+
or all-cells, depending on data availability)."""
import numpy as np
from .populations import load_xenium_cells

ANCHOR_MIN_CELLS = 1000
ANCHOR_PLATEAU_FRAC = 0.80


def select_anchor_section(cfg, verbose=True):
    counts = {}
    for sec in cfg.sections:
        try:
            _, xy, _, _ = load_xenium_cells(cfg, sec, aligned=True)
            counts[sec] = len(xy)
        except FileNotFoundError:
            continue
    if not counts:
        raise RuntimeError(f"no usable sections found for subject {cfg.subject_id}")
    max_count = max(counts.values())
    for sec in sorted(counts):
        if counts[sec] >= ANCHOR_MIN_CELLS and counts[sec] >= ANCHOR_PLATEAU_FRAC * max_count:
            if verbose:
                print(f"anchor section = {sec} (count={counts[sec]}, plateau max={max_count})", flush=True)
            return sec, counts
    # fallback: just the densest section
    best = max(counts, key=counts.get)
    if verbose:
        print(f"no section met the plateau criterion; falling back to densest section {best} "
              f"(count={counts[best]})", flush=True)
    return best, counts
