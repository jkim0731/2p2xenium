"""xenium_autocoreg -- GT-free coregistration of an in-vivo 2-photon z-stack to Xenium spatial
transcriptomics sections. See the top-level README for the process, required input data, and
output structure.

Default acquisition parameters -- starting points only, every one is overridable per subject via
`SubjectConfig` (see config.py). Nothing in the algorithm assumes these values.
"""
DEFAULT_XENIUM_PX_UM = 0.2125 * 3.9996   # Xenium morphology-image pixel size (um/px)
DEFAULT_Z_STEP_UM = 1.0                  # z-stack native axial resolution (um/plane)
DEFAULT_ZSTACK_SCALE_TO_XENIUM = 0.80    # linear factor: multiply a z-stack point's um coordinates
                                          # by this to land in the Xenium-aligned frame's um scale

__all__ = ["DEFAULT_XENIUM_PX_UM", "DEFAULT_Z_STEP_UM", "DEFAULT_ZSTACK_SCALE_TO_XENIUM"]
__version__ = "0.1.0"
