"""xenium_autocoreg -- GT-free coregistration of an in-vivo 2-photon z-stack to Xenium spatial
transcriptomics sections. See the top-level README for the process, required input data, and
output structure.
"""
XENIUM_S2_UM = 0.2125 * 3.9996   # morphology_focus/s2 pixel size
ZSTACK_XY_UM = 700.0 / 512.0     # 700 um FOV / 512 px (subject-specific override: SubjectConfig.zstack_xy_um)
Z_STEP_UM = 1.0
SURFACE_PLANE = 50               # pia convention

__all__ = ["XENIUM_S2_UM", "ZSTACK_XY_UM", "Z_STEP_UM", "SURFACE_PLANE"]
__version__ = "0.1.0"
