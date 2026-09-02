"""Normalization and QC-overlay helpers (curated from alignment_fucntions.py)."""
import numpy as np
import matplotlib.pyplot as plt


def normalize_image(image):
    """Min-max normalize to [0, 1]."""
    image = image.astype(np.float32)
    rng = np.max(image) - np.min(image)
    return (image - np.min(image)) / rng if rng else np.zeros_like(image)


def normalize_to_uint8(img, plow=1, phigh=99):
    """Percentile contrast-stretch to uint8 (default 1–99%)."""
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, [plow, phigh])
    if vmax <= vmin:
        return np.zeros(img.shape, np.uint8)
    return np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)


def overlay_rgb(moving, fixed, plow=5, phigh=99):
    """Red = moving (z-stack), cyan = fixed (Xenium) — the canonical overlap view."""
    r = normalize_to_uint8(moving, plow, phigh)
    c = normalize_to_uint8(fixed, plow, phigh)
    return np.stack([r, c, c], axis=-1)


def plot_alignment_qc(img1, img2, title_1="Image 1", title_2="Image 2",
                      title_overlay="Alignment QC", save=None, show=True):
    """Three-panel red/cyan/overlay QC figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    r = normalize_to_uint8(img1); c = normalize_to_uint8(img2)
    z = np.zeros_like(r)
    for ax, im, t in zip(axes,
                         [np.stack([r, z, z], -1), np.stack([z, c, c], -1),
                          np.stack([r, c, c], -1)],
                         [title_1, title_2, title_overlay]):
        ax.imshow(im); ax.set_title(t); ax.axis("off")
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
