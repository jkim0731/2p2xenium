import numpy as np
from xenium_autocoreg.somaprint import match, register_full


def _rigid_seed(rot_deg, scale, t):
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def seed(p):
        return (R @ (np.asarray(p) * scale).T).T + t
    return seed


def test_match_recovers_correspondence_under_known_rigid_transform_and_noise():
    """Two point clouds related by a known rigid transform (+ scale + small noise + a few
    unmatched clutter points on the xen side, simulating real cross-modal population mismatch),
    seeded with the (near-)correct transform. `match` should recover the true correspondence for
    the large majority of points -- this is the core primitive the whole initial-match /
    tilt-fitting pipeline is built on."""
    rng = np.random.default_rng(42)
    n = 80
    cz_xy = rng.uniform(0, 300, size=(n, 2))   # a scattered (non-lattice) point cloud, um

    rot_deg, scale, t = 8.0, 0.8, np.array([50.0, -20.0])
    seed = _rigid_seed(rot_deg, scale, t)
    xen_xy_true = seed(cz_xy)
    noise = rng.normal(0, 1.0, size=xen_xy_true.shape)   # 1um noise, small vs. ~40um spacing
    xen_xy = xen_xy_true + noise

    clutter = rng.uniform(xen_xy.min(0), xen_xy.max(0), size=(15, 2))
    xen_xy_full = np.vstack([xen_xy, clutter])

    seed_fn = _rigid_seed(rot_deg + 1.5, scale * 1.02, t + np.array([3.0, -2.0]))  # slightly off, realistic
    result = match(cz_xy, xen_xy_full, seed_fn, R_cand=30.0, k_cz=10, k_xen=10, n_best=5, anchor_frac=0.8)

    accepted = result["accepted"]
    assert len(accepted) >= int(0.5 * n), f"only {len(accepted)}/{n} accepted"
    correct = np.sum(accepted[:, 0] == accepted[:, 1])
    assert correct / len(accepted) >= 0.9, f"only {correct}/{len(accepted)} correct correspondences"


def test_register_full_certified_recovers_most_correspondence_under_known_transform():
    rng = np.random.default_rng(7)
    n = 90
    cz_xy = rng.uniform(0, 300, size=(n, 2))

    rot_deg, scale, t = 7.0, 0.8, np.array([45.0, -15.0])
    true_seed = _rigid_seed(rot_deg, scale, t)
    xen_xy_true = true_seed(cz_xy)
    xen_xy = xen_xy_true + rng.normal(0, 1.0, size=xen_xy_true.shape)
    clutter = rng.uniform(xen_xy.min(0), xen_xy.max(0), size=(20, 2))
    xen_xy_full = np.vstack([xen_xy, clutter])

    seed_fn = _rigid_seed(rot_deg + 2.0, scale * 1.03, t + np.array([4.0, -3.0]))
    out = register_full(
        cz_xy,
        xen_xy_full,
        seed_fn,
        R_cand=50.0,
        anchor_frac=0.6,
        tight_R_cand=25.0,
        tight_anchor_frac=0.8,
        k_cz=10,
        k_xen=10,
        n_best=5,
        min_anchors=6,
    )

    certified = out["certified"]
    assert len(certified) >= int(0.6 * n), f"only {len(certified)}/{n} certified"
    correct = np.sum(certified[:, 0] == certified[:, 1])
    assert correct / len(certified) >= 0.9, f"only {correct}/{len(certified)} certified correspondences correct"
