import numpy as np
from xenium_autocoreg.geometry import (
    find_affine_transformation_2d, decompose_affine, apply_affine_based_on_reference_2d,
)


def _make_affine(scale=0.8, rot_deg=12.0, tx=5.0, ty=-3.0):
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * scale
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = [tx, ty]
    return M


def test_find_affine_transformation_2d_recovers_known_affine():
    M_true = _make_affine()
    rng = np.random.default_rng(0)
    pts1 = rng.uniform(-50, 50, size=(20, 2))
    pts1_h = np.hstack([pts1, np.ones((20, 1))])
    pts2 = (M_true @ pts1_h.T).T[:, :2]

    M_fit = find_affine_transformation_2d(pts1, pts2)
    assert np.allclose(M_fit, M_true, atol=1e-8)


def test_decompose_affine_recovers_scale_and_rotation():
    M = _make_affine(scale=0.77, rot_deg=15.0)
    dec = decompose_affine(M)
    assert np.isclose(dec["scale1"], 0.77, atol=1e-6)
    assert np.isclose(dec["scale2"], 0.77, atol=1e-6)
    assert np.isclose(dec["rotation_deg"], 15.0, atol=1e-6)
    assert np.isclose(dec["shear"], 0.0, atol=1e-6)


def test_apply_affine_based_on_reference_2d_identity_is_passthrough():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = apply_affine_based_on_reference_2d(img, np.eye(3), (10, 10), order=0)
    assert np.allclose(out, img)


def test_apply_affine_based_on_reference_2d_translation():
    img = np.zeros((20, 20), dtype=np.float32)
    img[5, 5] = 1.0
    # M maps Xenium(rc) -> z-stack(rc); a pure +2,+2 translation in the OUTPUT (z-stack) frame
    # means the source pixel sampled at output (7,7) should be the one at input (5,5).
    M = np.array([[1, 0, 2], [0, 1, 2], [0, 0, 1]], float)
    out = apply_affine_based_on_reference_2d(img, M, (20, 20), order=0)
    assert out[7, 7] == 1.0
