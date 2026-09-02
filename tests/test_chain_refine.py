import numpy as np
from xenium_autocoreg.chain_refine import clip_affine_scale, SCALE_CLIP


def test_clip_affine_scale_bounds_a_distorted_candidate():
    """A weak-signal fit can produce a wildly anisotropic scale (e.g. 0.94 / 0.55) that should
    never survive uncorrected."""
    M = np.eye(3)
    M[:2, :2] = np.diag([0.94, 0.55])   # anisotropic, one axis far outside the physical prior
    clipped = clip_affine_scale(M)
    U, S, Vt = np.linalg.svd(clipped[:2, :2])
    assert np.all(S >= SCALE_CLIP[0] - 1e-9)
    assert np.all(S <= SCALE_CLIP[1] + 1e-9)


def test_clip_affine_scale_preserves_rotation():
    a = np.radians(12.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = np.eye(3)
    M[:2, :2] = R * 1.3   # isotropic but out-of-range scale, pure rotation otherwise
    clipped = clip_affine_scale(M)
    rot_after = np.degrees(np.arctan2(clipped[1, 0], clipped[0, 0]))
    assert np.isclose(rot_after, 12.0, atol=1e-6)


def test_clip_affine_scale_is_a_noop_when_already_in_range():
    a = np.radians(-5.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) * 1.0
    M = np.eye(3); M[:2, :2] = R
    assert np.allclose(clip_affine_scale(M), M, atol=1e-9)


def test_clip_affine_scale_does_not_mutate_input():
    M = np.eye(3); M[:2, :2] = np.diag([2.0, 2.0])
    M_copy = M.copy()
    clip_affine_scale(M)
    assert np.allclose(M, M_copy)
