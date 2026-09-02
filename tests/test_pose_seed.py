import pytest
from xenium_autocoreg.pose_seed import PoseSeed, seed_from_center_rotation, seed_from_corners


def test_seed_from_center_rotation_is_a_plain_passthrough():
    seed = seed_from_center_rotation(center_um=(1363.149, 1535.675), rotation_deg=9.610, scale=0.80)
    assert isinstance(seed, PoseSeed)
    assert seed.center_um == (1363.149, 1535.675)
    assert seed.rotation_deg == pytest.approx(9.610)
    assert seed.scale == pytest.approx(0.80)


def test_seed_from_center_rotation_default_scale():
    seed = seed_from_center_rotation((0.0, 0.0), 0.0)
    assert seed.scale == pytest.approx(0.80)


def test_seed_from_corners_is_not_yet_implemented():
    """Documents the TODO as an explicit, checked contract rather than a silent gap -- see
    pose_seed.seed_from_corners's docstring for why this isn't implemented yet."""
    with pytest.raises(NotImplementedError):
        seed_from_corners(
            xenium_trapezoid_corners_um=[(0, 0), (100, 0), (100, 100), (0, 100)],
            top_edge=0,
        )
