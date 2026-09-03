import os
import pytest
from xenium_autocoreg.resources import resolve_num_cpus, pool_map


def _available():
    return os.cpu_count() or 1


def test_resolve_num_cpus_none_is_auto():
    assert resolve_num_cpus(None) == (None if _available() == 1 else _available())


def test_resolve_num_cpus_zero_is_auto():
    assert resolve_num_cpus(0) == resolve_num_cpus(None)


def test_resolve_num_cpus_exceeding_available_falls_back_to_auto():
    assert resolve_num_cpus(_available() + 1000) == resolve_num_cpus(None)


def test_resolve_num_cpus_one_means_serial():
    assert resolve_num_cpus(1) is None


def test_resolve_num_cpus_valid_value_passed_through():
    if _available() >= 2:
        assert resolve_num_cpus(2) == 2


def test_resolve_num_cpus_negative_raises():
    with pytest.raises(ValueError):
        resolve_num_cpus(-1)


def _square(x):
    return x * x


def test_pool_map_serial_matches_parallel_order_and_values():
    items = list(range(10))
    serial = pool_map(_square, items, 1)
    parallel = pool_map(_square, items, None)  # auto -- may or may not actually parallelize
    assert serial == [x * x for x in items]
    assert serial == parallel


_OFFSET = None


def _init_offset(offset):
    global _OFFSET
    _OFFSET = offset


def _initted_add(x):
    return x + _OFFSET


def test_pool_map_serial_runs_initializer_in_process():
    # Mirrors the real _pool_init pattern (initial_match.py/tile_warp.py): the initializer sets a
    # module-level global that the mapped function then reads -- for the serial path, pool_map
    # must call the initializer directly in-process (not skip it) before mapping.
    result = pool_map(_initted_add, [1, 2, 3], 1, initializer=_init_offset, initargs=(100,))
    assert result == [101, 102, 103]
