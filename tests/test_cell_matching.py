import numpy as np
from xenium_autocoreg.reference_ported import find_mask_matches_fast, calculate_centroid


def _blob(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def test_find_mask_matches_fast_high_overlap_pair_matches():
    labels1 = np.zeros((50, 50), np.uint16)
    labels2 = np.zeros((50, 50), np.uint16)
    labels1[_blob(labels1.shape, 15, 15, 6)] = 1   # two nearly-identical blobs -> high IoU
    labels2[_blob(labels2.shape, 15, 15, 6)] = 1

    matches, unmatched1, unmatched2, iou = find_mask_matches_fast(labels1, labels2, threshold=0)
    assert (0, 0) in matches
    assert iou[0, 0] > 0.9


def test_find_mask_matches_fast_no_overlap_is_unmatched():
    labels1 = np.zeros((50, 50), np.uint16)
    labels2 = np.zeros((50, 50), np.uint16)
    labels1[_blob(labels1.shape, 10, 10, 5)] = 1
    labels2[_blob(labels2.shape, 40, 40, 5)] = 1   # far away, zero overlap

    matches, unmatched1, unmatched2, iou = find_mask_matches_fast(labels1, labels2, threshold=0)
    assert matches == []
    assert iou[0, 0] == 0.0


def test_find_mask_matches_fast_partial_shift_gives_intermediate_iou():
    labels1 = np.zeros((50, 50), np.uint16)
    labels2 = np.zeros((50, 50), np.uint16)
    labels1[_blob(labels1.shape, 25, 25, 8)] = 1
    labels2[_blob(labels2.shape, 25, 33, 8)] = 1   # shifted by ~1 radius -> partial overlap

    matches, unmatched1, unmatched2, iou = find_mask_matches_fast(labels1, labels2, threshold=0)
    assert (0, 0) in matches
    assert 0.05 < iou[0, 0] < 0.6


def test_calculate_centroid_matches_known_blob_center():
    mask = _blob((60, 60), 30, 20, 5)
    cx, cy = calculate_centroid(mask)
    assert np.isclose(cx, 20, atol=0.5)
    assert np.isclose(cy, 30, atol=0.5)


def test_calculate_centroid_empty_mask_is_nan():
    mask = np.zeros((10, 10), bool)
    cx, cy = calculate_centroid(mask)
    assert np.isnan(cx) and np.isnan(cy)
