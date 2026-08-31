"""Regression tests for the historical SimCortex topology checks.

These tests intentionally preserve the behavior of the original implementation.
In particular, ``check_topology`` returns ``None`` when the center voxel is
already inside the processed object. Although this may look unusual, changing
that behavior would alter the established topology-correction implementation.
"""

import numpy as np

from simcortex.utils.tca import bit_map, check_topology


def test_check_topology_center_one_preserves_historical_none():
    """Center voxel equal to one must preserve the original None return."""
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1, 1] = 1

    bit = bit_map()

    # The LUT is not consulted in this branch.
    lut = np.zeros(1, dtype=np.uint8)

    result = check_topology(img, lut, bit)

    assert result is None


def test_check_topology_zero_lut_entry_returns_false():
    """A zero LUT entry must be classified as topology-critical."""
    img = np.zeros((3, 3, 3), dtype=np.uint8)

    bit = bit_map()

    # All-zero neighborhood produces key == 0.
    lut = np.zeros(1, dtype=np.uint8)

    result = check_topology(img, lut, bit)

    assert result is False


def test_check_topology_nonzero_lut_entry_returns_true():
    """A nonzero LUT entry must be classified as non-critical."""
    img = np.zeros((3, 3, 3), dtype=np.uint8)

    bit = bit_map()

    # All-zero neighborhood again produces key == 0.
    lut = np.ones(1, dtype=np.uint8)

    result = check_topology(img, lut, bit)

    assert result is True


def test_bit_map_preserves_center_as_zero():
    """The center voxel must not contribute to the 26-neighbor LUT key."""
    bit = bit_map()

    assert bit.shape == (3, 3, 3)
    assert bit[1, 1, 1] == 0
    assert np.count_nonzero(bit) == 26
