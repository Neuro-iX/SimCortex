import numpy as np

from simcortex.utils.tca import bit_map, check_topology


def test_check_topology_returns_true_for_processed_center():
    image = np.zeros((3, 3, 3), dtype=np.float64)
    image[1, 1, 1] = 1.0

    lookup_table = np.zeros(1, dtype=np.bool_)
    bit = bit_map()

    result = check_topology(
        image,
        lookup_table,
        bit,
    )

    assert result is True


def test_check_topology_uses_false_lookup_entry():
    image = np.zeros((3, 3, 3), dtype=np.float64)

    lookup_table = np.zeros(1, dtype=np.bool_)
    bit = bit_map()

    result = check_topology(
        image,
        lookup_table,
        bit,
    )

    assert result is False


def test_check_topology_uses_true_lookup_entry():
    image = np.zeros((3, 3, 3), dtype=np.float64)

    lookup_table = np.ones(1, dtype=np.bool_)
    bit = bit_map()

    result = check_topology(
        image,
        lookup_table,
        bit,
    )

    assert result is True
