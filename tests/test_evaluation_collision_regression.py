from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLISION_RUNNER_PATH = (
    REPO_ROOT
    / "evaluation"
    / "evaluate_collisions.py"
)

METHOD_NAME = "SimCortex"

EXPECTED_PAIR_ORDER = [
    "white_pial_left",
    "white_pial_right",
    "pial_lr",
    "white_lr",
    "cross_lhwhite_rhpial",
    "cross_rhwhite_lhpial",
]

EXPECTED_PAIR_RESULTS = {
    "white_pial_left": {
        "num_contacts": 20,
        "intersecting_faces_A": 6,
        "intersecting_faces_B": 6,
        "pct_faces_A": 50.0,
        "pct_faces_B": 50.0,
    },
    "white_pial_right": {
        "num_contacts": 20,
        "intersecting_faces_A": 6,
        "intersecting_faces_B": 6,
        "pct_faces_A": 50.0,
        "pct_faces_B": 50.0,
    },
    "pial_lr": {
        "num_contacts": 20,
        "intersecting_faces_A": 6,
        "intersecting_faces_B": 6,
        "pct_faces_A": 50.0,
        "pct_faces_B": 50.0,
    },
    "white_lr": {
        "num_contacts": 44,
        "intersecting_faces_A": 8,
        "intersecting_faces_B": 8,
        "pct_faces_A":
            66.66666666666667,
        "pct_faces_B":
            66.66666666666667,
    },
    "cross_lhwhite_rhpial": {
        "num_contacts": 20,
        "intersecting_faces_A": 6,
        "intersecting_faces_B": 6,
        "pct_faces_A": 50.0,
        "pct_faces_B": 50.0,
    },
    "cross_rhwhite_lhpial": {
        "num_contacts": 20,
        "intersecting_faces_A": 6,
        "intersecting_faces_B": 6,
        "pct_faces_A": 50.0,
        "pct_faces_B": 50.0,
    },
}

EXPECTED_SURFACE_UNION = {
    "lh_white": {
        "faces_total": 12,
        "faces_colliding_union": 10,
        "collision_pct_union":
            83.33333333333334,
    },
    "lh_pial": {
        "faces_total": 12,
        "faces_colliding_union": 10,
        "collision_pct_union":
            83.33333333333334,
    },
    "rh_white": {
        "faces_total": 12,
        "faces_colliding_union": 8,
        "collision_pct_union":
            66.66666666666666,
    },
    "rh_pial": {
        "faces_total": 12,
        "faces_colliding_union": 8,
        "collision_pct_union":
            66.66666666666666,
    },
}

EXPECTED_UNION_MEAN4 = 75.0
EXPECTED_UNION_MAX4 = (
    83.33333333333334
)
EXPECTED_UNION_SUM4 = 36


def _load_runner():
    spec = (
        importlib.util
        .spec_from_file_location(
            "simcortex_collision_regression",
            COLLISION_RUNNER_PATH,
        )
    )

    assert spec is not None
    assert spec.loader is not None

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


@pytest.fixture(scope="module")
def runner():
    module = _load_runner()

    if not module.HAS_FCL:
        pytest.skip(
            "python-fcl is not available: "
            f"{module.FCL_IMPORT_ERROR}"
        )

    return module


def _make_box(
    path: Path,
    center: tuple[
        float,
        float,
        float,
    ],
) -> None:
    mesh = trimesh.creation.box(
        extents=(
            2.0,
            2.0,
            2.0,
        )
    )

    mesh.apply_translation(
        np.asarray(
            center,
            dtype=np.float64,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh.export(
        path
    )


@pytest.fixture(scope="module")
def collision_payload(
    tmp_path_factory,
):
    root = (
        tmp_path_factory
        .mktemp(
            "collision_regression"
        )
    )

    # These small offsets avoid coincident meshes
    # while ensuring that every one of the six
    # anatomical surface pairs intersects.
    centers = {
        "lh_white": (
            -0.30,
            0.00,
            0.00,
        ),
        "lh_pial": (
            -0.10,
            0.05,
            0.07,
        ),
        "rh_white": (
            0.30,
            0.00,
            -0.05,
        ),
        "rh_pial": (
            0.10,
            -0.05,
            0.12,
        ),
    }

    paths = {}

    for surface, center in (
        centers.items()
    ):
        path = (
            root
            / f"{surface}.ply"
        )

        _make_box(
            path,
            center,
        )

        paths[
            surface
        ] = str(path)

    return {
        "method": METHOD_NAME,
        "mesh_set": "pred",
        "dataset": "demo",
        "case_id":
            "sub-001_ses-01",
        "subject": "sub-001",
        "session": "ses-01",
        "paths": paths,
        "caps": [
            50000,
            200000,
            500000,
        ],
        # Disable worker address-space limiting
        # for this small regression test.
        "mem_gb": 0,
    }


@pytest.fixture(scope="module")
def direct_result(
    runner,
    collision_payload,
):
    return (
        runner.worker_evaluate_case(
            collision_payload
        )
    )


def _normalize(
    value: Any,
) -> Any:
    """Normalize non-scientific serialization details."""
    if isinstance(
        value,
        dict,
    ):
        result = {}

        for key, item in (
            value.items()
        ):
            if key == (
                "worker_stderr_tail"
            ):
                result[
                    key
                ] = "<diagnostic>"
            else:
                result[
                    key
                ] = _normalize(
                    item
                )

        return result

    if isinstance(
        value,
        list,
    ):
        return [
            _normalize(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return tuple(
            _normalize(item)
            for item in value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return _normalize(
            value.tolist()
        )

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        result = float(
            value
        )

        if math.isnan(
            result
        ):
            return "<nan>"

        return result

    if (
        isinstance(
            value,
            float,
        )
        and math.isnan(
            value
        )
    ):
        return "<nan>"

    return value


def test_collision_pair_order_and_detection(
    runner,
    direct_result,
) -> None:
    rows = direct_result[
        "pair_rows"
    ]

    assert len(
        rows
    ) == 6

    assert [
        row["pair"]
        for row in rows
    ] == EXPECTED_PAIR_ORDER

    assert list(
        runner.COLLISION_PAIRS.keys()
    ) == EXPECTED_PAIR_ORDER

    for row in rows:
        assert bool(
            row[
                "collision_detected"
            ]
        )

        assert (
            row[
                "count_status"
            ]
            == "OK"
        )


def test_pairwise_fcl_regression(
    direct_result,
) -> None:
    rows = {
        row["pair"]: row
        for row in direct_result[
            "pair_rows"
        ]
    }

    assert set(
        rows
    ) == set(
        EXPECTED_PAIR_RESULTS
    )

    for pair, expected in (
        EXPECTED_PAIR_RESULTS.items()
    ):
        row = rows[pair]

        assert int(
            row[
                "num_contacts"
            ]
        ) == expected[
            "num_contacts"
        ]

        assert int(
            row[
                "intersecting_faces_A"
            ]
        ) == expected[
            "intersecting_faces_A"
        ]

        assert int(
            row[
                "intersecting_faces_B"
            ]
        ) == expected[
            "intersecting_faces_B"
        ]

        assert int(
            row[
                "total_faces_A"
            ]
        ) == 12

        assert int(
            row[
                "total_faces_B"
            ]
        ) == 12

        assert float(
            row[
                "pct_faces_A"
            ]
        ) == pytest.approx(
            expected[
                "pct_faces_A"
            ],
            rel=0.0,
            abs=1e-12,
        )

        assert float(
            row[
                "pct_faces_B"
            ]
        ) == pytest.approx(
            expected[
                "pct_faces_B"
            ],
            rel=0.0,
            abs=1e-12,
        )

        assert not bool(
            row[
                "num_contacts_saturated"
            ]
        )

        assert bool(
            row[
                "contact_count_exact"
            ]
        )

        assert int(
            row[
                "contact_index_failures"
            ]
        ) == 0

        assert int(
            row[
                "max_contacts_used"
            ]
        ) == 50000


def test_surface_union_regression(
    direct_result,
) -> None:
    row = direct_result[
        "union_row"
    ]

    assert (
        row[
            "union_status"
        ]
        == "OK"
    )

    assert int(
        row[
            "n_pair_rows"
        ]
    ) == 6

    assert int(
        row[
            "n_collision_true_pairs"
        ]
    ) == 6

    assert int(
        row[
            "n_saturated_pairs"
        ]
    ) == 0

    assert int(
        row[
            "n_nonexact_true_pairs"
        ]
    ) == 0

    for surface, expected in (
        EXPECTED_SURFACE_UNION
        .items()
    ):
        assert int(
            row[
                f"{surface}_faces_total"
            ]
        ) == expected[
            "faces_total"
        ]

        assert int(
            row[
                (
                    f"{surface}_"
                    "faces_colliding_union"
                )
            ]
        ) == expected[
            "faces_colliding_union"
        ]

        assert float(
            row[
                (
                    f"{surface}_"
                    "collision_pct_union"
                )
            ]
        ) == pytest.approx(
            expected[
                "collision_pct_union"
            ],
            rel=0.0,
            abs=1e-12,
        )

    assert float(
        row[
            "collision_pct_union_mean4"
        ]
    ) == pytest.approx(
        EXPECTED_UNION_MEAN4,
        rel=0.0,
        abs=1e-12,
    )

    assert float(
        row[
            "collision_pct_union_max4"
        ]
    ) == pytest.approx(
        EXPECTED_UNION_MAX4,
        rel=0.0,
        abs=1e-12,
    )

    assert int(
        row[
            "collision_faces_union_sum4"
        ]
    ) == EXPECTED_UNION_SUM4


def test_contact_total_regression(
    direct_result,
) -> None:
    rows = direct_result[
        "pair_rows"
    ]

    expected_contacts = sum(
        result[
            "num_contacts"
        ]
        for result in (
            EXPECTED_PAIR_RESULTS
            .values()
        )
    )

    assert (
        expected_contacts
        == 144
    )

    actual_contacts = sum(
        int(
            row[
                "num_contacts"
            ]
        )
        for row in rows
    )

    assert (
        actual_contacts
        == expected_contacts
    )

    union_row = direct_result[
        "union_row"
    ]

    assert int(
        union_row[
            "total_contacts_sum"
        ]
    ) == expected_contacts


def test_subprocess_worker_matches_direct_worker(
    runner,
    collision_payload,
    direct_result,
) -> None:
    subprocess_result = (
        runner.run_case_worker(
            collision_payload,
            timeout_sec=60,
        )
    )

    assert _normalize(
        subprocess_result
    ) == _normalize(
        direct_result
    )


def test_public_collision_identity(
    runner,
) -> None:
    assert (
        runner.METHOD_NAME
        == METHOD_NAME
    )

    assert (
        runner.COLLISION_SCHEMA_VERSION
        == (
            "simcortex_evaluation_"
            "fcl_collision_union_v1.0"
        )
    )

    assert list(
        runner.SURFACE_KEYS
    ) == [
        "lh_white",
        "lh_pial",
        "rh_white",
        "rh_pial",
    ]
