from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    REPO_ROOT
    / "evaluation"
    / "summarize_results.py"
)


def _load_summary_module():
    spec = importlib.util.spec_from_file_location(
        "simcortex_summary_regression",
        SUMMARY_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def summary_module():
    return _load_summary_module()


@pytest.fixture
def synthetic_inputs():
    surfaces = [
        "lh_white",
        "lh_pial",
        "rh_white",
        "rh_pial",
    ]

    surface_rows = []
    thickness_rows = []
    collision_rows = []

    for case_index in range(2):
        subject = f"sub-{case_index:03d}"
        session = "ses-01"
        case_id = (
            f"{subject}_{session}"
        )

        for surface_index, surface in enumerate(
            surfaces
        ):
            base = (
                1.0
                + case_index
                + surface_index * 0.1
            )

            surface_rows.append(
                {
                    "method": "SimCortex",
                    "dataset": "demo",
                    "subject": subject,
                    "session": session,
                    "case_id": case_id,
                    "surface": surface,
                    "ChamferPCL1_mm": base,
                    "ASSD_mm": base + 1.0,
                    "HD90_mm": base + 2.0,
                    "SIF_pct": base + 3.0,
                }
            )

        thickness_rows.append(
            {
                "method": "SimCortex",
                "dataset": "demo",
                "subject": subject,
                "session": session,
                "case_id": case_id,
                "lh_thickness_abs_error_mm":
                    0.2 + case_index,
                "rh_thickness_abs_error_mm":
                    0.4 + case_index,
            }
        )

        collision_rows.append(
            {
                "method": "SimCortex",
                "dataset": "demo",
                "subject": subject,
                "session": session,
                "case_id": case_id,
                "collision_pct_union_mean4":
                    2.0 + case_index,
                "union_status": "OK",
            }
        )

    return (
        pd.DataFrame(
            surface_rows
        ),
        pd.DataFrame(
            thickness_rows
        ),
        pd.DataFrame(
            collision_rows
        ),
    )


def test_case_aggregation_exact(
    summary_module,
    synthetic_inputs,
) -> None:
    (
        surface_df,
        thickness_df,
        collision_df,
    ) = synthetic_inputs

    surface_case = (
        summary_module
        .build_case_surface_metrics(
            surface_df,
            None,
        )
    )

    thickness_case = (
        summary_module
        .build_case_thickness_metrics(
            thickness_df,
            None,
        )
    )

    collision_case = (
        summary_module
        .build_case_collision_metrics(
            collision_df,
            None,
        )
    )

    case_df, problems = (
        summary_module
        .build_case_metrics(
            surface_case,
            thickness_case,
            collision_case,
        )
    )

    assert problems == []

    assert len(case_df) == 2

    expected = pd.DataFrame(
        [
            {
                "method": "SimCortex",
                "dataset": "demo",
                "subject": "sub-000",
                "session": "ses-01",
                "case_id":
                    "sub-000_ses-01",
                "ChamferPCL1_mm": 1.15,
                "ASSD_mm": 2.15,
                "HD90_mm": 3.15,
                "SIF_pct": 4.15,
                "ThicknessAbsErr_mm": 0.3,
                "CollisionPctUnion_mean4":
                    2.0,
            },
            {
                "method": "SimCortex",
                "dataset": "demo",
                "subject": "sub-001",
                "session": "ses-01",
                "case_id":
                    "sub-001_ses-01",
                "ChamferPCL1_mm": 2.15,
                "ASSD_mm": 3.15,
                "HD90_mm": 4.15,
                "SIF_pct": 5.15,
                "ThicknessAbsErr_mm": 1.3,
                "CollisionPctUnion_mean4":
                    3.0,
            },
        ]
    )

    compare_columns = [
        "method",
        "dataset",
        "subject",
        "session",
        "case_id",
        "ChamferPCL1_mm",
        "ASSD_mm",
        "HD90_mm",
        "SIF_pct",
        "ThicknessAbsErr_mm",
        "CollisionPctUnion_mean4",
    ]

    actual = (
        case_df[
            compare_columns
        ]
        .sort_values(
            [
                "dataset",
                "case_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )


def test_overall_summary_exact(
    summary_module,
    synthetic_inputs,
) -> None:
    (
        surface_df,
        thickness_df,
        collision_df,
    ) = synthetic_inputs

    surface_case = (
        summary_module
        .build_case_surface_metrics(
            surface_df,
            None,
        )
    )

    thickness_case = (
        summary_module
        .build_case_thickness_metrics(
            thickness_df,
            None,
        )
    )

    collision_case = (
        summary_module
        .build_case_collision_metrics(
            collision_df,
            None,
        )
    )

    case_df, problems = (
        summary_module
        .build_case_metrics(
            surface_case,
            thickness_case,
            collision_case,
        )
    )

    assert problems == []

    overall = (
        summary_module
        .summarize_overall(
            case_df
        )
    )

    assert len(overall) == 1

    row = overall.iloc[0]

    assert row["method"] == "SimCortex"
    assert row["n_cases"] == 2

    expected = {
        "ChamferPCL1_mm_count": 2,
        "ChamferPCL1_mm_mean":
            1.65,
        "ChamferPCL1_mm_std":
            0.7071067811865476,
        "ChamferPCL1_mm_median":
            1.65,
        "ChamferPCL1_mm_min":
            1.15,
        "ChamferPCL1_mm_max":
            2.15,

        "ASSD_mm_count": 2,
        "ASSD_mm_mean":
            2.65,
        "ASSD_mm_std":
            0.7071067811865476,
        "ASSD_mm_median":
            2.65,
        "ASSD_mm_min":
            2.15,
        "ASSD_mm_max":
            3.15,

        "HD90_mm_count": 2,
        "HD90_mm_mean":
            3.65,
        "HD90_mm_std":
            0.7071067811865476,
        "HD90_mm_median":
            3.65,
        "HD90_mm_min":
            3.15,
        "HD90_mm_max":
            4.15,

        "SIF_pct_count": 2,
        "SIF_pct_mean":
            4.65,
        "SIF_pct_std":
            0.7071067811865476,
        "SIF_pct_median":
            4.65,
        "SIF_pct_min":
            4.15,
        "SIF_pct_max":
            5.15,

        "ThicknessAbsErr_mm_count":
            2,
        "ThicknessAbsErr_mm_mean":
            0.8,
        "ThicknessAbsErr_mm_std":
            0.7071067811865476,
        "ThicknessAbsErr_mm_median":
            0.8,
        "ThicknessAbsErr_mm_min":
            0.3,
        "ThicknessAbsErr_mm_max":
            1.3,

        "CollisionPctUnion_mean4_count":
            2,
        "CollisionPctUnion_mean4_mean":
            2.5,
        "CollisionPctUnion_mean4_std":
            0.7071067811865476,
        "CollisionPctUnion_mean4_median":
            2.5,
        "CollisionPctUnion_mean4_min":
            2.0,
        "CollisionPctUnion_mean4_max":
            3.0,
    }

    for column, expected_value in (
        expected.items()
    ):
        actual_value = row[column]

        if column.endswith(
            "_count"
        ):
            assert int(
                actual_value
            ) == expected_value
        else:
            assert float(
                actual_value
            ) == pytest.approx(
                expected_value,
                rel=0.0,
                abs=1e-12,
            )


def test_dataset_summary_exact(
    summary_module,
    synthetic_inputs,
) -> None:
    (
        surface_df,
        thickness_df,
        collision_df,
    ) = synthetic_inputs

    surface_case = (
        summary_module
        .build_case_surface_metrics(
            surface_df,
            None,
        )
    )

    thickness_case = (
        summary_module
        .build_case_thickness_metrics(
            thickness_df,
            None,
        )
    )

    collision_case = (
        summary_module
        .build_case_collision_metrics(
            collision_df,
            None,
        )
    )

    case_df, problems = (
        summary_module
        .build_case_metrics(
            surface_case,
            thickness_case,
            collision_case,
        )
    )

    assert problems == []

    by_dataset = (
        summary_module
        .summarize_by_dataset(
            case_df
        )
    )

    assert len(by_dataset) == 1

    row = by_dataset.iloc[0]

    assert row["method"] == "SimCortex"
    assert row["dataset"] == "demo"
    assert row["n_cases"] == 2

    assert float(
        row[
            "ChamferPCL1_mm_mean"
        ]
    ) == pytest.approx(
        1.65,
        rel=0.0,
        abs=1e-12,
    )

    assert float(
        row[
            "ThicknessAbsErr_mm_mean"
        ]
    ) == pytest.approx(
        0.8,
        rel=0.0,
        abs=1e-12,
    )

    assert float(
        row[
            "CollisionPctUnion_mean4_mean"
        ]
    ) == pytest.approx(
        2.5,
        rel=0.0,
        abs=1e-12,
    )


def test_surface_aggregation_requires_four_surfaces(
    summary_module,
    synthetic_inputs,
) -> None:
    (
        surface_df,
        _,
        _,
    ) = synthetic_inputs

    bad = surface_df[
        ~(
            (
                surface_df["case_id"]
                == "sub-000_ses-01"
            )
            & (
                surface_df["surface"]
                == "lh_white"
            )
        )
    ].copy()

    with pytest.raises(
        ValueError,
        match=(
            "exactly the expected four surfaces"
        ),
    ):
        summary_module.build_case_surface_metrics(
            bad,
            None,
        )


def test_collision_summary_rejects_non_ok_union(
    summary_module,
    synthetic_inputs,
) -> None:
    (
        _,
        _,
        collision_df,
    ) = synthetic_inputs

    bad = collision_df.copy()

    bad.loc[
        bad.index[0],
        "union_status",
    ] = "problem"

    with pytest.raises(
        ValueError,
        match="union_status != OK",
    ):
        summary_module.build_case_collision_metrics(
            bad,
            None,
        )


def test_case_key_mismatch_is_reported(
    summary_module,
    synthetic_inputs,
) -> None:
    (
        surface_df,
        thickness_df,
        collision_df,
    ) = synthetic_inputs

    surface_case = (
        summary_module
        .build_case_surface_metrics(
            surface_df,
            None,
        )
    )

    thickness_case = (
        summary_module
        .build_case_thickness_metrics(
            thickness_df.iloc[
                [0]
            ].copy(),
            None,
        )
    )

    collision_case = (
        summary_module
        .build_case_collision_metrics(
            collision_df,
            None,
        )
    )

    case_df, problems = (
        summary_module
        .build_case_metrics(
            surface_case,
            thickness_case,
            collision_case,
        )
    )

    assert len(case_df) == 1

    assert any(
        "thickness metrics: missing 1 cases"
        in problem
        for problem in problems
    )
