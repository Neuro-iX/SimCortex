from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation"
RUNNER_PATH = EVAL_DIR / "evaluate_metrics.py"

METHOD_NAME = "SimCortex"

N_SAMPLES = 2000
N_THICKNESS_SAMPLES = 1000
SEED = 12345


EXPECTED_SURFACE_METRICS = {
    "lh_white": {
        "ASSD_mm": 0.19308952242136002,
        "HD90_mm": 0.371336430311203,
        "ChamferPCL1_mm": 0.4550492912530899,
        "SIF_pct": 0.0,
    },
    "lh_pial": {
        "ASSD_mm": 0.1926257386803627,
        "HD90_mm": 0.3713364899158478,
        "ChamferPCL1_mm": 0.5159916579723358,
        "SIF_pct": 0.0,
    },
    "rh_white": {
        "ASSD_mm": 0.13314079493284225,
        "HD90_mm": 0.27698370814323425,
        "ChamferPCL1_mm": 0.4151132106781006,
        "SIF_pct": 0.0,
    },
    "rh_pial": {
        "ASSD_mm": 0.13343573361635208,
        "HD90_mm": 0.27698421478271484,
        "ChamferPCL1_mm": 0.4910179078578949,
        "SIF_pct": 0.0,
    },
}

EXPECTED_THICKNESS = {
    "lh_thickness_pred_mean_mm":
        1.5051764249801636,
    "lh_thickness_gt_mean_mm":
        1.8814969062805176,
    "lh_thickness_abs_error_mm":
        0.376320481300354,
    "rh_thickness_pred_mean_mm":
        1.6935982704162598,
    "rh_thickness_gt_mean_mm":
        1.8813966512680054,
    "rh_thickness_abs_error_mm":
        0.1877983808517456,
}


def _load_runner():
    # evaluate_metrics.py imports metrics_core.py as a sibling module.
    eval_dir_string = str(
        EVAL_DIR.resolve()
    )

    if eval_dir_string not in sys.path:
        sys.path.insert(
            0,
            eval_dir_string,
        )

    spec = importlib.util.spec_from_file_location(
        "simcortex_metrics_regression_runner",
        RUNNER_PATH,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _make_sphere(
    path: Path,
    *,
    radius: float,
    center: tuple[
        float,
        float,
        float,
    ],
) -> None:
    mesh = trimesh.creation.icosphere(
        subdivisions=1,
        radius=radius,
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


def _build_case(
    root: Path,
):
    pred_root = (
        root / "pred"
    )
    gt_root = (
        root / "gt"
    )

    surfaces = {
        "lh_white": {
            "gt_radius": 10.0,
            "pred_radius": 10.2,
            "gt_center": (
                -20.0,
                0.0,
                0.0,
            ),
            "pred_center": (
                -19.8,
                -0.1,
                0.05,
            ),
        },
        "lh_pial": {
            "gt_radius": 12.0,
            "pred_radius": 11.8,
            "gt_center": (
                -20.0,
                0.0,
                0.0,
            ),
            "pred_center": (
                -19.8,
                -0.1,
                0.05,
            ),
        },
        "rh_white": {
            "gt_radius": 10.0,
            "pred_radius": 10.1,
            "gt_center": (
                20.0,
                0.0,
                0.0,
            ),
            "pred_center": (
                20.2,
                0.1,
                -0.05,
            ),
        },
        "rh_pial": {
            "gt_radius": 12.0,
            "pred_radius": 11.9,
            "gt_center": (
                20.0,
                0.0,
                0.0,
            ),
            "pred_center": (
                20.2,
                0.1,
                -0.05,
            ),
        },
    }

    dataset = "demo"
    case_id = "sub-001_ses-01"
    subject = "sub-001"
    session = "ses-01"

    pred_map = {}
    gt_map = {}

    for surface, config in (
        surfaces.items()
    ):
        pred_path = (
            pred_root
            / f"{surface}.ply"
        )

        gt_path = (
            gt_root
            / f"{surface}.ply"
        )

        _make_sphere(
            pred_path,
            radius=(
                config[
                    "pred_radius"
                ]
            ),
            center=(
                config[
                    "pred_center"
                ]
            ),
        )

        _make_sphere(
            gt_path,
            radius=(
                config[
                    "gt_radius"
                ]
            ),
            center=(
                config[
                    "gt_center"
                ]
            ),
        )

        pred_map[
            (
                METHOD_NAME,
                dataset,
                case_id,
                surface,
            )
        ] = pred_path

        gt_map[
            (
                dataset,
                case_id,
                surface,
            )
        ] = gt_path

    return {
        "method": METHOD_NAME,
        "dataset": dataset,
        "case_id": case_id,
        "subject": subject,
        "session": session,
        "pred_map": pred_map,
        "gt_map": gt_map,
        "device": torch.device(
            "cpu"
        ),
        "eval_set": "sample40",
        "n_samples": N_SAMPLES,
        "n_thickness_samples":
            N_THICKNESS_SAMPLES,
        "seed": SEED,
    }


def _run_case(
    runner,
    tmp_path: Path,
):
    kwargs = _build_case(
        tmp_path
    )

    return runner.evaluate_one_case(
        **kwargs
    )


def test_surface_metric_regression(
    runner,
    tmp_path,
) -> None:
    surface_rows, _ = _run_case(
        runner,
        tmp_path,
    )

    assert [
        row["surface"]
        for row in surface_rows
    ] == [
        "lh_white",
        "lh_pial",
        "rh_white",
        "rh_pial",
    ]

    assert len(
        surface_rows
    ) == 4

    for row in surface_rows:
        surface = row[
            "surface"
        ]

        expected = (
            EXPECTED_SURFACE_METRICS[
                surface
            ]
        )

        assert row[
            "method"
        ] == METHOD_NAME

        for metric, value in (
            expected.items()
        ):
            assert float(
                row[metric]
            ) == pytest.approx(
                value,
                rel=0.0,
                abs=1e-7,
            )


def test_thickness_metric_regression(
    runner,
    tmp_path,
) -> None:
    _, pair_row = _run_case(
        runner,
        tmp_path,
    )

    assert pair_row[
        "method"
    ] == METHOD_NAME

    for metric, value in (
        EXPECTED_THICKNESS.items()
    ):
        assert float(
            pair_row[metric]
        ) == pytest.approx(
            value,
            rel=0.0,
            abs=1e-7,
        )


def test_same_seed_reproduces_same_metrics(
    runner,
    tmp_path,
) -> None:
    first_surface, first_pair = (
        _run_case(
            runner,
            tmp_path / "first",
        )
    )

    second_surface, second_pair = (
        _run_case(
            runner,
            tmp_path / "second",
        )
    )

    assert len(
        first_surface
    ) == len(
        second_surface
    )

    for first, second in zip(
        first_surface,
        second_surface,
    ):
        assert (
            first["surface"]
            == second["surface"]
        )

        for metric in [
            "ASSD_mm",
            "HD90_mm",
            "ChamferPCL1_mm",
            "SIF_pct",
        ]:
            assert float(
                first[metric]
            ) == pytest.approx(
                float(
                    second[metric]
                ),
                rel=0.0,
                abs=0.0,
            )

    for metric in (
        EXPECTED_THICKNESS
    ):
        assert float(
            first_pair[metric]
        ) == pytest.approx(
            float(
                second_pair[metric]
            ),
            rel=0.0,
            abs=0.0,
        )


def test_seed_identity_is_simcortex(
    runner,
    tmp_path,
) -> None:
    surface_rows, pair_row = (
        _run_case(
            runner,
            tmp_path,
        )
    )

    assert all(
        row["method"]
        == METHOD_NAME
        for row in surface_rows
    )

    assert pair_row[
        "method"
    ] == METHOD_NAME

    assert runner.METHOD_NAME == (
        METHOD_NAME
    )
