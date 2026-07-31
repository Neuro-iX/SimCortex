from __future__ import annotations

import json
from pathlib import Path

import pytest

from simcortex.initsurf.generate import (
    _status_counts,
    _validate_params,
    expected_output_paths,
    outputs_complete,
    write_dataset_description,
)


DEFAULT_PARAMS = {
    "overwrite": False,
    "validate_affine": True,
    "affine_atol": 1.0e-4,
    "gap_size": 1,
    "sdf_sigma": 0.5,
    "topo_threshold": 16.0,
    "n_smooth": 1,
    "wm_start_level": -0.2,
    "wm_step": -0.08,
    "wm_min_level": -3.0,
    "wm_inset": 1.0,
    "pial_min_level": 1.8,
    "pial_max_level": 2.7,
    "pial_grid_step": 0.1,
    "pial_absolute_floor": 0.1,
}


def test_default_parameters_are_valid() -> None:
    _validate_params(DEFAULT_PARAMS)


def test_nonnegative_wm_step_is_rejected() -> None:
    params = dict(DEFAULT_PARAMS)
    params["wm_step"] = 0.08

    with pytest.raises(ValueError, match="wm_step"):
        _validate_params(params)


def test_completion_requires_success_marker(tmp_path: Path) -> None:
    paths = [
        Path(path)
        for path in expected_output_paths(
            str(tmp_path),
            "sub-0001",
            "01",
            "MNI152",
        )
    ]

    assert len(paths) == 13
    assert paths[-1].name.endswith("_desc-initsurf_qc.json")

    for path in paths[:-1]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    assert not outputs_complete(
        str(tmp_path),
        "sub-0001",
        "01",
        "MNI152",
    )

    paths[-1].write_text('{"Status": "ok"}\n', encoding="utf-8")

    assert outputs_complete(
        str(tmp_path),
        "sub-0001",
        "01",
        "MNI152",
    )


def test_unknown_status_counts_as_failure() -> None:
    assert _status_counts(
        [
            {"status": "ok"},
            {"status": "skipped_existing"},
            {"status": "unexpected"},
        ]
    ) == (1, 1, 1)


def test_dataset_description_is_rewritten_with_package_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset_description.json"
    path.write_text('{"GeneratedBy": [{"Version": "stale"}]}\n')

    write_dataset_description(str(tmp_path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["Name"] == "sc-initsurf"
    assert payload["DatasetType"] == "derivative"
    assert payload["GeneratedBy"][0]["Name"] == "SimCortex"
    assert payload["GeneratedBy"][0]["Version"] != "stale"
