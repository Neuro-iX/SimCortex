from __future__ import annotations

import json
from pathlib import Path

import pytest

import simcortex.initsurf.generate as generate_module
from simcortex.initsurf.generate import (
    _generate_subject,
    _run_jobs,
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


def _write_complete_outputs(
    root: str,
    subject: str = "sub-0001",
    ses: str = "01",
    space: str = "MNI152",
) -> None:
    for raw_path in expected_output_paths(root, subject, ses, space):
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


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


def test_subject_outputs_are_promoted_only_after_staged_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_root = tmp_path / "sc-initsurf"

    def fake_impl(**kwargs):
        _write_complete_outputs(
            kwargs["out_root"],
            kwargs["subject_id"],
            kwargs["ses"],
            kwargs["space"],
        )
        return {
            "status": "ok",
            "subject_id": kwargs["subject_id"],
            "ds_key": kwargs["ds_key"],
        }

    monkeypatch.setattr(
        generate_module,
        "_generate_subject_impl",
        fake_impl,
    )

    result = _generate_subject(
        subject_id="sub-0001",
        ds_key="TEST",
        preproc_root="/unused/preproc",
        seg_root="/unused/seg",
        out_root=str(out_root),
        ses="01",
        space="MNI152",
        params=dict(DEFAULT_PARAMS),
    )

    assert result["status"] == "ok"
    assert outputs_complete(
        str(out_root),
        "sub-0001",
        "01",
        "MNI152",
    )
    assert not list(out_root.glob(".initsurf-stage-*"))


def test_failed_subject_leaves_existing_output_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_root = tmp_path / "sc-initsurf"
    existing = (
        out_root
        / "sub-0001"
        / "ses-01"
        / "anat"
        / "existing.txt"
    )
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("keep\n", encoding="utf-8")

    def failing_impl(**kwargs):
        staged_file = (
            Path(kwargs["out_root"])
            / kwargs["subject_id"]
            / f"ses-{kwargs['ses']}"
            / "anat"
            / "partial.txt"
        )
        staged_file.parent.mkdir(parents=True, exist_ok=True)
        staged_file.write_text("partial\n", encoding="utf-8")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        generate_module,
        "_generate_subject_impl",
        failing_impl,
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        _generate_subject(
            subject_id="sub-0001",
            ds_key="TEST",
            preproc_root="/unused/preproc",
            seg_root="/unused/seg",
            out_root=str(out_root),
            ses="01",
            space="MNI152",
            params=dict(DEFAULT_PARAMS),
        )

    assert existing.read_text(encoding="utf-8") == "keep\n"
    assert not list(out_root.glob(".initsurf-stage-*"))
    assert not list(out_root.rglob("partial.txt"))


def test_existing_complete_subject_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_root = tmp_path / "sc-initsurf"
    _write_complete_outputs(str(out_root))

    def unexpected_impl(**kwargs):
        raise AssertionError("implementation should not run")

    monkeypatch.setattr(
        generate_module,
        "_generate_subject_impl",
        unexpected_impl,
    )

    result = _generate_subject(
        subject_id="sub-0001",
        ds_key="TEST",
        preproc_root="/unused/preproc",
        seg_root="/unused/seg",
        out_root=str(out_root),
        ses="01",
        space="MNI152",
        params=dict(DEFAULT_PARAMS),
    )

    assert result["status"] == "skipped_existing"


def test_worker_pool_recycles_after_configured_task_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakePool:
        def __init__(self, *, processes, maxtasksperchild):
            captured["processes"] = processes
            captured["maxtasksperchild"] = maxtasksperchild

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, function, jobs, chunksize):
            captured["chunksize"] = chunksize
            return iter(function(job) for job in reversed(list(jobs)))

    class FakeContext:
        Pool = FakePool

    monkeypatch.setattr(
        generate_module.mp,
        "get_context",
        lambda method: FakeContext(),
    )
    monkeypatch.setattr(
        generate_module,
        "_generate_subject_from_job",
        lambda job: {
            "status": "ok",
            "subject_id": job["subject_id"],
            "ds_key": "TEST",
            "elapsed": 0.0,
            "wm_final_level": 0.0,
            "pial_l": 0.0,
            "pial_r": 0.0,
        },
    )

    results = _run_jobs(
        [
            {"subject_id": "sub-0001"},
            {"subject_id": "sub-0002"},
        ],
        n_workers=2,
        max_tasks_per_worker=1,
    )

    assert captured == {
        "processes": 2,
        "maxtasksperchild": 1,
        "chunksize": 1,
    }
    assert {result["subject_id"] for result in results} == {
        "sub-0001",
        "sub-0002",
    }


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
