"""Regression tests for the SimCortex multi-dataset inference runner."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# scripts/ is intentionally not a Python package, so load the production
# orchestrator directly from its file while registering it in sys.modules.
MODULE_NAME = "_simcortex_multi_dataset_inference_contract"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_multi_dataset_inference.py"

_spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    SCRIPT_PATH,
)
assert _spec is not None
assert _spec.loader is not None

mdi = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = mdi
_spec.loader.exec_module(mdi)


def _runtime_tree(
    tmp_path: Path,
    dataset_names: tuple[str, ...] = (
        "dataset_a",
        "dataset_b",
        "dataset_c",
    ),
) -> dict[str, object]:
    project_root = tmp_path / "project"
    scripts_dir = project_root / "scripts"
    package_dir = project_root / "src" / "simcortex"

    scripts_dir.mkdir(parents=True)
    package_dir.mkdir(parents=True)

    pipeline_script = scripts_dir / "run_pipeline.py"
    pipeline_script.write_text(
        "#!/usr/bin/env python3\n",
        encoding="utf-8",
    )

    mni = tmp_path / "MNI152_T1_1mm.nii.gz"
    seg_ckpt = tmp_path / "seg_best_dice.pt"
    deform_ckpt = tmp_path / "deform_best_rmse.pth"

    mni.touch()
    seg_ckpt.touch()
    deform_ckpt.touch()

    dataset_roots = {}
    for name in dataset_names:
        root = tmp_path / "data" / name
        root.mkdir(parents=True)
        dataset_roots[name] = root

    return {
        "project_root": project_root,
        "pipeline_script": pipeline_script,
        "mni": mni,
        "seg_ckpt": seg_ckpt,
        "deform_ckpt": deform_ckpt,
        "datasets": dataset_roots,
        "out_root": tmp_path / "predictions",
        "work_root": tmp_path / "work",
        "log_root": tmp_path / "logs",
    }


def _base_cli(tree: dict[str, object]) -> list[str]:
    return [
        "--project-root",
        str(tree["project_root"]),
        "--mni",
        str(tree["mni"]),
        "--seg-ckpt",
        str(tree["seg_ckpt"]),
        "--deform-ckpt",
        str(tree["deform_ckpt"]),
        "--out-root",
        str(tree["out_root"]),
        "--work-root",
        str(tree["work_root"]),
        "--log-root",
        str(tree["log_root"]),
    ]


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )


def test_dataset_argument_parsing_contract(tmp_path):
    """Repeated NAME=PATH inputs retain explicit dataset identities."""
    root = tmp_path / "dataset"

    spec = mdi.parse_dataset_argument(
        f"dataset-01={root}"
    )

    assert spec.name == "dataset-01"
    assert spec.bids_root == root.resolve()

    for invalid in [
        "",
        "dataset-only",
        f"={root}",
        f"bad name={root}",
        f"bad/name={root}",
    ]:
        with pytest.raises(ValueError):
            mdi.parse_dataset_argument(invalid)


def test_manifest_resolves_relative_dataset_roots(
    tmp_path,
):
    """Relative manifest paths are resolved from the manifest directory."""
    manifest_dir = tmp_path / "manifest"
    data_dir = tmp_path / "data"

    manifest_dir.mkdir()
    (data_dir / "dataset_a").mkdir(
        parents=True,
    )
    (data_dir / "dataset_b").mkdir(
        parents=True,
    )

    manifest = manifest_dir / "datasets.csv"
    manifest.write_text(
        "dataset,bids_root\n"
        "dataset_a,../data/dataset_a\n"
        "dataset_b,../data/dataset_b\n",
        encoding="utf-8",
    )

    specs = mdi.load_dataset_manifest(
        manifest
    )

    assert [spec.name for spec in specs] == [
        "dataset_a",
        "dataset_b",
    ]

    assert specs[0].bids_root == (
        data_dir / "dataset_a"
    ).resolve()

    assert specs[1].bids_root == (
        data_dir / "dataset_b"
    ).resolve()


def test_duplicate_dataset_names_are_rejected(
    tmp_path,
):
    """A dataset name must identify exactly one output/work/log namespace."""
    args = argparse.Namespace(
        dataset_manifest=None,
        dataset=[
            f"dataset_a={tmp_path / 'first'}",
            f"dataset_a={tmp_path / 'second'}",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate dataset name",
    ):
        mdi.collect_datasets(args)


def test_pipeline_command_contract(tmp_path):
    """The wrapper must forward the established run_pipeline.py BIDS contract."""
    tree = _runtime_tree(
        tmp_path,
        dataset_names=("dataset_a",),
    )

    parser = mdi.build_parser()

    args = parser.parse_args(
        _base_cli(tree)
        + [
            "--dataset",
            (
                "dataset_a="
                f"{tree['datasets']['dataset_a']}"
            ),
            "--device",
            "cuda:0",
            "--cuda-visible-devices",
            "1",
            "--space",
            "MNI152",
            "--transform-type",
            "Rigid",
            "--initsurf-workers",
            "3",
            "--overwrite",
            "--keep-work",
            "--export-native",
            "--participant-label",
            "sub-0001",
            "--participant-label",
            "sub-0002",
            "--session",
            "ses-01",
            "--session",
            "ses-02",
        ]
    )

    spec = mdi.DatasetSpec(
        name="dataset_a",
        bids_root=tree["datasets"][
            "dataset_a"
        ],
    )

    out_root = tree["out_root"] / "dataset_a"
    work_root = (
        tree["work_root"]
        / "dataset_a"
    )

    command = mdi.build_pipeline_command(
        spec=spec,
        args=args,
        project_root=tree[
            "project_root"
        ],
        mni=tree["mni"],
        seg_ckpt=tree["seg_ckpt"],
        deform_ckpt=tree[
            "deform_ckpt"
        ],
        out_root=out_root,
        work_root=work_root,
    )

    expected = [
        sys.executable,
        str(
            tree["project_root"]
            / "scripts"
            / "run_pipeline.py"
        ),
        "bids",
        "--bids-root",
        str(
            tree["datasets"][
                "dataset_a"
            ]
        ),
        "--out-root",
        str(out_root),
        "--work-root",
        str(work_root),
        "--project-root",
        str(tree["project_root"]),
        "--mni",
        str(tree["mni"]),
        "--seg-ckpt",
        str(tree["seg_ckpt"]),
        "--deform-ckpt",
        str(tree["deform_ckpt"]),
        "--device",
        "cuda:0",
        "--space",
        "MNI152",
        "--transform-type",
        "Rigid",
        "--initsurf-workers",
        "3",
        "--overwrite",
        "--keep-work",
        "--export-native",
        "--participant-label",
        "sub-0001",
        "sub-0002",
        "--session",
        "ses-01",
        "ses-02",
    ]

    assert command == expected
    assert "--cuda-visible-devices" not in command


def test_subprocess_environment_contract(
    tmp_path,
    monkeypatch,
):
    """The wrapper only adjusts PYTHONPATH and optional CUDA visibility."""
    project_root = tmp_path / "project"

    monkeypatch.setenv(
        "PYTHONPATH",
        "/existing/pythonpath",
    )
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "9",
    )
    monkeypatch.setenv(
        "SIMCORTEX_TEST_SENTINEL",
        "preserved",
    )

    forbidden = [
        "CUBLAS_WORKSPACE_CONFIG",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "OMP_NUM_THREADS",
    ]

    for key in forbidden:
        monkeypatch.delenv(
            key,
            raising=False,
        )

    before = os.environ.copy()

    env = mdi.prepare_environment(
        project_root=project_root,
        cuda_visible_devices="1",
    )

    expected_pythonpath = (
        f"{project_root / 'src'}:"
        "/existing/pythonpath"
    )

    assert (
        env["PYTHONPATH"]
        == expected_pythonpath
    )
    assert (
        env["CUDA_VISIBLE_DEVICES"]
        == "1"
    )
    assert (
        env["SIMCORTEX_TEST_SENTINEL"]
        == "preserved"
    )

    for key, value in before.items():
        if key in {
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
        }:
            continue
        assert env[key] == value

    for key in forbidden:
        assert key not in env


def test_preflight_reports_missing_required_paths(
    tmp_path,
):
    """All datasets and model/input paths must exist before inference starts."""
    tree = _runtime_tree(
        tmp_path,
        dataset_names=("dataset_a",),
    )

    spec = mdi.DatasetSpec(
        name="dataset_a",
        bids_root=tree["datasets"][
            "dataset_a"
        ],
    )

    errors = mdi.check_required_paths(
        datasets=[spec],
        project_root=tree[
            "project_root"
        ],
        mni=tree["mni"],
        seg_ckpt=tree["seg_ckpt"],
        deform_ckpt=tree[
            "deform_ckpt"
        ],
    )

    assert errors == []

    tree["mni"].unlink()
    tree["datasets"][
        "dataset_a"
    ].rmdir()

    errors = mdi.check_required_paths(
        datasets=[spec],
        project_root=tree[
            "project_root"
        ],
        mni=tree["mni"],
        seg_ckpt=tree["seg_ckpt"],
        deform_ckpt=tree[
            "deform_ckpt"
        ],
    )

    assert any(
        "MNI template not found"
        in error
        for error in errors
    )
    assert any(
        "Dataset root not found "
        "[dataset_a]"
        in error
        for error in errors
    )


def test_dry_run_writes_dataset_logs_and_summary(
    tmp_path,
):
    """Dry-run mode validates orchestration without launching inference."""
    tree = _runtime_tree(
        tmp_path,
        dataset_names=(
            "dataset_a",
            "dataset_b",
        ),
    )

    argv = (
        _base_cli(tree)
        + [
            "--dataset",
            (
                "dataset_a="
                f"{tree['datasets']['dataset_a']}"
            ),
            "--dataset",
            (
                "dataset_b="
                f"{tree['datasets']['dataset_b']}"
            ),
            "--cuda-visible-devices",
            "1",
            "--device",
            "cuda:0",
            "--overwrite",
            "--export-native",
            "--dry-run",
        ]
    )

    returncode = mdi.main(argv)

    assert returncode == 0

    summary_path = (
        tree["out_root"]
        / "multi_dataset_summary.tsv"
    )
    rows = _read_summary(
        summary_path
    )

    assert [
        row["dataset"]
        for row in rows
    ] == [
        "dataset_a",
        "dataset_b",
    ]

    assert [
        row["status"]
        for row in rows
    ] == [
        "DRY_RUN",
        "DRY_RUN",
    ]

    assert [
        row["returncode"]
        for row in rows
    ] == [
        "0",
        "0",
    ]

    for name in [
        "dataset_a",
        "dataset_b",
    ]:
        log_path = (
            tree["log_root"]
            / f"{name}.log"
        )

        text = log_path.read_text(
            encoding="utf-8"
        )

        assert "COMMAND:" in text
        assert (
            "CUDA_VISIBLE_DEVICES=1"
            in text
        )
        assert "--overwrite" in text
        assert "--export-native" in text


def test_failed_dataset_does_not_stop_later_datasets(
    tmp_path,
    monkeypatch,
):
    """One dataset failure is recorded while subsequent datasets still run."""
    tree = _runtime_tree(
        tmp_path,
        dataset_names=(
            "dataset_a",
            "dataset_b",
            "dataset_c",
        ),
    )

    calls = []

    def fake_run(
        cmd,
        *,
        cwd,
        env,
        stdout,
        stderr,
        text,
    ):
        command = [
            str(value)
            for value in cmd
        ]

        bids_index = (
            command.index(
                "--bids-root"
            )
            + 1
        )
        dataset = Path(
            command[bids_index]
        ).name

        calls.append(
            {
                "dataset": dataset,
                "command": command,
                "cwd": cwd,
                "cuda": env.get(
                    "CUDA_VISIBLE_DEVICES"
                ),
                "stderr": stderr,
                "text": text,
            }
        )

        stdout.write(
            f"FAKE_PIPELINE_DATASET="
            f"{dataset}\n"
        )

        if dataset == "dataset_b":
            stdout.write(
                "INTENTIONAL_TEST_FAILURE\n"
            )
            stdout.flush()
            return SimpleNamespace(
                returncode=7
            )

        stdout.write(
            "INTENTIONAL_TEST_SUCCESS\n"
        )
        stdout.flush()
        return SimpleNamespace(
            returncode=0
        )

    monkeypatch.setattr(
        mdi.subprocess,
        "run",
        fake_run,
    )

    argv = (
        _base_cli(tree)
        + [
            "--dataset",
            (
                "dataset_a="
                f"{tree['datasets']['dataset_a']}"
            ),
            "--dataset",
            (
                "dataset_b="
                f"{tree['datasets']['dataset_b']}"
            ),
            "--dataset",
            (
                "dataset_c="
                f"{tree['datasets']['dataset_c']}"
            ),
            "--cuda-visible-devices",
            "1",
            "--device",
            "cuda:0",
            "--initsurf-workers",
            "2",
            "--overwrite",
            "--keep-work",
            "--export-native",
            "--participant-label",
            "sub-test01",
            "--session",
            "ses-01",
        ]
    )

    returncode = mdi.main(argv)

    assert returncode == 1

    assert [
        call["dataset"]
        for call in calls
    ] == [
        "dataset_a",
        "dataset_b",
        "dataset_c",
    ]

    assert all(
        call["cuda"] == "1"
        for call in calls
    )

    assert all(
        call["cwd"]
        == str(
            tree[
                "project_root"
            ]
        )
        for call in calls
    )

    summary_path = (
        tree["out_root"]
        / "multi_dataset_summary.tsv"
    )
    rows = _read_summary(
        summary_path
    )
    by_name = {
        row["dataset"]: row
        for row in rows
    }

    assert (
        by_name["dataset_a"][
            "status"
        ]
        == "OK"
    )
    assert (
        by_name["dataset_a"][
            "returncode"
        ]
        == "0"
    )

    assert (
        by_name["dataset_b"][
            "status"
        ]
        == "FAILED"
    )
    assert (
        by_name["dataset_b"][
            "returncode"
        ]
        == "7"
    )
    assert (
        by_name["dataset_b"][
            "error"
        ]
        == (
            "run_pipeline.py exited "
            "with code 7"
        )
    )

    assert (
        by_name["dataset_c"][
            "status"
        ]
        == "OK"
    )
    assert (
        by_name["dataset_c"][
            "returncode"
        ]
        == "0"
    )

    failed_log = (
        tree["log_root"]
        / "dataset_b.log"
    ).read_text(
        encoding="utf-8"
    )
    final_log = (
        tree["log_root"]
        / "dataset_c.log"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "INTENTIONAL_TEST_FAILURE"
        in failed_log
    )
    assert (
        "INTENTIONAL_TEST_SUCCESS"
        in final_log
    )

    command = calls[0]["command"]

    assert "--overwrite" in command
    assert "--keep-work" in command
    assert "--export-native" in command

    participant_index = (
        command.index(
            "--participant-label"
        )
    )
    assert (
        command[
            participant_index
            + 1
        ]
        == "sub-test01"
    )

    session_index = command.index(
        "--session"
    )
    assert (
        command[
            session_index
            + 1
        ]
        == "ses-01"
    )


def test_no_dataset_is_rejected():
    """At least one dataset source must be supplied."""
    args = argparse.Namespace(
        dataset_manifest=None,
        dataset=[],
    )

    with pytest.raises(
        ValueError,
        match="No datasets provided",
    ):
        mdi.collect_datasets(args)
