"""Regression tests for the SimCortex end-to-end pipeline orchestrator."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


# scripts/ is intentionally not a Python package, so load the production
# orchestrator directly from its file while registering it in sys.modules.
MODULE_NAME = "_simcortex_run_pipeline_contract"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_pipeline.py"

_spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    SCRIPT_PATH,
)
assert _spec is not None
assert _spec.loader is not None

rp = importlib.util.module_from_spec(_spec)
sys.modules[MODULE_NAME] = rp
_spec.loader.exec_module(rp)


def _cfg(
    tmp_path: Path,
    *,
    overwrite: bool = False,
    keep_work: bool = False,
    export_native: bool = False,
):
    return rp.PipelineConfig(
        out_root=tmp_path / "out",
        work_root=tmp_path / "work",
        project_root=tmp_path / "project",
        mni=tmp_path / "MNI152_T1_1mm.nii.gz",
        seg_ckpt=tmp_path / "seg_best_dice.pt",
        deform_ckpt=tmp_path / "deform_best_model.pth",
        device="cuda:0",
        space="MNI152",
        transform_type="Affine",
        overwrite=overwrite,
        keep_work=keep_work,
        initsurf_workers=1,
        export_native=export_native,
    )


def _subject(tmp_path: Path):
    return rp.SubjectInput(
        subject="0001",
        session="01",
        t1w_path=tmp_path / "input_T1w.nii.gz",
    )


def test_subject_and_session_normalization_contract():
    """Subject/session identifiers must retain the established BIDS prefixes."""
    assert rp.normalize_subject_id("0001") == "sub-0001"
    assert rp.normalize_subject_id("sub-0001") == "sub-0001"

    assert rp.normalize_session_id("01") == "ses-01"
    assert rp.normalize_session_id("ses-01") == "ses-01"

    assert rp.session_value_for_hydra("01") == "ses-01"
    assert rp.session_value_for_hydra("ses-01") == "ses-01"


def test_build_layout_uses_canonical_stage_directories(tmp_path):
    """The orchestrator must use the finalized version-independent derivatives."""
    cfg = _cfg(tmp_path)
    layout = rp.build_layout(
        _subject(tmp_path),
        cfg,
    )

    assert layout.subject == "sub-0001"
    assert layout.session == "ses-01"

    assert layout.preproc_root == cfg.work_root / "sc-preproc"
    assert layout.seg_root == cfg.work_root / "sc-seg"
    assert layout.initsurf_root == cfg.work_root / "sc-initsurf"
    assert layout.deform_root == cfg.work_root / "sc-deform"

    assert layout.tmp_dir == (
        cfg.work_root
        / "tmp"
        / "sub-0001"
        / "ses-01"
    )

    assert layout.log_dir == (
        cfg.work_root
        / "logs"
        / "sub-0001"
        / "ses-01"
    )

    assert layout.final_dir == (
        cfg.out_root
        / "sub-0001"
        / "ses-01"
        / "surfaces"
    )

    assert layout.split_file == (
        layout.tmp_dir / "split_one_subject.csv"
    )


def test_single_subject_split_file_contract(tmp_path):
    """The one-subject orchestration split remains a test split."""
    cfg = _cfg(tmp_path)
    layout = rp.build_layout(
        _subject(tmp_path),
        cfg,
    )

    rp.write_split_file(layout)

    assert layout.split_file.read_text() == (
        "subject,split\n"
        "sub-0001,test\n"
    )


def test_final_surface_filename_contract(tmp_path):
    """Final deformation outputs retain the established four surface names."""
    paths = rp.expected_final_surface_paths(
        tmp_path / "surfaces",
        "sub-0001",
        "ses-01",
        "MNI152",
    )

    assert list(paths) == [
        "lh_pial",
        "lh_white",
        "rh_pial",
        "rh_white",
    ]

    assert paths["lh_pial"].name == (
        "sub-0001_ses-01_space-MNI152_"
        "desc-deform_hemi-L_pial.surf.ply"
    )
    assert paths["lh_white"].name == (
        "sub-0001_ses-01_space-MNI152_"
        "desc-deform_hemi-L_white.surf.ply"
    )
    assert paths["rh_pial"].name == (
        "sub-0001_ses-01_space-MNI152_"
        "desc-deform_hemi-R_pial.surf.ply"
    )
    assert paths["rh_white"].name == (
        "sub-0001_ses-01_space-MNI152_"
        "desc-deform_hemi-R_white.surf.ply"
    )


def test_surface_affine_application_contract():
    """Surface vertices use row-homogeneous coordinates times matrix.T."""
    vertices = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )

    matrix = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, -5.0],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    actual = rp.apply_affine_to_vertices(
        vertices,
        matrix,
    )

    expected = np.asarray(
        [
            [12.0, 1.0, 13.0],
            [8.0, -5.0, 17.0],
        ],
        dtype=np.float32,
    )

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_pipeline_cli_defaults():
    """CLI defaults must remain consistent with the finalized orchestrator."""
    parser = rp.build_parser()

    args = parser.parse_args(
        [
            "single",
            "--out-root",
            "/tmp/out",
            "--project-root",
            "/tmp/project",
            "--mni",
            "/tmp/mni.nii.gz",
            "--seg-ckpt",
            "/tmp/seg.pt",
            "--deform-ckpt",
            "/tmp/deform.pth",
            "--t1w",
            "/tmp/input.nii.gz",
            "--subject",
            "0001",
        ]
    )

    assert args.mode == "single"
    assert args.work_root is None

    assert args.device == "cuda:0"
    assert args.space == "MNI152"
    assert args.transform_type == "Affine"

    assert args.overwrite is False
    assert args.keep_work is False
    assert args.initsurf_workers == 1
    assert args.export_native is False

    assert args.session == "ses-01"


def test_subprocess_environment_only_prepends_project_src(tmp_path):
    """Pipeline orchestration must not inject a new seed/thread policy."""
    cfg = _cfg(tmp_path)

    before = rp.os.environ.copy()
    old_pythonpath = before.get(
        "PYTHONPATH",
        "",
    )

    env = rp.subprocess_env(cfg)

    src = str(cfg.project_root / "src")

    expected_pythonpath = (
        src
        if not old_pythonpath
        else f"{src}:{old_pythonpath}"
    )

    assert env["PYTHONPATH"] == expected_pythonpath

    for key, value in before.items():
        if key == "PYTHONPATH":
            continue
        assert env[key] == value

    assert set(env) - set(before) <= {"PYTHONPATH"}

    source = inspect.getsource(
        rp.subprocess_env
    )

    for forbidden in [
        "CUBLAS_WORKSPACE_CONFIG",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
        "OMP_NUM_THREADS",
        "DEFAULT_PREPROC_RANDOM_SEED",
        "DEFAULT_ITK_THREADS",
    ]:
        assert forbidden not in source


def test_stage_command_contracts(
    tmp_path,
    monkeypatch,
):
    """Each stage must invoke the established production module and overrides."""
    cfg = _cfg(
        tmp_path,
        overwrite=True,
    )

    sub = _subject(tmp_path)
    layout = rp.build_layout(
        sub,
        cfg,
    )

    calls = []

    def fake_run_cmd(
        stage,
        cmd,
        log_path,
        cwd,
        passed_cfg,
    ):
        calls.append(
            {
                "stage": stage,
                "cmd": [str(x) for x in cmd],
                "log_path": log_path,
                "cwd": cwd,
                "cfg": passed_cfg,
            }
        )
        return 1.25

    monkeypatch.setattr(
        rp,
        "run_cmd",
        fake_run_cmd,
    )

    # Do not require real scientific outputs in this unit contract.
    monkeypatch.setattr(
        rp,
        "check_required_file",
        lambda *args, **kwargs: None,
    )

    # Preprocessing creates a convenience alias after execution.
    monkeypatch.setattr(
        rp.shutil,
        "copy2",
        lambda *args, **kwargs: None,
    )

    assert (
        rp.run_preprocessing_mri_only(
            sub,
            layout,
            cfg,
        )
        == 1.25
    )

    assert (
        rp.run_segmentation(
            layout,
            cfg,
        )
        == 1.25
    )

    assert (
        rp.run_initsurf(
            layout,
            cfg,
        )
        == 1.25
    )

    assert (
        rp.run_deform(
            layout,
            cfg,
        )
        == 1.25
    )

    assert [
        call["stage"]
        for call in calls
    ] == [
        "preproc",
        "segmentation",
        "initsurf",
        "deform",
    ]

    preproc = calls[0]["cmd"]
    assert preproc[1:4] == [
        "-m",
        "simcortex.preproc.mri_to_mni_inference",
        "--t1w",
    ]
    assert "--transform-type" in preproc
    assert "Affine" in preproc
    assert "--derivative-name" in preproc
    assert "sc-preproc" in preproc

    seg = calls[1]["cmd"]
    assert seg[1:3] == [
        "-m",
        "simcortex.seg.inference",
    ]
    assert f"dataset.path={layout.preproc_root}" in seg
    assert f"outputs.out_root={layout.seg_root}" in seg
    assert f"model.ckpt_path={cfg.seg_ckpt}" in seg
    assert "trainer.batch_size=1" in seg
    assert "trainer.num_workers=0" in seg

    init = calls[2]["cmd"]
    assert init[1:3] == [
        "-m",
        "simcortex.initsurf.generate",
    ]
    assert f"dataset.path={layout.preproc_root}" in init
    assert f"dataset.seg_root={layout.seg_root}" in init
    assert f"outputs.out_root={layout.initsurf_root}" in init
    assert "n_workers=1" in init

    deform = calls[3]["cmd"]
    assert deform[1:3] == [
        "-m",
        "simcortex.deform.inference",
    ]
    assert f"dataset.path={layout.preproc_root}" in deform
    assert (
        f"dataset.initsurf_root={layout.initsurf_root}"
        in deform
    )
    assert f"outputs.out_root={layout.deform_root}" in deform
    assert f"model.ckpt_path={cfg.deform_ckpt}" in deform
    assert "inference.batch_size=1" in deform
    assert "inference.num_workers=0" in deform
    assert "inference.overwrite=true" in deform


@pytest.mark.parametrize(
    ("export_native", "expected"),
    [
        (
            False,
            [
                "preproc",
                "segmentation",
                "initsurf",
                "deform",
                "collect",
            ],
        ),
        (
            True,
            [
                "preproc",
                "segmentation",
                "initsurf",
                "deform",
                "collect",
                "export_native",
            ],
        ),
    ],
)
def test_run_one_subject_stage_order(
    tmp_path,
    monkeypatch,
    export_native,
    expected,
):
    """A subject must traverse the established pipeline stages in order."""
    cfg = _cfg(
        tmp_path,
        export_native=export_native,
    )
    sub = _subject(tmp_path)

    calls = []

    def stage(name, value):
        def _run(*args, **kwargs):
            calls.append(name)
            return value

        return _run

    monkeypatch.setattr(
        rp,
        "run_preprocessing_mri_only",
        stage("preproc", 1.0),
    )
    monkeypatch.setattr(
        rp,
        "run_segmentation",
        stage("segmentation", 2.0),
    )
    monkeypatch.setattr(
        rp,
        "run_initsurf",
        stage("initsurf", 3.0),
    )
    monkeypatch.setattr(
        rp,
        "run_deform",
        stage("deform", 4.0),
    )
    monkeypatch.setattr(
        rp,
        "collect_final_surfaces",
        stage("collect", 5.0),
    )
    monkeypatch.setattr(
        rp,
        "export_native_surfaces",
        stage("export_native", 6.0),
    )

    rp.run_one_subject(
        sub,
        cfg,
        index=1,
        total=1,
    )

    assert calls == expected

    summary = cfg.out_root / "pipeline_summary.tsv"
    assert summary.exists()

    with summary.open(newline="") as f:
        rows = list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )

    assert len(rows) == 1
    row = rows[0]

    assert row["subject"] == "sub-0001"
    assert row["session"] == "ses-01"
    assert row["status"] == "OK"

    assert float(row["preproc_sec"]) == 1.0
    assert float(row["segmentation_sec"]) == 2.0
    assert float(row["initsurf_sec"]) == 3.0
    assert float(row["deform_sec"]) == 4.0
    assert float(row["collect_sec"]) == 5.0

    if export_native:
        assert float(row["export_native_sec"]) == 6.0
        assert row["native_lh_pial"] != ""
    else:
        assert float(row["export_native_sec"]) == 0.0
        assert row["native_lh_pial"] == ""


def test_pipeline_failure_stops_later_stages_and_records_failure(
    tmp_path,
    monkeypatch,
):
    """A failed stage must stop execution and record a FAILED summary row."""
    cfg = _cfg(tmp_path)
    sub = _subject(tmp_path)

    calls = []

    def preproc(*args, **kwargs):
        calls.append("preproc")
        return 1.0

    def segmentation(*args, **kwargs):
        calls.append("segmentation")
        raise RuntimeError("synthetic segmentation failure")

    def should_not_run(*args, **kwargs):
        calls.append("unexpected")
        return 0.0

    monkeypatch.setattr(
        rp,
        "run_preprocessing_mri_only",
        preproc,
    )
    monkeypatch.setattr(
        rp,
        "run_segmentation",
        segmentation,
    )
    monkeypatch.setattr(
        rp,
        "run_initsurf",
        should_not_run,
    )
    monkeypatch.setattr(
        rp,
        "run_deform",
        should_not_run,
    )
    monkeypatch.setattr(
        rp,
        "collect_final_surfaces",
        should_not_run,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic segmentation failure",
    ):
        rp.run_one_subject(
            sub,
            cfg,
            index=1,
            total=1,
        )

    assert calls == [
        "preproc",
        "segmentation",
    ]

    summary = cfg.out_root / "pipeline_summary.tsv"

    with summary.open(newline="") as f:
        rows = list(
            csv.DictReader(
                f,
                delimiter="\t",
            )
        )

    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"
    assert (
        "synthetic segmentation failure"
        in rows[0]["error"]
    )
