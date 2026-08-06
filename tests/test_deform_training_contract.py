from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from simcortex.deform.data.dataloader import CSRDeformDataset
from simcortex.deform.models.surfdeform import DualMUNetV2, SurfDeform
from simcortex.deform.train import (
    _atomic_torch_save,
    _validate_output_root,
    collision_coverage_status,
    validate_deform_training_config,
    validate_split_dataframe,
    validation_coverage_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "src/simcortex/configs/deform/train.yaml"
TRAIN_PATH = REPO_ROOT / "src/simcortex/deform/train.py"
SURFDEFORM_PATH = REPO_ROOT / "src/simcortex/deform/models/surfdeform.py"

EXPECTED_SURFACES = ["lh_pial", "lh_white", "rh_pial", "rh_white"]


def test_public_yaml_is_portable_and_preserves_exp33_contract() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    cfg = OmegaConf.load(CONFIG_PATH)

    assert "/project/hippocampus" not in text
    assert cfg.dataset.split_file == "/path/to/datasets/splits/dataset_split.csv"
    assert bool(cfg.dataset.strict_missing) is True
    assert bool(cfg.outputs.allow_existing) is False

    assert list(cfg.dataset.surface_name) == EXPECTED_SURFACES
    assert list(cfg.model.inshape) == [184, 224, 184]
    assert list(cfg.model.c_hid) == [8, 16, 32, 64, 128, 128]
    assert int(cfg.model.c_in) == 2
    assert int(cfg.model.geom_depth) == 6
    assert int(cfg.model.n_steps) == 8

    assert float(cfg.objective.signed_nested_weight) == 8.0
    assert float(cfg.objective.signed_margin_mm) == 0.55
    assert int(cfg.objective.signed_points) == 50000
    assert float(cfg.checkpoint.alpha_wp) == 0.03
    assert float(cfg.checkpoint.alpha_lr) == 0.015
    assert bool(cfg.checkpoint.require_collision_for_best) is True

    validate_deform_training_config(cfg)


def test_internal_model_defaults_match_published_depth_six() -> None:
    dual_default = inspect.signature(DualMUNetV2).parameters["geom_depth"].default
    model_default = inspect.signature(SurfDeform).parameters["geom_depth"].default
    assert dual_default == 6
    assert model_default == 6

    model = SurfDeform(
        C_in=2,
        C_hid=(8, 16, 32, 64, 128, 128),
        inshape=(184, 224, 184),
        sigma=1.0,
        geom_ratio=0.5,
        geom_depth=6,
        gn_groups=8,
        gate_init=-3.0,
        dropout=0.1,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable == 6_037_486


def test_model_contract_rejects_wrong_channels_and_negative_steps() -> None:
    model = SurfDeform(
        C_in=2,
        C_hid=(4, 4, 8, 8, 8, 8),
        inshape=(16, 16, 16),
        geom_depth=6,
        gn_groups=4,
        dropout=0.0,
    ).eval()
    vertices = torch.zeros((1, 3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="channels"):
        model(vertices, torch.zeros((1, 3, 16, 16, 16)), 1)

    with pytest.raises(ValueError, match="n_steps"):
        model(vertices, torch.zeros((1, 2, 16, 16, 16)), -1)


def test_collision_aware_selection_requires_matching_intervals() -> None:
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.trainer.collision_interval = int(cfg.trainer.validation_interval) + 1
    with pytest.raises(ValueError, match="collision_interval"):
        validate_deform_training_config(cfg)


def test_training_dataset_fails_on_any_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing required inputs"):
        CSRDeformDataset(
            preproc_root=str(tmp_path / "sc-preproc"),
            initsurf_root=str(tmp_path / "sc-initsurf"),
            subjects=["sub-0001"],
            session_label="01",
            space="MNI152",
            surface_names=EXPECTED_SURFACES,
            inshape_dhw=(184, 224, 184),
        )


def test_duplicate_multi_dataset_split_rows_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["HCP_YA", "HCP_YA"],
            "subject": ["100307", "sub-100307"],
            "split": ["train", "validation"],
        }
    )
    with pytest.raises(ValueError, match="duplicate subject rows"):
        validate_split_dataframe(frame, mode="multi")


def test_validation_coverage_requires_every_subject_surface() -> None:
    complete = {
        surface: {"chamfer_sq": 1.0, "count": 2.0}
        for surface in EXPECTED_SURFACES
    }
    ok, message = validation_coverage_status(
        8.0,
        complete,
        expected_subjects=2,
        surface_names=EXPECTED_SURFACES,
    )
    assert ok is True
    assert message == ""

    incomplete = {surface: dict(values) for surface, values in complete.items()}
    incomplete["rh_pial"]["count"] = 1.0
    ok, message = validation_coverage_status(
        7.0,
        incomplete,
        expected_subjects=2,
        surface_names=EXPECTED_SURFACES,
    )
    assert ok is False
    assert "rh_pial=1/2" in message


def test_collision_coverage_rejects_partial_or_unknown_pairs() -> None:
    ok, message = collision_coverage_status(
        2.0, 2.0, 2.0, 0.0, 0.0, 0.0, expected_subjects=2
    )
    assert ok is True
    assert message == ""

    ok, message = collision_coverage_status(
        2.0, 1.0, 2.0, 0.0, 1.0, 0.0, expected_subjects=2
    )
    assert ok is False
    assert "RH white-pial" in message
    assert "unknown=1" in message


def test_atomic_checkpoint_save_replaces_complete_file(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoints" / "model.pth"
    _atomic_torch_save({"value": torch.tensor([1])}, str(destination))
    _atomic_torch_save({"value": torch.tensor([1, 2, 3])}, str(destination))
    loaded = torch.load(destination, map_location="cpu")
    assert torch.equal(loaded["value"], torch.tensor([1, 2, 3]))
    assert not list(destination.parent.glob(".*.tmp"))


def test_fresh_run_refuses_existing_training_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Refusing to start a fresh run"):
        _validate_output_root(
            str(output_root),
            resume_from="",
            allow_existing=False,
        )

    _validate_output_root(
        str(output_root),
        resume_from="/path/to/deform_last_full.pth",
        allow_existing=False,
    )


def test_source_has_clean_ascii_logs_and_no_deprecated_scheduler_verbose() -> None:
    train_source = TRAIN_PATH.read_text(encoding="utf-8")
    model_source = SURFDEFORM_PATH.read_text(encoding="utf-8")

    for token in ("Whiteâ", "Pialâ", "ðŸ", "🌟", "🛑"):
        assert token not in train_source
    assert "verbose=(rank == 0)" not in train_source
    assert "geom_depth=4" not in model_source
    assert "geom_depth: int = 4" not in model_source
