"""Regression tests for published SimCortex release checkpoint identities."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"
DEFORM_INFERENCE_CONFIG_PATH = (
    REPO_ROOT / "src/simcortex/configs/deform/inference.yaml"
)

RELEASE_CHECKPOINTS = (
    (
        "seg_best_dice.pt",
        "2cea6dc1606958dbdb2aea50748b73ff73e604b0c367b3b1588cef7a112c8e06",
    ),
    (
        "deform_best_rmse.pth",
        "3fb84c917953e479e72d59291837bd6b08f40fb197534046b95dc5a452e86b95",
    ),
)


@pytest.mark.parametrize(
    ("filename", "sha256"),
    RELEASE_CHECKPOINTS,
)
def test_release_checkpoint_identity_is_documented_once(
    filename: str,
    sha256: str,
) -> None:
    """Each validated release filename and SHA256 must appear exactly once."""
    readme = README_PATH.read_text(encoding="utf-8")

    assert readme.count(f"Filename: {filename}") == 1
    assert readme.count(sha256) == 1

def test_deformation_inference_config_uses_release_checkpoint() -> None:
    """The shipped inference config must point to the validated release weights."""
    config_text = DEFORM_INFERENCE_CONFIG_PATH.read_text(
        encoding="utf-8"
    )

    assert (
        config_text.count(
            "ckpt_path: /path/to/deform_best_rmse.pth"
        )
        == 1
    )
    assert "deform_best_model.pth" not in config_text
