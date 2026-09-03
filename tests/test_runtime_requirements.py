"""Regression tests for the validated SimCortex runtime requirements."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REQUIREMENTS = (
    PROJECT_ROOT / "docker" / "requirements.runtime.txt"
)

EXPECTED_RUNTIME_REQUIREMENTS = [
    "monai==1.3.2",
    "nibabel==5.2.0",
    "numpy==1.24.3",
    "scipy==1.10.1",
    "scikit-image==0.21.0",
    "trimesh==4.1.3",
    "hydra-core==1.3.2",
    "omegaconf==2.3.0",
    "pandas==2.0.3",
    "openpyxl==3.1.5",
    "typer==0.23.0",
    "tensorboard==2.20.0",
    "tqdm==4.67.1",
    "python-fcl==0.7.0.10",
]


def _runtime_requirements() -> list[str]:
    return [
        line.strip()
        for line in RUNTIME_REQUIREMENTS.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_runtime_requirements_are_exactly_pinned():
    """Every validated Docker runtime dependency must use an exact pin."""
    requirements = _runtime_requirements()

    invalid = [
        requirement
        for requirement in requirements
        if re.fullmatch(
            r"[A-Za-z0-9_.-]+==[^=<>!~\s]+",
            requirement,
        )
        is None
    ]

    assert invalid == []


def test_runtime_requirements_match_validated_stack():
    """The runtime file must preserve the validated SimCortex stack."""
    assert _runtime_requirements() == EXPECTED_RUNTIME_REQUIREMENTS
