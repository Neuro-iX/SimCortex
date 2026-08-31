"""Regression tests for SimCortex package metadata and shipped resources."""

from __future__ import annotations

import hashlib
from importlib import metadata, resources
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = PROJECT_ROOT / "src" / "simcortex"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
MANIFEST = PROJECT_ROOT / "MANIFEST.in"

RESOURCE_FILES = [
    "configs/seg/train.yaml",
    "configs/seg/inference.yaml",
    "configs/seg/eval.yaml",
    "configs/initsurf/generate.yaml",
    "configs/deform/train.yaml",
    "configs/deform/inference.yaml",
    "configs/deform/eval.yaml",
    "utils/critical186LUT.raw.gz",
]

EXPECTED_LUT_SHA256 = (
    "cbf1e339d78a0a2c673c939ee2c8a30e"
    "e3a35bfd9603d07ed2d48864bb0e88be"
)


def test_installed_distribution_version():
    """Installed package metadata must match the finalized release version."""
    assert metadata.version("simcortex") == "2.0.0"


def test_required_resources_are_available_via_importlib():
    """All runtime configuration and topology resources must be discoverable."""
    package_root = resources.files("simcortex")

    missing = [
        relative
        for relative in RESOURCE_FILES
        if not package_root.joinpath(relative).is_file()
    ]

    assert missing == []


def test_packaged_resources_match_source_tree_exactly():
    """Resource loading must expose the exact bytes present in the source tree."""
    package_root = resources.files("simcortex")

    for relative in RESOURCE_FILES:
        installed_bytes = package_root.joinpath(relative).read_bytes()
        source_bytes = (SOURCE_PACKAGE / relative).read_bytes()

        assert installed_bytes == source_bytes, relative


def test_topology_lut_checksum_is_locked():
    """The scientific topology LUT must never change silently."""
    lut = resources.files("simcortex").joinpath(
        "utils/critical186LUT.raw.gz"
    )

    digest = hashlib.sha256(
        lut.read_bytes()
    ).hexdigest()

    assert digest == EXPECTED_LUT_SHA256


def test_pyproject_declares_required_package_data():
    """Setuptools must explicitly ship configs and the topology LUT."""
    text = PYPROJECT.read_text(
        encoding="utf-8"
    )

    assert '[project]' in text
    assert 'name = "simcortex"' in text
    assert 'version = "2.0.0"' in text

    assert "include-package-data = true" in text

    assert '"configs/**/*.yaml"' in text
    assert '"utils/critical186LUT.raw.gz"' in text

    assert (
        'simcortex = "simcortex.cli.main:app"'
        in text
    )


def test_manifest_includes_required_docker_distribution_files():
    """Source distributions must retain the Docker release scaffolding."""
    lines = {
        line.strip()
        for line in MANIFEST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    }

    expected = {
        "include docker/Dockerfile",
        "include docker/README.md",
        "include docker/entrypoint.sh",
        "include docker/requirements.freeze.txt",
        "include docker/requirements.runtime.txt",
    }

    assert expected <= lines


def test_packaging_rules_do_not_include_generated_artifacts():
    """Packaging declarations must not intentionally ship local build/cache files."""
    text = (
        PYPROJECT.read_text(encoding="utf-8")
        + "\n"
        + MANIFEST.read_text(encoding="utf-8")
    ).lower()

    forbidden = [
        "__pycache__",
        ".pytest_cache",
        ".egg-info",
        "src/simcortex.egg-info",
    ]

    for value in forbidden:
        assert value not in text
