"""Regression tests for SimCortex software-version provenance metadata."""

from __future__ import annotations

import json

import simcortex
import simcortex.initsurf.generate as initsurf_generate
import simcortex.preproc.fs_to_mni as fs_to_mni
import simcortex.seg.inference as seg_inference


def _read_dataset_description(root):
    path = root / "dataset_description.json"
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


def test_preproc_metadata_uses_package_version(tmp_path):
    """FreeSurfer preprocessing provenance must report the package version."""
    fs_to_mni.write_dataset_description(tmp_path)

    payload = _read_dataset_description(tmp_path)

    assert payload["GeneratedBy"][0]["Version"] == simcortex.__version__


def test_segmentation_metadata_uses_package_version(tmp_path):
    """Segmentation provenance must report the package version."""
    seg_inference._write_dataset_description(
        tmp_path,
        name="SimCortex Segmentation",
        version=seg_inference.__version__,
        overwrite=False,
    )

    payload = _read_dataset_description(tmp_path)

    assert seg_inference.__version__ == simcortex.__version__
    assert payload["GeneratedBy"][0]["Version"] == simcortex.__version__


def test_initsurf_metadata_uses_package_version(tmp_path):
    """InitSurf provenance must report the package version."""
    initsurf_generate.write_dataset_description(
        str(tmp_path),
        name="sc-initsurf",
    )

    payload = _read_dataset_description(tmp_path)

    assert initsurf_generate.__version__ == simcortex.__version__
    assert payload["GeneratedBy"][0]["Version"] == simcortex.__version__
