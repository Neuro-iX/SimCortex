"""Regression tests for the canonical MNI152 geometry contract."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import simcortex.preproc.fs_to_mni as fs_to_mni
import simcortex.preproc.mri_to_mni_inference as mri_to_mni
from simcortex.preproc.mni_geometry import (
    MNI152_TEMPLATE_AFFINE,
    MNI152_TEMPLATE_SHAPE,
    validate_mni152_template,
)


def _write_nifti(
    path: Path,
    *,
    shape=MNI152_TEMPLATE_SHAPE,
    affine=MNI152_TEMPLATE_AFFINE,
) -> None:
    nib.save(
        nib.Nifti1Image(
            np.zeros(shape, dtype=np.uint8),
            np.asarray(affine, dtype=np.float64),
        ),
        path,
    )


def test_canonical_mni152_geometry_is_accepted(tmp_path):
    path = tmp_path / "mni.nii.gz"
    _write_nifti(path)

    validate_mni152_template(path)


def test_wrong_mni152_shape_is_rejected(tmp_path):
    path = tmp_path / "wrong_shape.nii.gz"
    _write_nifti(
        path,
        shape=(10, 10, 10),
    )

    with pytest.raises(
        ValueError,
        match="MNI152 template shape mismatch",
    ):
        validate_mni152_template(path)


def test_wrong_mni152_affine_is_rejected(tmp_path):
    path = tmp_path / "wrong_affine.nii.gz"

    affine = MNI152_TEMPLATE_AFFINE.copy()
    affine[0, 3] += 1.0

    _write_nifti(
        path,
        affine=affine,
    )

    with pytest.raises(
        ValueError,
        match="MNI152 template affine does not match",
    ):
        validate_mni152_template(path)


def test_mri_preprocessing_rejects_invalid_mni_before_antspy(
    tmp_path,
    monkeypatch,
):
    wrong_mni = tmp_path / "wrong_mni.nii.gz"
    _write_nifti(
        wrong_mni,
        shape=(10, 10, 10),
    )

    def forbidden_antspy():
        raise AssertionError(
            "ANTsPy must not run before MNI geometry validation"
        )

    monkeypatch.setattr(
        mri_to_mni,
        "_require_antspy",
        forbidden_antspy,
    )

    with pytest.raises(
        ValueError,
        match="MNI152 template shape mismatch",
    ):
        mri_to_mni.preprocess_one_t1w_to_mni(
            t1w_path=tmp_path / "input.nii.gz",
            subject="sub-0001",
            session="ses-01",
            out_root=tmp_path / "out",
            mni_path=wrong_mni,
            overwrite=False,
        )


def test_mri_preprocessing_existing_outputs_preserve_skip_semantics(
    tmp_path,
    monkeypatch,
):
    out_root = tmp_path / "out"
    subject = "sub-0001"
    session = "ses-01"

    wrong_mni = tmp_path / "wrong_mni.nii.gz"
    _write_nifti(
        wrong_mni,
        shape=(10, 10, 10),
    )

    (
        out_t1,
        compat_t1,
        native_to_mni,
        mni_to_native,
    ) = mri_to_mni.expected_output_paths(
        out_root=out_root,
        derivative_name=mri_to_mni.PREPROC_DERIVATIVE_NAME,
        subject=subject,
        session=session,
    )

    for path in (
        out_t1,
        compat_t1,
        native_to_mni,
        mni_to_native,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(b"existing")

    def forbidden_validator(path):
        raise AssertionError(
            "MNI validation must not run for skipped_existing"
        )

    def forbidden_antspy():
        raise AssertionError(
            "ANTsPy must not run for skipped_existing"
        )

    monkeypatch.setattr(
        mri_to_mni,
        "validate_mni152_template",
        forbidden_validator,
    )
    monkeypatch.setattr(
        mri_to_mni,
        "_require_antspy",
        forbidden_antspy,
    )

    result = mri_to_mni.preprocess_one_t1w_to_mni(
        t1w_path=tmp_path / "input.nii.gz",
        subject=subject,
        session=session,
        out_root=out_root,
        mni_path=wrong_mni,
        overwrite=False,
    )

    assert result["status"] == "skipped_existing"


def _prepare_fs_reuse_tree(
    tmp_path: Path,
    *,
    include_mni_aseg: bool,
):
    fs_sub = tmp_path / "fs-subject"
    fs_mri = fs_sub / "mri"
    fs_mri.mkdir(parents=True)

    (fs_mri / "orig.mgz").write_bytes(b"x")
    (fs_mri / "aseg.mgz").write_bytes(b"x")

    out_root = tmp_path / "out"
    anat = out_root / "sub-0001" / "ses-01" / "anat"
    xfm = out_root / "sub-0001" / "ses-01" / "xfm"
    anat.mkdir(parents=True)
    xfm.mkdir(parents=True)

    stem = "sub-0001_ses-01"

    required = [
        anat / f"{stem}_desc-preproc_T1w.nii.gz",
        anat / f"{stem}_desc-aseg_dseg.nii.gz",
        anat / f"{stem}_space-MNI152_desc-preproc_T1w.nii.gz",
        xfm / f"{stem}_from-native_to-MNI152_mode-image_desc-antsAffine.mat",
        xfm / f"{stem}_desc-surface_xfm.json",
    ]

    if include_mni_aseg:
        required.append(
            anat / f"{stem}_space-MNI152_desc-aseg_dseg.nii.gz"
        )

    for path in required:
        path.write_bytes(b"existing")

    np.savetxt(
        xfm
        / f"{stem}_from-native_to-MNI152_mode-surface_xfm.txt",
        np.eye(4),
    )
    np.savetxt(
        xfm
        / f"{stem}_from-MNI152_to-native_mode-surface_xfm.txt",
        np.eye(4),
    )

    return fs_sub, out_root


def _run_fs_process_one(
    *,
    tmp_path: Path,
    fs_sub: Path,
    out_root: Path,
    mni_template: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fs_to_mni,
        "find_fs_subject_dir",
        lambda *args, **kwargs: fs_sub,
    )
    monkeypatch.setattr(
        fs_to_mni,
        "get_tkras_to_scanner_ras_from_orig",
        lambda *args, **kwargs: np.eye(4),
    )

    fs_to_mni.process_one(
        fs_root=tmp_path / "unused-fs-root",
        out_root=out_root,
        participant_label="sub-0001",
        session_label="ses-01",
        mni_template=mni_template,
        space="MNI152",
        surface_names=(),
        hemis=(),
        write_aparc_aseg=False,
        write_filled=False,
        strict_volumes=False,
        use_n4=False,
        n4_use_mask=True,
        n4_shrink_factor=4,
        n4_spline_distance=200.0,
        transform_type="affine",
        save_raw_t1_export=False,
        random_seed=None,
        skip_existing=True,
        strict_surfaces=False,
        write_legacy_anat_xfms=False,
        logger=logging.getLogger("test-mni-geometry"),
    )


def test_fs_preprocessing_full_reuse_does_not_validate_template(
    tmp_path,
    monkeypatch,
):
    fs_sub, out_root = _prepare_fs_reuse_tree(
        tmp_path,
        include_mni_aseg=True,
    )

    def forbidden_validator(path):
        raise AssertionError(
            "MNI validation must not run when all "
            "template-dependent outputs are reused"
        )

    monkeypatch.setattr(
        fs_to_mni,
        "validate_mni152_template",
        forbidden_validator,
    )

    _run_fs_process_one(
        tmp_path=tmp_path,
        fs_sub=fs_sub,
        out_root=out_root,
        mni_template=tmp_path / "unused-template.nii.gz",
        monkeypatch=monkeypatch,
    )


def test_fs_preprocessing_registration_validates_template_first(
    tmp_path,
    monkeypatch,
):
    fs_sub, out_root = _prepare_fs_reuse_tree(
        tmp_path,
        include_mni_aseg=True,
    )

    stem = "sub-0001_ses-01"
    f_t1_mni = (
        out_root
        / "sub-0001"
        / "ses-01"
        / "anat"
        / f"{stem}_space-MNI152_desc-preproc_T1w.nii.gz"
    )
    f_t1_mni.unlink()

    def validator_sentinel(path):
        raise RuntimeError("MNI_REGISTRATION_SENTINEL")

    def forbidden_registration(*args, **kwargs):
        raise AssertionError(
            "registration must not begin before MNI validation"
        )

    monkeypatch.setattr(
        fs_to_mni,
        "validate_mni152_template",
        validator_sentinel,
    )
    monkeypatch.setattr(
        fs_to_mni,
        "estimate_linear_registration",
        forbidden_registration,
    )

    with pytest.raises(
        RuntimeError,
        match="MNI_REGISTRATION_SENTINEL",
    ):
        _run_fs_process_one(
            tmp_path=tmp_path,
            fs_sub=fs_sub,
            out_root=out_root,
            mni_template=tmp_path / "bad-template.nii.gz",
            monkeypatch=monkeypatch,
        )


def test_fs_preprocessing_resample_only_validates_template_first(
    tmp_path,
    monkeypatch,
):
    fs_sub, out_root = _prepare_fs_reuse_tree(
        tmp_path,
        include_mni_aseg=False,
    )

    def validator_sentinel(path):
        raise RuntimeError("MNI_RESAMPLE_SENTINEL")

    def forbidden_apply(*args, **kwargs):
        raise AssertionError(
            "resampling must not begin before MNI validation"
        )

    monkeypatch.setattr(
        fs_to_mni,
        "validate_mni152_template",
        validator_sentinel,
    )
    monkeypatch.setattr(
        fs_to_mni,
        "apply_linear_transform_to_volume",
        forbidden_apply,
    )

    with pytest.raises(
        RuntimeError,
        match="MNI_RESAMPLE_SENTINEL",
    ):
        _run_fs_process_one(
            tmp_path=tmp_path,
            fs_sub=fs_sub,
            out_root=out_root,
            mni_template=tmp_path / "bad-template.nii.gz",
            monkeypatch=monkeypatch,
        )
