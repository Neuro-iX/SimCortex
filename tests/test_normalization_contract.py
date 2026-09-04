"""Regression tests for SimCortex MRI intensity-normalization behavior."""

from __future__ import annotations

import numpy as np
import pytest

from simcortex.deform.data.dataloader import normalize_mri_mean_std
from simcortex.seg.data.dataloader import robust_normalize


def test_segmentation_robust_normalize_uses_positive_p99() -> None:
    """Segmentation uses positive-voxel p99 clipping and scaling."""
    vol = np.array(
        [0.0, -5.0, 1.0, 2.0, 3.0, 100.0],
        dtype=np.float32,
    )

    out = robust_normalize(vol)

    p99 = float(np.percentile(np.array([1.0, 2.0, 3.0, 100.0]), 99))

    assert out.dtype == np.float32
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.0 / p99)
    assert out[-1] == pytest.approx(1.0)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_segmentation_robust_normalize_sanitizes_nonfinite_no_foreground() -> None:
    """If no positive foreground remains, return the sanitized volume."""
    vol = np.array(
        [-1.0, 0.0, np.nan, np.inf, -np.inf],
        dtype=np.float32,
    )

    out = robust_normalize(vol)

    np.testing.assert_array_equal(
        out,
        np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_deformation_normalization_uses_nonzero_mask_at_100_voxels() -> None:
    """With >=100 nonzero voxels, deformation statistics exclude zeros."""
    nonzero = np.arange(1, 101, dtype=np.float32)
    mri = np.concatenate(
        [np.zeros(2, dtype=np.float32), nonzero]
    )

    out = normalize_mri_mean_std(mri)

    mean = float(nonzero.mean())
    std = float(nonzero.std())

    assert out.dtype == np.float32
    assert float(out[2:].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(out[2:].std()) == pytest.approx(1.0, abs=1e-6)
    assert float(out[0]) == pytest.approx((0.0 - mean) / std, rel=1e-6)


def test_deformation_normalization_falls_back_to_full_volume_below_100() -> None:
    """With <100 nonzero voxels, deformation uses full-volume statistics."""
    mri = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    out = normalize_mri_mean_std(mri)

    expected = (mri - float(mri.mean())) / float(mri.std())

    np.testing.assert_allclose(
        out,
        expected.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_deformation_normalization_clamps_zero_standard_deviation() -> None:
    """The deformation std denominator is clamped to at least 1e-6."""
    mri = np.zeros((4,), dtype=np.float32)

    out = normalize_mri_mean_std(mri)

    np.testing.assert_array_equal(
        out,
        np.zeros_like(mri, dtype=np.float32),
    )
