"""Geometry contract for the canonical SimCortex MNI152 1 mm template."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


MNI152_TEMPLATE_SHAPE = (182, 218, 182)

MNI152_TEMPLATE_AFFINE = np.asarray(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

MNI152_AFFINE_ATOL = 1.0e-4


def validate_mni152_template(path: str | Path) -> None:
    """Reject templates that do not match the canonical SimCortex MNI152 grid."""
    template_path = Path(path)

    if not template_path.is_file():
        raise FileNotFoundError(
            f"MNI152 template not found: {template_path}"
        )

    img = nib.load(str(template_path))

    shape = tuple(int(v) for v in img.shape)
    if shape != MNI152_TEMPLATE_SHAPE:
        raise ValueError(
            "MNI152 template shape mismatch: "
            f"got {shape}, expected {MNI152_TEMPLATE_SHAPE}"
        )

    affine = np.asarray(img.affine, dtype=np.float64)

    if affine.shape != (4, 4):
        raise ValueError(
            "MNI152 template affine must have shape (4, 4), "
            f"got {affine.shape}"
        )

    if not np.isfinite(affine).all():
        raise ValueError(
            "MNI152 template affine contains non-finite values"
        )

    if not np.allclose(
        affine,
        MNI152_TEMPLATE_AFFINE,
        atol=MNI152_AFFINE_ATOL,
        rtol=0.0,
    ):
        raise ValueError(
            "MNI152 template affine does not match the canonical "
            "SimCortex MNI152 1 mm grid"
        )
