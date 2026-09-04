"""Regression tests for fail-fast surface geometry validation."""

from __future__ import annotations

import numpy as np
import pytest

import simcortex.deform.inference as deform_infer
import simcortex.initsurf.generate as initsurf_generate


def test_deform_export_rejects_nonfinite_vertices_before_write(tmp_path):
    """Deformation inference must never write NaN/Inf surface vertices."""
    faces = np.asarray(
        [[0, 1, 2]],
        dtype=np.int64,
    )

    base = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    for name, value in [
        ("nan", np.nan),
        ("posinf", np.inf),
        ("neginf", -np.inf),
    ]:
        vertices = base.copy()
        vertices[1, 1] = value

        out_path = tmp_path / f"{name}.ply"

        with pytest.raises(
            RuntimeError,
            match="Non-finite predicted vertices",
        ):
            deform_infer._export_predicted_surface(
                str(out_path),
                vertices,
                faces,
            )

        assert not out_path.exists()


def test_initsurf_rejects_nonfinite_world_vertices_before_trimesh(
    monkeypatch,
):
    """InitSurf must reject NaN/Inf before constructing a Trimesh."""
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    faces = np.asarray(
        [[0, 1, 2]],
        dtype=np.int64,
    )

    def fake_marching_cubes(*args, **kwargs):
        return (
            vertices.copy(),
            faces.copy(),
            np.zeros(len(vertices), dtype=np.float32),
            np.zeros(len(vertices), dtype=np.float32),
        )

    def forbidden_trimesh(*args, **kwargs):
        raise AssertionError(
            "Trimesh construction must not occur "
            "for non-finite InitSurf vertices"
        )

    monkeypatch.setattr(
        initsurf_generate,
        "marching_cubes",
        fake_marching_cubes,
    )
    monkeypatch.setattr(
        initsurf_generate.trimesh,
        "Trimesh",
        forbidden_trimesh,
    )

    for value in [
        np.nan,
        np.inf,
        -np.inf,
    ]:
        def fake_apply_affine(
            matrix,
            verts,
            bad_value=value,
        ):
            out = np.asarray(
                verts,
                dtype=np.float64,
            ).copy()
            out[1, 1] = bad_value
            return out

        monkeypatch.setattr(
            initsurf_generate,
            "apply_affine",
            fake_apply_affine,
        )

        with pytest.raises(
            RuntimeError,
            match=r"Non-finite InitSurf vertices at level=0\.25",
        ):
            initsurf_generate.mesh_from_topo_sdf(
                sdf_topo=np.zeros(
                    (2, 2, 2),
                    dtype=np.float32,
                ),
                level=0.25,
                brain_affine=np.eye(
                    4,
                    dtype=np.float64,
                ),
                n_smooth=0,
            )
