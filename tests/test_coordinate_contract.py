"""Regression tests for the SimCortex coordinate-space contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import simcortex.initsurf.generate as initsurf_generate
import simcortex.preproc.fs_to_mni as fs_to_mni
from simcortex.deform.models.surfdeform import SurfDeform
from simcortex.deform.utils.coords import (
    make_center_crop_pad_slices,
    voxel_to_world,
    world_to_voxel,
)


def test_antspy_point_transform_direction_and_lps_to_ras(
    monkeypatch,
) -> None:
    """ANTs point affine is MNI->native; surface export uses its RAS inverse."""
    mni_to_native_lps = np.eye(4, dtype=np.float64)
    mni_to_native_lps[:3, 3] = [10.0, 20.0, 30.0]

    monkeypatch.setattr(
        fs_to_mni,
        "ants_affine_to_homogeneous_lps",
        lambda _: mni_to_native_lps.copy(),
    )

    native_to_mni_ras, mni_to_native_ras = (
        fs_to_mni.compute_surface_point_matrices_from_ants_mat(
            Path("unused.mat")
        )
    )

    expected_mni_to_native_ras = np.eye(4, dtype=np.float64)
    expected_mni_to_native_ras[:3, 3] = [-10.0, -20.0, 30.0]

    expected_native_to_mni_ras = np.linalg.inv(
        expected_mni_to_native_ras
    )

    np.testing.assert_allclose(
        mni_to_native_ras,
        expected_mni_to_native_ras,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        native_to_mni_ras,
        expected_native_to_mni_ras,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        native_to_mni_ras @ mni_to_native_ras,
        np.eye(4),
        rtol=0.0,
        atol=1e-12,
    )


def test_initsurf_marching_cubes_vertices_are_exported_in_world_mm(
    monkeypatch,
) -> None:
    """InitSurf applies the NIfTI voxel-to-world affine before mesh export."""
    verts_vox = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    def fake_marching_cubes(*args, **kwargs):
        normals = np.zeros_like(verts_vox)
        values = np.zeros(len(verts_vox), dtype=np.float32)
        return verts_vox.copy(), faces.copy(), normals, values

    monkeypatch.setattr(
        initsurf_generate,
        "marching_cubes",
        fake_marching_cubes,
    )
    monkeypatch.setattr(
        initsurf_generate.trimesh.repair,
        "fix_winding",
        lambda mesh: None,
    )
    monkeypatch.setattr(
        initsurf_generate.trimesh.repair,
        "fix_normals",
        lambda mesh: None,
    )

    affine = np.asarray(
        [
            [-1.0, 0.0, 0.0, 90.0],
            [0.0, 1.0, 0.0, -126.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    mesh = initsurf_generate.mesh_from_topo_sdf(
        sdf_topo=np.zeros((2, 2, 2), dtype=np.float32),
        level=0.0,
        brain_affine=affine,
        n_smooth=0,
    )

    expected_world = np.column_stack(
        [
            90.0 - verts_vox[:, 0],
            -126.0 + verts_vox[:, 1],
            -72.0 + verts_vox[:, 2],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(mesh.vertices),
        expected_world,
        rtol=0.0,
        atol=1e-6,
    )


def test_deform_center_padding_shift_preserves_world_coordinates() -> None:
    """The 182x218x182 -> 184x224x184 shift is exactly reversible."""
    _, pad_before, pad_after, crop_before = (
        make_center_crop_pad_slices(
            (182, 218, 182),
            (184, 224, 184),
        )
    )

    assert pad_before == (1, 3, 1)
    assert pad_after == (1, 3, 1)
    assert crop_before == (0, 0, 0)

    shift = torch.tensor(
        np.asarray(pad_before) - np.asarray(crop_before),
        dtype=torch.float32,
    )

    affine = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 90.0],
            [0.0, 1.0, 0.0, -126.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    verts_world = torch.tensor(
        [
            [80.0, -106.0, -42.0],
            [40.0, -66.0, 8.0],
        ],
        dtype=torch.float32,
    )

    verts_vox_original = world_to_voxel(
        verts_world,
        affine,
    )
    verts_vox_internal = verts_vox_original + shift

    # This mirrors deformation export:
    # v_vox_orig = v_vox_cp - shift_ijk
    restored_vox = verts_vox_internal - shift
    restored_world = voxel_to_world(
        restored_vox,
        affine,
    )

    torch.testing.assert_close(
        restored_vox,
        verts_vox_original,
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        restored_world,
        verts_world,
        rtol=0.0,
        atol=1e-5,
    )


def test_deform_interpolation_uses_ijk_input_and_grid_sample_xyz_order() -> None:
    """Exact IJK voxel coordinates sample the matching DHW voxel."""
    D, H, W = 3, 4, 5

    d = torch.arange(D, dtype=torch.float32)[:, None, None]
    h = torch.arange(H, dtype=torch.float32)[None, :, None]
    w = torch.arange(W, dtype=torch.float32)[None, None, :]

    vol = 100.0 * d + 10.0 * h + w
    src = vol.unsqueeze(0).unsqueeze(0)

    # IJK / DHW coordinate: d=1, h=2, w=3.
    coord_ijk = torch.tensor(
        [[[[[1.0, 2.0, 3.0]]]]],
        dtype=torch.float32,
    )

    out = SurfDeform.interpolate(
        None,
        coord_ijk,
        src,
    )

    assert out.shape == (1, 1, 1, 1, 1)
    torch.testing.assert_close(
        out[0, 0, 0, 0, 0],
        torch.tensor(123.0),
        rtol=0.0,
        atol=1e-5,
    )
