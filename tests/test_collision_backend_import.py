from pathlib import Path

import pytest
import trimesh


def _load_backend():
    import simcortex.utils.collision_backend as backend

    return backend


def _require_fcl(backend):
    if not backend.HAS_FCL:
        import_error = getattr(
            backend,
            "FCL_IMPORT_ERROR",
            "unknown import error",
        )
        pytest.skip(
            f"python-fcl is unavailable: {import_error}"
        )


def test_collision_backend_public_api():
    backend = _load_backend()

    repository_root = Path(__file__).resolve().parents[1]
    expected_src = (repository_root / "src").resolve()
    module_path = Path(backend.__file__).resolve()

    assert expected_src in module_path.parents
    assert isinstance(backend.HAS_FCL, bool)

    assert callable(backend.make_fcl_object)
    assert callable(backend.collision_bool_from_objects)
    assert callable(backend.collision_pair_from_objects)
    assert callable(backend.collision_pair_from_meshes)


def test_non_intersecting_meshes():
    backend = _load_backend()
    _require_fcl(backend)

    mesh_a = trimesh.creation.icosphere(
        subdivisions=1,
        radius=1.0,
    )
    mesh_b = trimesh.creation.icosphere(
        subdivisions=1,
        radius=1.0,
    )
    mesh_b.apply_translation((3.0, 0.0, 0.0))

    result = backend.collision_pair_from_meshes(
        mesh_a,
        mesh_b,
    )

    assert result["fcl_status"] == "OK"
    assert result["collision_detected"] is False
    assert result["num_contacts"] == 0
    assert result["intersecting_faces_A"] == 0
    assert result["intersecting_faces_B"] == 0
    assert result["pct_faces_A"] == 0.0
    assert result["pct_faces_B"] == 0.0
    assert result["total_faces_A"] == len(mesh_a.faces)
    assert result["total_faces_B"] == len(mesh_b.faces)


def test_intersecting_meshes():
    backend = _load_backend()
    _require_fcl(backend)

    mesh_a = trimesh.creation.icosphere(
        subdivisions=1,
        radius=1.0,
    )
    mesh_b = trimesh.creation.icosphere(
        subdivisions=1,
        radius=1.0,
    )
    mesh_b.apply_translation((1.5, 0.1, 0.2))

    result = backend.collision_pair_from_meshes(
        mesh_a,
        mesh_b,
    )

    assert result["fcl_status"] == "OK"
    assert result["collision_detected"] is True
    assert result["num_contacts"] > 0
    assert result["intersecting_faces_A"] > 0
    assert result["intersecting_faces_B"] > 0
    assert 0.0 < result["pct_faces_A"] <= 100.0
    assert 0.0 < result["pct_faces_B"] <= 100.0
    assert result["total_faces_A"] == len(mesh_a.faces)
    assert result["total_faces_B"] == len(mesh_b.faces)
