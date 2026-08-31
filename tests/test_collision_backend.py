"""Regression tests for the shared SimCortex FCL collision backend."""

import numpy as np
import pytest
import trimesh

import simcortex.utils.collision_backend as cb


def _sphere(center=(0.0, 0.0, 0.0)):
    """Create a small deterministic triangular test mesh."""
    mesh = trimesh.creation.icosphere(
        subdivisions=1,
        radius=1.0,
    )
    mesh.apply_translation(np.asarray(center, dtype=np.float64))
    return mesh


def test_collision_backend_schema_version():
    """The public collision-row schema must remain versioned explicitly."""
    assert (
        cb.COLLISION_BACKEND_SCHEMA_VERSION
        == "simcortex_collision_backend_v1.1"
    )


def test_fcl_max_contacts_default(monkeypatch):
    """The historical default contact ceiling must remain two million."""
    monkeypatch.delenv(
        "SIMCORTEX_FCL_MAX_CONTACTS",
        raising=False,
    )
    monkeypatch.delenv(
        "SCPP_FCL_MAX_CONTACTS",
        raising=False,
    )

    assert cb._read_fcl_max_contacts() == 2_000_000


def test_fcl_max_contacts_historical_env_fallback(monkeypatch):
    """The historical SCPP environment variable remains supported."""
    monkeypatch.delenv(
        "SIMCORTEX_FCL_MAX_CONTACTS",
        raising=False,
    )
    monkeypatch.setenv(
        "SCPP_FCL_MAX_CONTACTS",
        "12345",
    )

    assert cb._read_fcl_max_contacts() == 12345


def test_fcl_max_contacts_canonical_env_has_precedence(monkeypatch):
    """The canonical SimCortex environment variable wins when both exist."""
    monkeypatch.setenv(
        "SCPP_FCL_MAX_CONTACTS",
        "12345",
    )
    monkeypatch.setenv(
        "SIMCORTEX_FCL_MAX_CONTACTS",
        "67890",
    )

    assert cb._read_fcl_max_contacts() == 67890


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "not-an-integer"],
)
def test_fcl_max_contacts_rejects_invalid_values(
    monkeypatch,
    value,
):
    """Invalid canonical contact limits must fail explicitly."""
    monkeypatch.setenv(
        "SIMCORTEX_FCL_MAX_CONTACTS",
        value,
    )

    with pytest.raises(ValueError):
        cb._read_fcl_max_contacts()


def test_make_fcl_object_preserves_original_face_count():
    """FCL conversion must not simplify or decimate the input mesh."""
    if not cb.HAS_FCL:
        pytest.skip(
            f"python-fcl unavailable: {cb.FCL_IMPORT_ERROR}"
        )

    mesh = _sphere()
    expected_faces = len(mesh.faces)

    obj, n_faces = cb.make_fcl_object(mesh)

    assert obj is not None
    assert n_faces == expected_faces
    assert len(mesh.faces) == expected_faces


def test_make_fcl_object_empty_mesh():
    """An empty mesh must return no FCL object and zero faces."""
    mesh = trimesh.Trimesh(
        vertices=np.empty((0, 3), dtype=np.float64),
        faces=np.empty((0, 3), dtype=np.int64),
        process=False,
    )

    obj, n_faces = cb.make_fcl_object(mesh)

    assert obj is None
    assert n_faces == 0


def test_separated_meshes_report_no_collision():
    """Clearly separated surfaces must report zero collision/contact."""
    if not cb.HAS_FCL:
        pytest.skip(
            f"python-fcl unavailable: {cb.FCL_IMPORT_ERROR}"
        )

    mesh_a = _sphere((0.0, 0.0, 0.0))
    mesh_b = _sphere((3.0, 0.0, 0.0))

    obj_a, n_a = cb.make_fcl_object(mesh_a)
    obj_b, n_b = cb.make_fcl_object(mesh_b)

    boolean_row = cb.collision_bool_from_objects(
        obj_a,
        n_a,
        obj_b,
        n_b,
    )

    assert boolean_row["fcl_status"] == "OK"
    assert boolean_row["count_mode"] == "boolean"
    assert boolean_row["collision_detected"] is False
    assert boolean_row["num_contacts"] == 0
    assert boolean_row["num_contacts_saturated"] is False
    assert boolean_row["total_faces_A"] == n_a
    assert boolean_row["total_faces_B"] == n_b

    full_row = cb.collision_pair_from_objects(
        obj_a,
        n_a,
        obj_b,
        n_b,
    )

    assert full_row["fcl_status"] == "OK"
    assert full_row["count_mode"] == "all_contacts"
    assert full_row["collision_detected"] is False
    assert full_row["num_contacts"] == 0
    assert full_row["num_contacts_saturated"] is False
    assert full_row["intersecting_faces_A"] == 0
    assert full_row["intersecting_faces_B"] == 0
    assert full_row["pct_faces_A"] == 0.0
    assert full_row["pct_faces_B"] == 0.0


def test_intersecting_meshes_report_collision():
    """Overlapping surfaces must produce positive collision diagnostics."""
    if not cb.HAS_FCL:
        pytest.skip(
            f"python-fcl unavailable: {cb.FCL_IMPORT_ERROR}"
        )

    mesh_a = _sphere((0.0, 0.0, 0.0))
    mesh_b = _sphere((0.5, 0.0, 0.0))

    obj_a, n_a = cb.make_fcl_object(mesh_a)
    obj_b, n_b = cb.make_fcl_object(mesh_b)

    boolean_row = cb.collision_bool_from_objects(
        obj_a,
        n_a,
        obj_b,
        n_b,
    )

    assert boolean_row["fcl_status"] == "OK"
    assert boolean_row["collision_detected"] is True

    full_row = cb.collision_pair_from_objects(
        obj_a,
        n_a,
        obj_b,
        n_b,
    )

    assert full_row["fcl_status"] == "OK"
    assert full_row["count_mode"] == "all_contacts"
    assert full_row["collision_detected"] is True

    assert full_row["num_contacts"] > 0

    assert 0 < full_row["intersecting_faces_A"] <= n_a
    assert 0 < full_row["intersecting_faces_B"] <= n_b

    assert 0.0 < full_row["pct_faces_A"] <= 100.0
    assert 0.0 < full_row["pct_faces_B"] <= 100.0

    assert full_row["total_faces_A"] == n_a
    assert full_row["total_faces_B"] == n_b

    assert full_row["contact_index_failures"] == 0
