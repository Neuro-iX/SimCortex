#!/usr/bin/env python3
from __future__ import annotations

"""
Shared FCL-based surface-surface collision diagnostics for SimCortex.

This module provides a boolean collision check and an optional full contact-count
query with a configurable maximum number of contacts.

Important behavior
------------------
- FCL_MAX_CONTACTS is read from the SIMCORTEX_FCL_MAX_CONTACTS environment
  variable when this module is imported.
- make_fcl_object does not simplify or decimate meshes. Face percentages are
  therefore calculated using the original mesh face counts.
- If python-fcl is unavailable or cannot be initialized, the module reports an
  explicit backend status instead of raising an opaque import error.

These functions report raw mesh-intersection diagnostics. Same-hemisphere
white/pial surfaces may include medial-wall contacts depending on the surface
definition, so anatomical interpretation is performed downstream.
"""

import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import trimesh

LOG = logging.getLogger("collision_backend")

COLLISION_BACKEND_SCHEMA_VERSION = "simcortex_collision_backend_v1.1"


def _read_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from an environment variable."""
    raw = os.environ.get(name, "")
    if raw == "":
        return int(default)
    try:
        value = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be positive, got {value}")
    return value


# If the number of returned contacts reaches this ceiling, the pair is marked
# as saturated and the reported contact counts are lower bounds.
FCL_MAX_CONTACTS = _read_positive_int_env(
    "SIMCORTEX_FCL_MAX_CONTACTS",
    2_000_000,
)

try:
    import fcl  # type: ignore  # python-fcl

    # python-fcl's BVHModel.beginModel signature is conventionally
    # beginModel(num_vertices, num_tris). Keep this order consistent with the
    # python-fcl examples and with addSubModel(vertices, faces).
    _probe_mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    _probe_bvh = fcl.BVHModel()
    _probe_bvh.beginModel(int(len(_probe_mesh.vertices)), int(len(_probe_mesh.faces)))
    _probe_bvh.addSubModel(
        np.asarray(_probe_mesh.vertices, dtype=np.float64),
        np.asarray(_probe_mesh.faces, dtype=np.int64),
    )
    _probe_bvh.endModel()
    HAS_FCL = True
    FCL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on environment
    fcl = None  # type: ignore
    HAS_FCL = False
    FCL_IMPORT_ERROR = repr(exc)


def _base_row(n_a: int, n_b: int, status: str, error: str = "") -> Dict[str, Any]:
    """Return the shared output schema for both boolean and full-count queries.

    num_contacts_saturated is unknown by default. It is set explicitly to
    False/True only for successful queries where a contact-count attempt was
    actually made or when a true no-collision result is known.
    """
    return {
        "backend_schema_version": COLLISION_BACKEND_SCHEMA_VERSION,
        "fcl_status": status,
        "fcl_error": error,
        "collision_detected": np.nan,
        "num_contacts": np.nan,
        "num_contacts_saturated": np.nan,
        "intersecting_faces_A": np.nan,
        "intersecting_faces_B": np.nan,
        "pct_faces_A": np.nan,
        "pct_faces_B": np.nan,
        "total_faces_A": int(n_a),
        "total_faces_B": int(n_b),
        "contact_index_failures": 0,
        "count_mode": "",
        "fcl_max_contacts": int(FCL_MAX_CONTACTS),
    }


def _empty_or_unavailable_row(n_a: int, n_b: int, status: str, error: str = "") -> Dict[str, Any]:
    """Return a row for unavailable FCL or empty meshes."""
    return _base_row(n_a, n_b, status, error)


def _validate_trimesh(tri: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(tri.vertices, dtype=np.float64)
    faces = np.asarray(tri.faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertex array shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid face array shape: {faces.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains non-finite vertices")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Mesh faces contain invalid vertex indices")

    return vertices, faces


def make_fcl_object(tri: trimesh.Trimesh) -> Tuple[Optional[object], int]:
    """Build a python-fcl CollisionObject from a trimesh.

    Returns
    -------
    obj, n_faces
        obj is None when FCL is unavailable or the mesh has zero faces.
        n_faces is always the original input face count; no simplification is
        performed here.
    """
    n_faces = int(len(tri.faces)) if tri.faces is not None else 0
    if not HAS_FCL or n_faces == 0:
        return None, n_faces

    vertices, faces = _validate_trimesh(tri)

    model = fcl.BVHModel()
    # python-fcl convention: beginModel(num_vertices, num_tris).
    model.beginModel(int(len(vertices)), int(len(faces)))
    model.addSubModel(vertices, faces)
    model.endModel()
    return fcl.CollisionObject(model, fcl.Transform()), n_faces


def collision_bool_from_objects(obj_a: object, n_a: int, obj_b: object, n_b: int) -> Dict[str, Any]:
    """Cheap boolean collision query between two prebuilt FCL objects.

    This avoids collecting all triangle contacts. It is used by script 07 to
    skip the expensive counting worker when two surfaces clearly do not collide.
    The returned num_contacts is diagnostic only in boolean mode; exact counts
    are produced by collision_pair_from_objects.
    """
    if not HAS_FCL:
        return _empty_or_unavailable_row(n_a, n_b, "no_fcl", FCL_IMPORT_ERROR)
    if obj_a is None or obj_b is None or int(n_a) == 0 or int(n_b) == 0:
        return _empty_or_unavailable_row(n_a, n_b, "empty_mesh")

    row = _base_row(n_a, n_b, "OK")
    row["count_mode"] = "boolean"

    try:
        # The return value of fcl.collide is often the number of contacts found.
        # With enable_contact=False this is best treated as boolean evidence / a
        # diagnostic count only, not an exact contact count.
        req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
        res = fcl.CollisionResult()
        ret = fcl.collide(obj_a, obj_b, req, res)

        detected = bool(ret)
        is_collision = getattr(res, "is_collision", None)
        if isinstance(is_collision, bool):
            detected = bool(detected or is_collision)

        row.update(
            {
                "collision_detected": bool(detected),
                "num_contacts": int(ret),
                "num_contacts_saturated": False,
                "intersecting_faces_A": np.nan,
                "intersecting_faces_B": np.nan,
                "pct_faces_A": np.nan,
                "pct_faces_B": np.nan,
                "contact_index_failures": 0,
            }
        )
    except Exception as exc:
        row["fcl_status"] = "error"
        row["fcl_error"] = repr(exc)
        row["collision_detected"] = np.nan
        LOG.warning("Boolean FCL collision query failed: %s", exc)

    return row


def collision_pair_from_objects(obj_a: object, n_a: int, obj_b: object, n_b: int) -> Dict[str, Any]:
    """Full triangle-triangle contact diagnostics between two FCL objects.

    The function returns exact contact/face percentages unless the number of
    reported contacts reaches FCL_MAX_CONTACTS, in which case
    num_contacts_saturated=True and the counts are lower bounds.
    """
    if not HAS_FCL:
        return _empty_or_unavailable_row(n_a, n_b, "no_fcl", FCL_IMPORT_ERROR)
    if obj_a is None or obj_b is None or int(n_a) == 0 or int(n_b) == 0:
        return _empty_or_unavailable_row(n_a, n_b, "empty_mesh")

    row = _base_row(n_a, n_b, "OK")
    row["count_mode"] = "all_contacts"

    try:
        req = fcl.CollisionRequest(num_max_contacts=int(FCL_MAX_CONTACTS), enable_contact=True)
        res = fcl.CollisionResult()
        fcl.collide(obj_a, obj_b, req, res)

        contacts = list(getattr(res, "contacts", []))
        n_contacts = int(len(contacts))
        detected = n_contacts > 0

        if not detected:
            row.update(
                {
                    "collision_detected": False,
                    "num_contacts": 0,
                    "num_contacts_saturated": False,
                    "intersecting_faces_A": 0,
                    "intersecting_faces_B": 0,
                    "pct_faces_A": 0.0,
                    "pct_faces_B": 0.0,
                    "contact_index_failures": 0,
                }
            )
            return row

        faces_a: Set[int] = set()
        faces_b: Set[int] = set()
        index_failures = 0

        for contact in contacts:
            try:
                faces_a.add(int(contact.b1))
                faces_b.add(int(contact.b2))
            except Exception:
                index_failures += 1

        intersecting_a = int(len(faces_a))
        intersecting_b = int(len(faces_b))
        pct_a = float(intersecting_a / int(n_a) * 100.0) if int(n_a) > 0 else np.nan
        pct_b = float(intersecting_b / int(n_b) * 100.0) if int(n_b) > 0 else np.nan

        row.update(
            {
                "collision_detected": True,
                "num_contacts": n_contacts,
                "num_contacts_saturated": bool(n_contacts >= int(FCL_MAX_CONTACTS)),
                "intersecting_faces_A": intersecting_a,
                "intersecting_faces_B": intersecting_b,
                "pct_faces_A": pct_a,
                "pct_faces_B": pct_b,
                "contact_index_failures": int(index_failures),
            }
        )

        if n_contacts > 0 and intersecting_a == 0 and intersecting_b == 0:
            row["fcl_status"] = "contacts_without_indices"
            row["fcl_error"] = "FCL returned contacts but no usable face indices."

    except Exception as exc:
        row["fcl_status"] = "error"
        row["fcl_error"] = repr(exc)
        row["collision_detected"] = np.nan
        LOG.warning("Full FCL collision diagnostics failed: %s", exc)

    return row


def collision_pair_from_meshes(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> Dict[str, Any]:
    """Convenience wrapper: build FCL objects and run full diagnostics."""
    obj_a, n_a = make_fcl_object(mesh_a)
    obj_b, n_b = make_fcl_object(mesh_b)
    return collision_pair_from_objects(obj_a, n_a, obj_b, n_b)
