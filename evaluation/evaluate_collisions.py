#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Evaluate SimCortex cortical-surface collisions with python-fcl.

For each predicted case, six anatomically relevant cortical-surface pairs are
evaluated:

  1. white_pial_left        = lh_white vs lh_pial
  2. white_pial_right       = rh_white vs rh_pial
  3. pial_lr                = lh_pial vs rh_pial
  4. white_lr               = lh_white vs rh_white
  5. cross_lhwhite_rhpial   = lh_white vs rh_pial
  6. cross_rhwhite_lhpial   = rh_white vs lh_pial

For each surface, the evaluator also takes the union of unique colliding face
IDs across the other three surfaces:

    collision_pct_union =
        100 * unique_colliding_faces / total_surface_faces

The case-level collision metric is the mean of the four surface-level union
percentages.

Outputs are written under:

  <eval-root>/collisions/

The case worker uses python-fcl directly so that saturated contact queries can
be repeated using the configured maximum-contact ladder.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

# Canonical backend is used only for FCL availability and backend metadata.
# The case worker below intentionally uses python-fcl directly so the historical
# maximum-contact retry ladder remains unchanged.
from simcortex.utils import collision_backend as _collision_backend

HAS_FCL = bool(getattr(_collision_backend, "HAS_FCL", False))
FCL_IMPORT_ERROR = str(getattr(_collision_backend, "FCL_IMPORT_ERROR", ""))
COLLISION_BACKEND_FILE = str(getattr(_collision_backend, "__file__", "unknown"))

METHOD_NAME = "SimCortex"

COLLISION_SCHEMA_VERSION = (
    "simcortex_evaluation_fcl_collision_union_v1.0"
)
STDERR_TAIL_CHARS = 6000

SURFACE_KEYS = ("lh_white", "lh_pial", "rh_white", "rh_pial")

COLLISION_PAIRS: Dict[str, Tuple[str, str]] = {
    "white_pial_left": ("lh_white", "lh_pial"),
    "white_pial_right": ("rh_white", "rh_pial"),
    "pial_lr": ("lh_pial", "rh_pial"),
    "white_lr": ("lh_white", "rh_white"),
    "cross_lhwhite_rhpial": ("lh_white", "rh_pial"),
    "cross_rhwhite_lhpial": ("rh_white", "lh_pial"),
}

PAIR_NAMES = tuple(COLLISION_PAIRS.keys())

COLLISION_COLUMNS = [
    "schema_version",
    "method",
    "mesh_set",
    "dataset",
    "case_id",
    "subject",
    "session",
    "pair",
    "surface_A",
    "surface_B",
    "path_A",
    "path_B",
    "bool_fcl_status",
    "bool_fcl_error",
    "collision_detected",
    "bool_num_contacts",
    "bool_count_mode",
    "fcl_object_faces_A",
    "fcl_object_faces_B",
    "count_status",
    "count_error",
    "num_contacts",
    "num_contacts_saturated",
    "intersecting_faces_A",
    "intersecting_faces_B",
    "pct_faces_A",
    "pct_faces_B",
    "total_faces_A",
    "total_faces_B",
    "count_total_faces_A",
    "count_total_faces_B",
    "face_count_mismatch_A",
    "face_count_mismatch_B",
    "contact_index_failures",
    "contact_count_exact",
    "max_contacts_used",
    "worker_returncode",
    "worker_stderr_tail",
    "traceback",
]

UNION_COLUMNS = [
    "schema_version",
    "method",
    "mesh_set",
    "dataset",
    "case_id",
    "subject",
    "session",
    "lh_white_faces_total",
    "lh_pial_faces_total",
    "rh_white_faces_total",
    "rh_pial_faces_total",
    "lh_white_faces_colliding_union",
    "lh_pial_faces_colliding_union",
    "rh_white_faces_colliding_union",
    "rh_pial_faces_colliding_union",
    "lh_white_collision_pct_union",
    "lh_pial_collision_pct_union",
    "rh_white_collision_pct_union",
    "rh_pial_collision_pct_union",
    "collision_pct_union_mean4",
    "collision_pct_union_max4",
    "collision_faces_union_sum4",
    "n_pair_rows",
    "n_collision_true_pairs",
    "n_saturated_pairs",
    "n_nonexact_true_pairs",
    "total_contacts_sum",
    "total_contact_index_failures",
    "union_contact_count_exact_all_pairs",
    "union_status",
    "union_error",
]

UNION_SUMMARY_VALUE_COLS = [
    "lh_white_collision_pct_union",
    "lh_pial_collision_pct_union",
    "rh_white_collision_pct_union",
    "rh_pial_collision_pct_union",
    "collision_pct_union_mean4",
    "collision_pct_union_max4",
    "collision_faces_union_sum4",
    "n_collision_true_pairs",
    "n_saturated_pairs",
    "n_nonexact_true_pairs",
    "total_contacts_sum",
    "total_contact_index_failures",
]

MISSING_COLUMNS = [
    "method",
    "dataset",
    "case_id",
    "subject",
    "session",
    "status",
    "error",
    "traceback",
]


def parse_cap_ladder(s: str) -> List[int]:
    vals: List[int] = []
    for raw in str(s).replace(";", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid contact cap {token!r}; expected a positive integer") from exc
        if not np.isfinite(value) or value <= 0 or not float(value).is_integer():
            raise ValueError(f"Invalid contact cap {token!r}; expected a positive integer")
        vals.append(int(value))
    if not vals:
        raise ValueError("Empty --max-contact-ladder")
    return sorted(set(vals))


def safe_json_value(v: Any) -> Any:
    try:
        scalar_na = pd.isna(v)
        if isinstance(scalar_na, (bool, np.bool_)) and bool(scalar_na):
            return None
    except Exception:
        pass
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        value = float(v)
        return value if np.isfinite(value) else None
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return [safe_json_value(x) for x in v.tolist()]
    if isinstance(v, pd.DataFrame):
        return v.to_dict(orient="records")
    if isinstance(v, (pd.Series, pd.Index)):
        return [safe_json_value(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): safe_json_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [safe_json_value(x) for x in v]
    try:
        return str(v)
    except Exception:
        return None


def to_bool_or_none(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        if s in {"", "nan", "none", "null", "na", "pd.na"}:
            return None
        return None
    try:
        scalar_na = pd.isna(v)
        if isinstance(scalar_na, (bool, np.bool_)) and bool(scalar_na):
            return None
    except Exception:
        pass
    if isinstance(v, (list, tuple, dict, np.ndarray, pd.Series)):
        return None
    try:
        return bool(v)
    except Exception:
        return None


def bool_count(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(sum(to_bool_or_none(v) is True for v in df[col].values))


def safe_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def pct_from_count(intersecting_faces: Any, total_faces: Any) -> float:
    n = safe_float_or_nan(intersecting_faces)
    total = safe_float_or_nan(total_faces)
    if not np.isfinite(n) or not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(100.0 * n / total)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def true_collision_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "collision_detected" not in df.columns:
        return pd.Series([], dtype=bool)
    return df["collision_detected"].map(lambda x: to_bool_or_none(x) is True)


def dataframe_with_columns(rows: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    public_rows = []
    for row in rows:
        public_rows.append({k: v for k, v in row.items() if not str(k).startswith("_")})
    df = pd.DataFrame(public_rows)
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    extra = [c for c in df.columns if c not in columns]
    return df[columns + extra]


def flatten_columns(columns: Iterable[Any]) -> List[str]:
    flat: List[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) != ""]
            flat.append("_".join(parts).rstrip("_"))
        else:
            flat.append(str(col))
    return flat


def load_tri(path: Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"white", "pial"} or path.name.endswith(".pial.T1"):
        v, f = nib.freesurfer.io.read_geometry(str(path))
        vertices = np.asarray(v, dtype=np.float64)
        faces = np.asarray(f, dtype=np.int64)
    else:
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if not geoms:
                raise ValueError(f"Empty scene: {path}")
            mesh = trimesh.util.concatenate(geoms)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertex shape {vertices.shape}: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid face shape {faces.shape}: {path}")
    if len(vertices) == 0:
        raise ValueError(f"Empty vertex array in {path}")
    if len(faces) == 0:
        raise ValueError(f"Empty face array in {path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite vertices in {path}")
    min_face = int(faces.min())
    max_face = int(faces.max())
    if min_face < 0 or max_face >= len(vertices):
        raise ValueError(
            f"Face indices out of range in {path}: min={min_face}, max={max_face}, n_vertices={len(vertices)}"
        )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def make_fcl_object_worker(fcl_mod: Any, tri: trimesh.Trimesh) -> Tuple[object, int]:
    vertices = np.asarray(tri.vertices, dtype=np.float64)
    faces = np.asarray(tri.faces, dtype=np.int64)
    n_faces = int(len(faces))
    model = fcl_mod.BVHModel()
    model.beginModel(int(len(vertices)), n_faces)
    model.addSubModel(vertices, faces)
    model.endModel()
    return fcl_mod.CollisionObject(model, fcl_mod.Transform()), n_faces


def worker_bool_pair(fcl_mod: Any, obj_a: object, n_a: int, obj_b: object, n_b: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "fcl_status": "OK",
        "fcl_error": "",
        "collision_detected": np.nan,
        "num_contacts": np.nan,
        "count_mode": "boolean",
        "num_contacts_saturated": False,
        "total_faces_A": int(n_a),
        "total_faces_B": int(n_b),
        "contact_index_failures": 0,
    }
    try:
        req = fcl_mod.CollisionRequest(num_max_contacts=1, enable_contact=False)
        res = fcl_mod.CollisionResult()
        ret = fcl_mod.collide(obj_a, obj_b, req, res)
        detected = bool(ret)
        is_collision = getattr(res, "is_collision", None)
        if isinstance(is_collision, bool):
            detected = bool(detected or is_collision)
        out.update(collision_detected=bool(detected), num_contacts=int(ret))
    except Exception as exc:
        out.update(fcl_status="error", fcl_error=repr(exc), collision_detected=np.nan)
    return out


def worker_full_pair_with_cap(
    fcl_mod: Any,
    obj_a: object,
    n_a: int,
    obj_b: object,
    n_b: int,
    cap: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "fcl_status": "OK",
        "fcl_error": "",
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
        "count_mode": "all_contacts",
        "max_contacts_used": int(cap),
        "face_ids_A": [],
        "face_ids_B": [],
    }
    try:
        req = fcl_mod.CollisionRequest(num_max_contacts=int(cap), enable_contact=True)
        res = fcl_mod.CollisionResult()
        fcl_mod.collide(obj_a, obj_b, req, res)
        contacts = list(getattr(res, "contacts", []))
        n_contacts = int(len(contacts))
        if n_contacts == 0:
            out.update(
                collision_detected=False,
                num_contacts=0,
                num_contacts_saturated=False,
                intersecting_faces_A=0,
                intersecting_faces_B=0,
                pct_faces_A=0.0,
                pct_faces_B=0.0,
            )
            return out
        faces_a: set[int] = set()
        faces_b: set[int] = set()
        index_failures = 0
        for c in contacts:
            try:
                faces_a.add(int(c.b1))
                faces_b.add(int(c.b2))
            except Exception:
                index_failures += 1
        ids_a = sorted(faces_a)
        ids_b = sorted(faces_b)
        int_a = int(len(ids_a))
        int_b = int(len(ids_b))
        out.update(
            collision_detected=True,
            num_contacts=n_contacts,
            num_contacts_saturated=bool(n_contacts >= int(cap)),
            intersecting_faces_A=int_a,
            intersecting_faces_B=int_b,
            pct_faces_A=float(int_a / int(n_a) * 100.0) if int(n_a) > 0 else np.nan,
            pct_faces_B=float(int_b / int(n_b) * 100.0) if int(n_b) > 0 else np.nan,
            contact_index_failures=int(index_failures),
            face_ids_A=ids_a,
            face_ids_B=ids_b,
        )
        if n_contacts > 0 and int_a == 0 and int_b == 0:
            out["fcl_status"] = "contacts_without_indices"
            out["fcl_error"] = "FCL returned contacts but no usable face indices."
    except Exception as exc:
        out.update(fcl_status="error", fcl_error=repr(exc), collision_detected=np.nan)
    return out


def worker_full_pair_ladder(
    fcl_mod: Any,
    obj_a: object,
    n_a: int,
    obj_b: object,
    n_b: int,
    caps: Sequence[int],
) -> Dict[str, Any]:
    last: Optional[Dict[str, Any]] = None
    for cap in caps:
        result = worker_full_pair_with_cap(fcl_mod, obj_a, n_a, obj_b, n_b, int(cap))
        last = result
        ok = result.get("fcl_status") in {"OK", "contacts_without_indices"}
        saturated = to_bool_or_none(result.get("num_contacts_saturated", True)) is True
        if ok and not saturated:
            result["count_status"] = "OK"
            result["count_error"] = ""
            result["contact_count_exact"] = True
            return result
    if last is None:
        last = {
            "fcl_status": "error",
            "fcl_error": "empty cap ladder",
            "count_status": "not_run",
            "count_error": "empty cap ladder",
            "num_contacts_saturated": True,
            "contact_count_exact": False,
            "face_ids_A": [],
            "face_ids_B": [],
        }
    else:
        last["count_status"] = "OK" if last.get("fcl_status") in {"OK", "contacts_without_indices"} else "count_error"
        last["count_error"] = "" if last["count_status"] == "OK" else str(last.get("fcl_error", ""))
        last["contact_count_exact"] = False
    return last


def pair_row_base(
    method: str,
    mesh_set: str,
    dataset: str,
    case_id: str,
    subject: str,
    session: str,
    pair_name: str,
    surface_a: str,
    surface_b: str,
    path_a: str,
    path_b: str,
) -> Dict[str, Any]:
    return {
        "schema_version": COLLISION_SCHEMA_VERSION,
        "method": method,
        "mesh_set": mesh_set,
        "dataset": dataset,
        "case_id": case_id,
        "subject": subject,
        "session": session,
        "pair": pair_name,
        "surface_A": surface_a,
        "surface_B": surface_b,
        "path_A": path_a,
        "path_B": path_b,
    }


def _face_count_mismatch(parent_count: Any, worker_count: Any) -> Any:
    try:
        if pd.isna(parent_count) or pd.isna(worker_count):
            return pd.NA
        return int(parent_count) != int(worker_count)
    except Exception:
        return pd.NA


def build_union_row_from_pair_rows(
    *,
    method: str,
    mesh_set: str,
    dataset: str,
    case_id: str,
    subject: str,
    session: str,
    total_faces: Dict[str, int],
    union_faces: Dict[str, set[int]],
    pair_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "schema_version": COLLISION_SCHEMA_VERSION,
        "method": method,
        "mesh_set": mesh_set,
        "dataset": dataset,
        "case_id": case_id,
        "subject": subject,
        "session": session,
    }
    pcts: List[float] = []
    counts: List[int] = []
    for s in SURFACE_KEYS:
        n_total = int(total_faces.get(s, 0))
        n_union = int(len(union_faces.get(s, set())))
        pct = float(n_union / n_total * 100.0) if n_total > 0 else float("nan")
        row[f"{s}_faces_total"] = n_total
        row[f"{s}_faces_colliding_union"] = n_union
        row[f"{s}_collision_pct_union"] = pct
        pcts.append(pct)
        counts.append(n_union)
    detected = [to_bool_or_none(r.get("collision_detected")) is True for r in pair_rows]
    saturated = [to_bool_or_none(r.get("num_contacts_saturated")) is True for r in pair_rows]
    nonexact_true = [d and (to_bool_or_none(r.get("contact_count_exact")) is not True) for d, r in zip(detected, pair_rows)]
    idx_fail = [safe_float_or_nan(r.get("contact_index_failures", 0)) for r in pair_rows]
    contacts = [safe_float_or_nan(r.get("num_contacts", 0)) for r in pair_rows]
    total_idx_fail = int(np.nansum([v if np.isfinite(v) else 0 for v in idx_fail]))
    row.update(
        collision_pct_union_mean4=float(np.nanmean(pcts)),
        collision_pct_union_max4=float(np.nanmax(pcts)),
        collision_faces_union_sum4=int(np.nansum(counts)),
        n_pair_rows=int(len(pair_rows)),
        n_collision_true_pairs=int(sum(detected)),
        n_saturated_pairs=int(sum(saturated)),
        n_nonexact_true_pairs=int(sum(nonexact_true)),
        total_contacts_sum=float(np.nansum([v if np.isfinite(v) else 0 for v in contacts])),
        total_contact_index_failures=total_idx_fail,
        union_contact_count_exact_all_pairs=bool(sum(saturated) == 0 and sum(nonexact_true) == 0 and total_idx_fail == 0),
        union_status="OK",
        union_error="",
    )
    if len(pair_rows) != len(COLLISION_PAIRS):
        row["union_status"] = "problem"
        row["union_error"] = f"expected {len(COLLISION_PAIRS)} pair rows, got {len(pair_rows)}"
    elif row["n_saturated_pairs"] or row["n_nonexact_true_pairs"] or row["total_contact_index_failures"]:
        row["union_status"] = "problem"
        row["union_error"] = "union counts may be inexact because at least one pair was saturated/nonexact or had contact-index failures"
    return row


def worker_evaluate_case(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import resource
        mem_gb = int(payload.get("mem_gb", 0))
        if mem_gb > 0:
            limit = mem_gb * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception as exc:
        print(f"WARNING: worker memory limit was not applied: {exc!r}", file=sys.stderr, flush=True)

    if not HAS_FCL:
        raise RuntimeError(f"FCL backend is unavailable: {FCL_IMPORT_ERROR}")
    try:
        import fcl as fcl_mod  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Could not import python-fcl in worker: {exc!r}") from exc

    method = str(payload["method"])
    mesh_set = str(payload.get("mesh_set", "pred"))
    dataset = str(payload["dataset"])
    case_id = str(payload["case_id"])
    subject = str(payload["subject"])
    session = str(payload.get("session", ""))
    paths = {s: str(payload["paths"][s]) for s in SURFACE_KEYS}
    caps = [int(x) for x in payload["caps"]]

    meshes: Dict[str, trimesh.Trimesh] = {}
    objects: Dict[str, object] = {}
    fcl_faces: Dict[str, int] = {}
    total_faces: Dict[str, int] = {}

    for s in SURFACE_KEYS:
        tri = load_tri(Path(paths[s]))
        meshes[s] = tri
        obj, n_faces = make_fcl_object_worker(fcl_mod, tri)
        objects[s] = obj
        fcl_faces[s] = int(n_faces)
        total_faces[s] = int(len(tri.faces))

    pair_rows: List[Dict[str, Any]] = []
    union_faces: Dict[str, set[int]] = {s: set() for s in SURFACE_KEYS}

    for pair_name, (surface_a, surface_b) in COLLISION_PAIRS.items():
        base = pair_row_base(method, mesh_set, dataset, case_id, subject, session, pair_name, surface_a, surface_b, paths[surface_a], paths[surface_b])
        obj_a, obj_b = objects[surface_a], objects[surface_b]
        n_a, n_b = total_faces[surface_a], total_faces[surface_b]
        fcl_n_a, fcl_n_b = fcl_faces[surface_a], fcl_faces[surface_b]

        bool_result = worker_bool_pair(fcl_mod, obj_a, fcl_n_a, obj_b, fcl_n_b)
        detected = to_bool_or_none(bool_result.get("collision_detected"))
        row = dict(base)
        row.update(
            bool_fcl_status=bool_result.get("fcl_status", ""),
            bool_fcl_error=bool_result.get("fcl_error", ""),
            collision_detected=detected,
            bool_num_contacts=bool_result.get("num_contacts", np.nan),
            bool_count_mode=bool_result.get("count_mode", "boolean"),
            fcl_object_faces_A=int(fcl_n_a),
            fcl_object_faces_B=int(fcl_n_b),
            total_faces_A=int(n_a),
            total_faces_B=int(n_b),
        )

        if detected is False:
            row.update(
                count_status="not_needed_no_collision",
                count_error="",
                num_contacts=0,
                num_contacts_saturated=False,
                intersecting_faces_A=0,
                intersecting_faces_B=0,
                pct_faces_A=0.0,
                pct_faces_B=0.0,
                count_total_faces_A=int(n_a),
                count_total_faces_B=int(n_b),
                face_count_mismatch_A=False,
                face_count_mismatch_B=False,
                contact_index_failures=0,
                contact_count_exact=True,
                max_contacts_used=0,
                worker_returncode=0,
                worker_stderr_tail="",
                traceback="",
            )
            pair_rows.append(row)
            continue

        if detected is None:
            row.update(
                count_status="bool_error",
                count_error=bool_result.get("fcl_error", "unknown boolean FCL error"),
                num_contacts=np.nan,
                num_contacts_saturated=np.nan,
                intersecting_faces_A=np.nan,
                intersecting_faces_B=np.nan,
                pct_faces_A=np.nan,
                pct_faces_B=np.nan,
                count_total_faces_A=np.nan,
                count_total_faces_B=np.nan,
                face_count_mismatch_A=pd.NA,
                face_count_mismatch_B=pd.NA,
                contact_index_failures=np.nan,
                contact_count_exact=False,
                max_contacts_used=0,
                worker_returncode=0,
                worker_stderr_tail="",
                traceback="",
            )
            pair_rows.append(row)
            continue

        full = worker_full_pair_ladder(fcl_mod, obj_a, fcl_n_a, obj_b, fcl_n_b, caps)
        face_ids_a = set(int(x) for x in full.get("face_ids_A", []) if int(x) >= 0)
        face_ids_b = set(int(x) for x in full.get("face_ids_B", []) if int(x) >= 0)
        union_faces[surface_a].update(face_ids_a)
        union_faces[surface_b].update(face_ids_b)

        intersecting_a = len(face_ids_a)
        intersecting_b = len(face_ids_b)
        row.update(
            count_status=full.get("count_status", ""),
            count_error=full.get("count_error", full.get("fcl_error", "")),
            num_contacts=full.get("num_contacts", np.nan),
            num_contacts_saturated=full.get("num_contacts_saturated", np.nan),
            intersecting_faces_A=intersecting_a,
            intersecting_faces_B=intersecting_b,
            pct_faces_A=pct_from_count(intersecting_a, n_a),
            pct_faces_B=pct_from_count(intersecting_b, n_b),
            worker_pct_faces_A=full.get("pct_faces_A", np.nan),
            worker_pct_faces_B=full.get("pct_faces_B", np.nan),
            count_total_faces_A=full.get("total_faces_A", np.nan),
            count_total_faces_B=full.get("total_faces_B", np.nan),
            face_count_mismatch_A=_face_count_mismatch(n_a, full.get("total_faces_A", np.nan)),
            face_count_mismatch_B=_face_count_mismatch(n_b, full.get("total_faces_B", np.nan)),
            contact_index_failures=full.get("contact_index_failures", np.nan),
            contact_count_exact=to_bool_or_none(full.get("contact_count_exact", False)) is True,
            max_contacts_used=full.get("max_contacts_used", np.nan),
            worker_returncode=0,
            worker_stderr_tail="",
            traceback="",
        )
        pair_rows.append(row)

    union_row = build_union_row_from_pair_rows(
        method=method,
        mesh_set=mesh_set,
        dataset=dataset,
        case_id=case_id,
        subject=subject,
        session=session,
        total_faces=total_faces,
        union_faces=union_faces,
        pair_rows=pair_rows,
    )
    return {"pair_rows": pair_rows, "union_row": union_row}


def worker_case_main(args: argparse.Namespace) -> None:
    if not args.worker_json:
        raise SystemExit("--worker-json is required in --worker-case mode")
    in_json = Path(args.worker_json)
    out_json = Path(os.environ.get("STAGE07_WORKER_OUTPUT_JSON", str(in_json.with_suffix(".out.json"))))
    payload = json.loads(in_json.read_text(encoding="utf-8"))
    result = worker_evaluate_case(payload)
    out_json.write_text(json.dumps(safe_json_value(result), indent=2, allow_nan=False), encoding="utf-8")


def run_case_worker(payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage07_case_") as td:
        td_path = Path(td)
        in_json = td_path / "case_input.json"
        out_json = td_path / "case_output.json"
        in_json.write_text(json.dumps(safe_json_value(payload), indent=2, allow_nan=False), encoding="utf-8")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-case",
            "--worker-json",
            str(in_json),
        ]
        env = os.environ.copy()
        env["STAGE07_WORKER_OUTPUT_JSON"] = str(out_json)
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=int(timeout_sec),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"case worker timeout after {timeout_sec}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"case worker failed returncode={proc.returncode}\nSTDOUT:\n{proc.stdout[-STDERR_TAIL_CHARS:]}\n"
                f"STDERR:\n{proc.stderr[-STDERR_TAIL_CHARS:]}"
            )
        if not out_json.exists():
            raise RuntimeError(
                f"case worker did not write output JSON. STDOUT:\n{proc.stdout[-STDERR_TAIL_CHARS:]}\n"
                f"STDERR:\n{proc.stderr[-STDERR_TAIL_CHARS:]}"
            )
        result = json.loads(out_json.read_text(encoding="utf-8"))
        # Attach stderr tail to pair rows for diagnostics.
        for r in result.get("pair_rows", []):
            r["worker_stderr_tail"] = str(proc.stderr[-STDERR_TAIL_CHARS:])
        return result



def validate_manifest_columns(pred: pd.DataFrame) -> None:
    pred_required = {
        "method",
        "dataset",
        "case_id",
        "subject",
        "session",
        "surface",
        "status",
        "pred_path",
    }
    pred_missing = sorted(
        pred_required - set(pred.columns)
    )
    if pred_missing:
        raise ValueError(
            "Prediction manifest is missing columns: "
            f"{pred_missing}"
        )




def build_maps(pred: pd.DataFrame):
    ok_pred = pred[
        (pred["method"].astype(str) == METHOD_NAME)
        & (pred["status"].astype(str) == "OK")
    ].copy()

    pred_key_cols = [
        "method",
        "dataset",
        "case_id",
        "surface",
    ]

    pred_dups = ok_pred[
        ok_pred.duplicated(
            pred_key_cols,
            keep=False,
        )
    ].copy()

    if len(pred_dups):
        sample = pred_dups[
            pred_key_cols + ["pred_path"]
        ].head(20).to_dict(
            orient="records"
        )
        raise ValueError(
            "Duplicate OK prediction rows for keys "
            f"{pred_key_cols}; sample={sample}"
        )

    pred_map = {}

    for row in ok_pred.itertuples(
        index=False
    ):
        pred_map[
            (
                row.method,
                row.dataset,
                row.case_id,
                row.surface,
            )
        ] = Path(row.pred_path)

    return pred_map




def select_pred_cases(
    pred: pd.DataFrame,
    datasets: Optional[Iterable[str]],
    max_cases_per_dataset: Optional[int],
) -> pd.DataFrame:
    if (
        max_cases_per_dataset is not None
        and int(max_cases_per_dataset) <= 0
    ):
        raise ValueError(
            "--max-cases-per-dataset must be positive "
            "when provided."
        )

    df = pred[
        (pred["method"].astype(str) == METHOD_NAME)
        & (pred["status"].astype(str) == "OK")
    ].copy()

    if datasets is not None:
        df = df[
            df["dataset"].isin(
                list(datasets)
            )
        ].copy()

    cases = (
        df[
            [
                "method",
                "dataset",
                "case_id",
                "subject",
                "session",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "method",
                "dataset",
                "case_id",
            ]
        )
        .reset_index(drop=True)
    )

    if max_cases_per_dataset is not None:
        cases = (
            cases.groupby(
                ["method", "dataset"],
                group_keys=False,
            )
            .head(
                int(
                    max_cases_per_dataset
                )
            )
            .reset_index(drop=True)
        )

    if cases.empty:
        return cases

    selected_keys = cases[
        [
            "method",
            "dataset",
            "case_id",
        ]
    ].drop_duplicates()

    selected_rows = df.merge(
        selected_keys,
        on=[
            "method",
            "dataset",
            "case_id",
        ],
        how="inner",
    )

    expected_surfaces = set(
        SURFACE_KEYS
    )

    per_case = (
        selected_rows.groupby(
            [
                "method",
                "dataset",
                "case_id",
            ],
            dropna=False,
        )
        .agg(
            n_rows=(
                "surface",
                "size",
            ),
            n_unique_surfaces=(
                "surface",
                "nunique",
            ),
            surfaces=(
                "surface",
                lambda x: ",".join(
                    sorted(
                        str(v)
                        for v in set(x)
                    )
                ),
            ),
            n_subjects=(
                "subject",
                "nunique",
            ),
            n_sessions=(
                "session",
                lambda x: (
                    x.fillna("")
                    .astype(str)
                    .nunique()
                ),
            ),
        )
        .reset_index()
    )

    expected_surfaces_str = ",".join(
        sorted(expected_surfaces)
    )

    bad = per_case[
        (
            per_case["n_rows"]
            != len(SURFACE_KEYS)
        )
        | (
            per_case["n_unique_surfaces"]
            != len(SURFACE_KEYS)
        )
        | (
            per_case["surfaces"]
            != expected_surfaces_str
        )
        | (
            per_case["n_subjects"]
            != 1
        )
        | (
            per_case["n_sessions"]
            != 1
        )
    ].copy()

    if len(bad):
        sample = bad.head(20).to_dict(
            orient="records"
        )
        raise ValueError(
            "Selected cases must have exactly one "
            "subject/session and the four expected "
            f"OK surfaces {sorted(expected_surfaces)}; "
            f"bad_count={len(bad)} sample={sample}"
        )

    return cases



def get_paths_for_pred_case(row, pred_map):
    pred_paths = {}
    for surface in SURFACE_KEYS:
        pred_key = (row.method, row.dataset, row.case_id, surface)
        if pred_key not in pred_map:
            raise FileNotFoundError(f"Missing pred manifest row: {pred_key}")
        pred_paths[surface] = pred_map[pred_key]
        if not pred_paths[surface].exists():
            raise FileNotFoundError(f"Missing pred file: {pred_paths[surface]}")
    return pred_paths




def summarize_numeric(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols)
    available = [c for c in value_cols if c in df.columns]
    if not available:
        return df[group_cols].drop_duplicates().reset_index(drop=True)
    agg = df.groupby(group_cols, dropna=False)[available].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    agg.columns = flatten_columns(agg.columns)
    # Add an explicit n_cases column for readability. The union case CSV is one row per completed case.
    n_cases = df.groupby(group_cols, dropna=False).size().reset_index(name="n_cases")
    return n_cases.merge(agg, on=group_cols, how="left")


def safe_write_excel(out_path: Path, sheets: Dict[str, pd.DataFrame]) -> bool:
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
        return True
    except ImportError:
        logging.warning("openpyxl is not installed; skipping Excel output: %s", out_path)
        return False


def write_outputs(
    *,
    pair_csv: Path,
    fail_csv: Path,
    union_case_csv: Path,
    pred_collision_rows: List[Dict[str, Any]],
    union_case_rows: List[Dict[str, Any]],
    missing_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = dataframe_with_columns(pred_collision_rows, COLLISION_COLUMNS)
    union_df = dataframe_with_columns(union_case_rows, UNION_COLUMNS)
    miss_df = dataframe_with_columns(missing_rows, MISSING_COLUMNS)
    pred_df.to_csv(pair_csv, index=False, encoding="utf-8")
    union_df.to_csv(union_case_csv, index=False, encoding="utf-8")
    miss_df.to_csv(fail_csv, index=False, encoding="utf-8")
    return pred_df, union_df, miss_df


def build_pair_strict_problems(
    pred_df: pd.DataFrame,
    miss_df: pd.DataFrame,
    expected_pred_pair_rows: Optional[int] = None,
) -> List[str]:
    problems: List[str] = []
    if len(miss_df) > 0:
        problems.append(f"missing_or_failed={len(miss_df)}")
    if expected_pred_pair_rows is not None and len(pred_df) != int(expected_pred_pair_rows):
        problems.append(f"pred_pair_rows={len(pred_df)} expected_from_succeeded_cases={int(expected_pred_pair_rows)}")
    if pred_df.empty:
        problems.append("pred_collision_rows_empty")
        return problems
    pred_true = true_collision_mask(pred_df)
    pred_saturated = bool_count(pred_df, "num_contacts_saturated")
    if pred_saturated:
        problems.append(f"pred_saturated_rows={pred_saturated}")
    status = pred_df.get("count_status", pd.Series(dtype=object)).astype(str)
    pred_bool_errors = int((status == "bool_error").sum())
    pred_generic_errors = int((status == "error").sum())
    pred_worker_errors = int(status.isin(["worker_error", "timeout", "json_parse_error", "not_run", "worker_cap_mismatch", "count_error"]).sum())
    if pred_bool_errors:
        problems.append(f"pred_bool_error_rows={pred_bool_errors}")
    if pred_generic_errors:
        problems.append(f"pred_error_rows={pred_generic_errors}")
    if pred_worker_errors:
        problems.append(f"pred_worker_or_timeout_rows={pred_worker_errors}")
    if "count_status" in pred_df.columns:
        bad_true_count = int((pred_true & (pred_df["count_status"].astype(str) != "OK")).sum())
        if bad_true_count:
            problems.append(f"true_collision_rows_with_non_OK_count_status={bad_true_count}")
    if "contact_count_exact" in pred_df.columns:
        nonexact_true = int((pred_true & (pred_df["contact_count_exact"].map(lambda x: to_bool_or_none(x) is not True))).sum())
        if nonexact_true:
            problems.append(f"true_collision_rows_not_exact={nonexact_true}")
    for col in ["pct_faces_A", "pct_faces_B"]:
        vals = numeric_series(pred_df, col)
        bad_nonfinite = int((pred_true & ~np.isfinite(vals)).sum())
        if bad_nonfinite:
            problems.append(f"true_collision_rows_with_nonfinite_{col}={bad_nonfinite}")
        bad_range = int((pred_true & np.isfinite(vals) & ((vals < 0.0) | (vals > 100.0))).sum())
        if bad_range:
            problems.append(f"true_collision_rows_with_out_of_range_{col}={bad_range}")
    for col in ["total_faces_A", "total_faces_B"]:
        vals = numeric_series(pred_df, col)
        bad_faces = int((~np.isfinite(vals) | (vals <= 0)).sum())
        if bad_faces:
            problems.append(f"pred_rows_with_invalid_{col}={bad_faces}")
    for col in ["face_count_mismatch_A", "face_count_mismatch_B"]:
        if col in pred_df.columns:
            mismatches = bool_count(pred_df, col)
            if mismatches:
                problems.append(f"{col}={mismatches}")
    if "contact_index_failures" in pred_df.columns:
        idx_fail = numeric_series(pred_df, "contact_index_failures").fillna(0)
        n_idx_fail = int((idx_fail > 0).sum())
        if n_idx_fail:
            problems.append(f"contact_index_failures_positive_rows={n_idx_fail}")
    if "num_contacts" in pred_df.columns:
        contacts = numeric_series(pred_df, "num_contacts")
        bad_contacts = int((pred_true & (~np.isfinite(contacts) | (contacts < 0))).sum())
        if bad_contacts:
            problems.append(f"true_collision_rows_with_invalid_num_contacts={bad_contacts}")
    return problems


def build_union_strict_problems(
    union_df: pd.DataFrame,
    expected_union_case_rows: Optional[int] = None,
) -> List[str]:
    problems: List[str] = []
    if expected_union_case_rows is not None and len(union_df) != int(expected_union_case_rows):
        problems.append(f"union_case_rows={len(union_df)} expected={int(expected_union_case_rows)}")
    if union_df.empty:
        problems.append("union_case_rows_empty")
        return problems
    if "union_status" in union_df.columns:
        bad = int((union_df["union_status"].astype(str) != "OK").sum())
        if bad:
            problems.append(f"union_status_problem_rows={bad}")
    for col in [f"{s}_collision_pct_union" for s in SURFACE_KEYS] + ["collision_pct_union_mean4", "collision_pct_union_max4"]:
        vals = numeric_series(union_df, col)
        bad_nonfinite = int((~np.isfinite(vals)).sum())
        if bad_nonfinite:
            problems.append(f"union_nonfinite_{col}={bad_nonfinite}")
        bad_range = int((np.isfinite(vals) & ((vals < 0.0) | (vals > 100.0))).sum())
        if bad_range:
            problems.append(f"union_out_of_range_{col}={bad_range}")
    for col in ["n_saturated_pairs", "n_nonexact_true_pairs", "total_contact_index_failures"]:
        vals = numeric_series(union_df, col).fillna(0)
        bad = int((vals > 0).sum())
        if bad:
            problems.append(f"union_rows_with_positive_{col}={bad}")
    return problems


def setup_logging(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval_fcl_collision.log"
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def log_print(*parts: Any) -> None:
    logging.info(" ".join(str(p) for p in parts))



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute SimCortex FCL collision and "
            "surface-union metrics."
        )
    )

    ap.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help=(
            "Root directory for evaluation inputs and outputs. "
            "Required for normal evaluation mode."
        ),
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
    )
    ap.add_argument(
        "--max-cases-per-dataset",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--eval-set",
        default="sample40",
    )
    ap.add_argument(
        "--pred-manifest",
        type=Path,
        default=None,
        help="Default: <eval-root>/manifests/pred_manifest.tsv.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <eval-root>/collisions.",
    )

    # Historical operational defaults are preserved exactly.
    ap.add_argument(
        "--max-contact-ladder",
        default="50000,200000,500000",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=600,
    )
    ap.add_argument(
        "--mem-gb",
        type=int,
        default=48,
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--slow-case-warn-sec",
        type=float,
        default=1800.0,
        help=(
            "Log a warning if one case exceeds this "
            "duration; use 0 to disable."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
    )

    # Internal worker interface used by run_case_worker().
    ap.add_argument(
        "--worker-case",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--worker-json",
        default="",
        help=argparse.SUPPRESS,
    )

    args = ap.parse_args()

    if args.worker_case:
        return args

    if args.eval_root is None:
        ap.error(
            "--eval-root is required unless "
            "--worker-case is used"
        )

    eval_root = (
        args.eval_root
        .expanduser()
        .resolve()
    )

    if args.pred_manifest is None:
        args.pred_manifest = (
            eval_root
            / "manifests"
            / "pred_manifest.tsv"
        )
    else:
        args.pred_manifest = (
            args.pred_manifest
            .expanduser()
            .resolve()
        )

    if args.out_dir is None:
        args.out_dir = (
            eval_root
            / "collisions"
        )
    else:
        args.out_dir = (
            args.out_dir
            .expanduser()
            .resolve()
        )

    args.eval_root = eval_root
    return args



def main() -> None:
    args = parse_args()
    if args.worker_case:
        worker_case_main(args)
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out_dir)

    try:
        caps = parse_cap_ladder(args.max_contact_ladder)
        if int(args.timeout_sec) <= 0:
            raise ValueError("--timeout-sec must be positive")
        if int(args.mem_gb) <= 0:
            raise ValueError("--mem-gb must be positive")
        if int(args.save_every) < 0:
            raise ValueError("--save-every must be non-negative")
        if args.max_cases_per_dataset is not None and int(args.max_cases_per_dataset) <= 0:
            raise ValueError("--max-cases-per-dataset must be positive when provided")
    except Exception as exc:
        logging.error("Invalid arguments: %s", exc)
        raise SystemExit(2) from exc

    pred_manifest = Path(args.pred_manifest)
    if not pred_manifest.exists():
        raise SystemExit(f"Missing pred manifest: {pred_manifest}")

    try:
        pred = pd.read_csv(pred_manifest, sep="\t", low_memory=False)
        validate_manifest_columns(pred)
        pred_map = build_maps(pred)
        cases = select_pred_cases(
            pred,
            args.datasets,
            args.max_cases_per_dataset,
        )
    except Exception as exc:
        logging.error("Manifest preparation failed: %s", exc)
        logging.error("%s", traceback.format_exc())
        raise SystemExit(1) from exc
    if cases.empty:
        raise SystemExit("No SimCortex prediction cases selected. Check --datasets and prediction manifest status.")

    pair_csv = out_dir / "collision_pair_metrics_long.csv"
    union_case_csv = out_dir / "collision_surface_union_case_level.csv"
    union_by_ds_csv = out_dir / "collision_surface_union_by_method_dataset.csv"
    union_overall_csv = out_dir / "collision_surface_union_overall.csv"
    union_xlsx = out_dir / "collision_surface_union_summary.xlsx"
    fail_csv = out_dir / "collision_missing_or_failed.csv"
    run_json = out_dir / "collision_run_summary.json"

    existing_outputs = [
        p for p in [pair_csv, union_case_csv, union_by_ds_csv, union_overall_csv, union_xlsx, fail_csv, run_json] if p.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise SystemExit(f"Output file(s) already exist in {out_dir}: {[p.name for p in existing_outputs]}. Use --overwrite.")

    log_print("=== SimCortex FCL collision evaluation ===")
    log_print("schema_version:", COLLISION_SCHEMA_VERSION)
    log_print("eval_root:", args.eval_root)
    log_print("pred_manifest:", pred_manifest)
    log_print("method:", METHOD_NAME)
    log_print("datasets:", args.datasets)
    log_print("cases requested:", len(cases))
    log_print("HAS_FCL:", HAS_FCL)
    log_print("collision backend:", COLLISION_BACKEND_FILE)
    log_print("contact cap ladder:", caps)
    log_print("timeout_sec per case worker:", args.timeout_sec)
    log_print("mem_gb per case worker:", args.mem_gb)
    log_print("strict:", args.strict)
    log_print("out_dir:", out_dir)
    log_print("log_path:", log_path)

    if args.strict and not HAS_FCL:
        raise SystemExit("Strict mode: FCL backend is unavailable.")

    pred_collision_rows: List[Dict[str, Any]] = []
    union_case_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    t0 = time.time()
    attempted = 0
    succeeded = 0
    failed = 0

    for row in tqdm(cases.itertuples(index=False), total=len(cases), desc="FCL collision + union"):
        attempted += 1
        t_case = time.time()
        try:
            pred_paths = get_paths_for_pred_case(row, pred_map)
            payload = {
                "method": row.method,
                "mesh_set": "pred",
                "dataset": row.dataset,
                "case_id": row.case_id,
                "subject": row.subject,
                "session": "" if pd.isna(row.session) else str(row.session),
                "paths": {s: str(pred_paths[s]) for s in SURFACE_KEYS},
                "caps": caps,
                "mem_gb": int(args.mem_gb),
            }
            result = run_case_worker(payload, timeout_sec=int(args.timeout_sec))
            pred_collision_rows.extend(result.get("pair_rows", []))
            union_case_rows.append(result.get("union_row", {}))
            succeeded += 1
        except Exception as exc:
            failed += 1
            missing_rows.append(
                {
                    "method": row.method,
                    "dataset": row.dataset,
                    "case_id": row.case_id,
                    "subject": row.subject,
                    "session": row.session,
                    "status": "FAILED",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            case_elapsed = time.time() - t_case
            if float(args.slow_case_warn_sec) > 0 and case_elapsed > float(args.slow_case_warn_sec):
                logging.warning(
                    "Slow collision case: method=%s dataset=%s case_id=%s elapsed_sec=%.1f",
                    row.method,
                    row.dataset,
                    row.case_id,
                    case_elapsed,
                )
        if args.save_every > 0 and attempted % args.save_every == 0:
            write_outputs(
                pair_csv=pair_csv,
                fail_csv=fail_csv,
                union_case_csv=union_case_csv,
                pred_collision_rows=pred_collision_rows,
                union_case_rows=union_case_rows,
                missing_rows=missing_rows,
            )

    pred_df, union_df, miss_df = write_outputs(
        pair_csv=pair_csv,
        fail_csv=fail_csv,
        union_case_csv=union_case_csv,
        pred_collision_rows=pred_collision_rows,
        union_case_rows=union_case_rows,
        missing_rows=missing_rows,
    )

    union_by_ds_df = summarize_numeric(union_df, ["method", "dataset"], UNION_SUMMARY_VALUE_COLS)
    union_overall_df = summarize_numeric(union_df, ["method"], UNION_SUMMARY_VALUE_COLS)
    union_by_ds_df.to_csv(union_by_ds_csv, index=False, encoding="utf-8")
    union_overall_df.to_csv(union_overall_csv, index=False, encoding="utf-8")
    xlsx_written = safe_write_excel(
        union_xlsx,
        {
            "overall": union_overall_df,
            "by_dataset": union_by_ds_df,
            "case_level": union_df,
            "pair_rows": pred_df,
            "missing_or_failed": miss_df,
        },
    )

    expected_pred_pair_rows = int(succeeded * len(COLLISION_PAIRS))
    expected_union_case_rows = int(succeeded)
    missing_pred_pair_rows = max(0, expected_pred_pair_rows - int(len(pred_df)))
    extra_pred_pair_rows = max(0, int(len(pred_df)) - expected_pred_pair_rows)
    pred_pair_rows_match_expected = int(len(pred_df)) == expected_pred_pair_rows
    union_case_rows_match_expected = int(len(union_df)) == expected_union_case_rows

    pair_strict_problems = build_pair_strict_problems(pred_df, miss_df, expected_pred_pair_rows=expected_pred_pair_rows)
    union_strict_problems = build_union_strict_problems(union_df, expected_union_case_rows=expected_union_case_rows)
    strict_problems = pair_strict_problems + union_strict_problems

    pred_true = bool_count(pred_df, "collision_detected")
    pred_saturated = bool_count(pred_df, "num_contacts_saturated")
    union_problem_rows = int((union_df.get("union_status", pd.Series(dtype=object)).astype(str) != "OK").sum()) if not union_df.empty else 0

    summary = {
        "stage": "evaluate_collisions",
        "schema_version": COLLISION_SCHEMA_VERSION,
        "eval_root": str(args.eval_root),
        "out_dir": str(out_dir),
        "eval_set": args.eval_set,
        "method": METHOD_NAME,
        "datasets": args.datasets,
        "num_pred_cases_requested": int(len(cases)),
        "num_pred_cases_attempted": int(attempted),
        "num_pred_cases_succeeded": int(succeeded),
        "num_pred_cases_failed": int(failed),
        "num_pred_cases": int(succeeded),
        "num_pred_pair_rows": int(len(pred_df)),
        "expected_pred_pair_rows_from_succeeded_cases": int(expected_pred_pair_rows),
        "pred_pair_rows_match_expected": bool(pred_pair_rows_match_expected),
        "incomplete_pred_pair_row_count": int(missing_pred_pair_rows),
        "extra_pred_pair_row_count": int(extra_pred_pair_rows),
        "num_union_case_rows": int(len(union_df)),
        "expected_union_case_rows_from_succeeded_cases": int(expected_union_case_rows),
        "union_case_rows_match_expected": bool(union_case_rows_match_expected),
        "num_union_method_dataset_rows": int(len(union_by_ds_df)),
        "num_union_overall_rows": int(len(union_overall_df)),
        "num_union_problem_rows": int(union_problem_rows),
        "num_missing_or_failed": int(len(miss_df)),
        "contact_cap_ladder": caps,
        "timeout_sec": int(args.timeout_sec),
        "mem_gb": int(args.mem_gb),
        "slow_case_warn_sec": float(args.slow_case_warn_sec),
        "HAS_FCL": bool(HAS_FCL),
        "collision_backend_file": COLLISION_BACKEND_FILE,
        "pred_collision_true_rows": int(pred_true),
        "pred_saturated_rows": int(pred_saturated),
        "xlsx_written": bool(xlsx_written),
        "strict": bool(args.strict),
        "strict_problems": strict_problems,
        "definitions": {
            "pairwise_collision": "Six anatomical surface pairs are evaluated for each predicted case.",
            "surface_union_collision": (
                "For each surface, take the union of unique face IDs on that surface that collide with any of the "
                "other three surfaces; divide by that surface face count."
            ),
            "collision_pct_union_mean4": "Mean of the four surface-level union collision percentages within a case.",
        },
        "outputs": {
            "collision_pair_metrics_long": str(pair_csv),
            "collision_surface_union_case_level": str(union_case_csv),
            "collision_surface_union_by_method_dataset": str(union_by_ds_csv),
            "collision_surface_union_overall": str(union_overall_csv),
            "collision_surface_union_summary_xlsx": str(union_xlsx),
            "collision_missing_or_failed": str(fail_csv),
            "collision_run_summary": str(run_json),
            "eval_fcl_collision_log": str(log_path),
        },
        "elapsed_min": float((time.time() - t0) / 60.0),
    }
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=safe_json_value)

    summary_text = json.dumps(summary, indent=2, default=safe_json_value)
    print(summary_text)
    logging.info("\n%s", summary_text)
    log_print("Saved:", pair_csv)
    log_print("Saved:", union_case_csv)
    log_print("Saved:", union_overall_csv)
    log_print("Saved:", fail_csv)
    log_print("Saved:", run_json)

    if args.strict and strict_problems:
        raise SystemExit("Strict collision evaluation failed: " + "; ".join(strict_problems))


if __name__ == "__main__":
    main()
