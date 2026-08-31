#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import trimesh
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds

LOG = logging.getLogger("metrics_core")

METRICS_SCHEMA_VERSION = "2.0"

SURFACE_KEYS = ("lh_white", "lh_pial", "rh_white", "rh_pial")
SURFACE_PARTS = {
    "lh_white": ("lh", "white"),
    "lh_pial": ("lh", "pial"),
    "rh_white": ("rh", "white"),
    "rh_pial": ("rh", "pial"),
}

_PointFaceDistanceOP = _PointFaceDistance.apply

# --------------------------------------------------------------------------- #
# Optional-dependency detection
# --------------------------------------------------------------------------- #
try:
    import pymeshlab as pyml  # noqa: N813

    try:
        _AVAILABLE_FILTERS = set(pyml.filter_list())
    except Exception:  # plugins failed to load
        _AVAILABLE_FILTERS = set()
    HAS_PYMESHLAB = True
except Exception:
    pyml = None
    _AVAILABLE_FILTERS = set()
    HAS_PYMESHLAB = False

_SIF_SELECT_CANDIDATES = [
    "compute_selection_by_self_intersections_per_face",  # current PyMeshLab
    "select_self_intersecting_faces",                    # legacy PyMeshLab
]


def _first_available(candidates: List[str]) -> Optional[str]:

    if not candidates:
        return None
    if not _AVAILABLE_FILTERS:
        return candidates[0]
    for c in candidates:
        if c in _AVAILABLE_FILTERS:
            return c
    return None


_SIF_SELECT_FILTER = _first_available(_SIF_SELECT_CANDIDATES) if HAS_PYMESHLAB else None

HAS_SIF = bool(HAS_PYMESHLAB)

METRIC_DEFINITIONS = {
    "schema_version": METRICS_SCHEMA_VERSION,
    "ASSD_mm": "Symmetric mean sampled point-to-triangle distance (= point-to-surface Chamfer-L1).",
    "HD90_mm": "Symmetric 90th-percentile sampled point-to-triangle distance.",
    "ChamferPCL1_mm": "Symmetric mean point-cloud nearest-neighbour distance (sampled). "
                      "Point-cloud NN, not point-to-surface; upward-biased vs ASSD. Backward-compat only.",
    "SIF_pct": "Percentage of predicted faces flagged self-intersecting by PyMeshLab.",
    "thickness_*_mm": "Nearest-surface white->pial mean distance estimate (NOT anatomical correspondence).",
}


@dataclass(frozen=True)
class MeshData:
    verts_np: np.ndarray
    faces_np: np.ndarray
    p3d: Meshes
    tri: trimesh.Trimesh


# --------------------------------------------------------------------------- #
# Device / seeding / IO
# --------------------------------------------------------------------------- #
def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device_arg}")
    return device


def stable_hash_int(*parts: str, modulo: int = 2_000_000_000) -> int:
    text = "||".join(str(p) for p in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def set_seed(base_seed: int, *parts: str) -> None:
    """Deterministic seeding keyed on (base_seed, identifying parts)."""
    seed = (int(base_seed) + stable_hash_int(*parts)) % 2_000_000_000
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))


def load_mesh(path: Path, device: torch.device) -> MeshData:
    """Load FreeSurfer geometry, PLY/OBJ/STL, or NPZ mesh (process=False)."""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")

    if suffix in {"white", "pial"} or path.name.endswith(".pial.T1"):
        verts, faces = nib.freesurfer.io.read_geometry(str(path))
    elif suffix == "npz":
        data = np.load(path)
        verts, faces = data["vertices"], data["faces"]
    else:
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if not geoms:
                raise ValueError(f"Empty mesh scene: {path}")
            mesh = trimesh.util.concatenate(geoms)
        verts, faces = mesh.vertices, mesh.faces

    verts_np = np.asarray(verts, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)

    if verts_np.ndim != 2 or verts_np.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape {verts_np.shape}: {path}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"Invalid faces shape {faces_np.shape}: {path}")
    if not np.isfinite(verts_np).all():
        raise ValueError(f"Non-finite vertices in mesh: {path}")

    verts_t = torch.as_tensor(verts_np, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces_np, dtype=torch.int64, device=device)
    p3d = Meshes(verts=[verts_t], faces=[faces_t])
    tri = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    return MeshData(verts_np=verts_np, faces_np=faces_np, p3d=p3d, tri=tri)


# --------------------------------------------------------------------------- #
# Surface distance metrics
# --------------------------------------------------------------------------- #
def point_to_mesh_dist(pointcloud: Pointclouds, mesh: Meshes) -> torch.Tensor:
    points = pointcloud.points_packed()
    points_first_idx = pointcloud.cloud_to_packed_first_idx()
    max_points = pointcloud.num_points_per_cloud().max().item()

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    tris = verts[faces]
    tris_first_idx = mesh.mesh_to_faces_packed_first_idx()

    d2 = _PointFaceDistanceOP(points, points_first_idx, tris, tris_first_idx, max_points)
    return d2.clamp_min(0.0).sqrt()


def compute_main_surface_metrics(pred_mesh: Meshes, gt_mesh: Meshes, n_samples: int) -> Dict[str, float]:
    """ASSD/HD90 from point-to-triangle distances; ChamferPCL1 from point-cloud NN.

    Sampling is shared across the three metrics; seed must be set by the caller.
    """
    if pred_mesh.faces_packed().shape[0] == 0 or gt_mesh.faces_packed().shape[0] == 0:
        return {"ASSD_mm": float("nan"), "HD90_mm": float("nan"), "ChamferPCL1_mm": float("nan")}

    pred_pts = sample_points_from_meshes(pred_mesh, num_samples=int(n_samples))
    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=int(n_samples))

    d_p2g = point_to_mesh_dist(Pointclouds(pred_pts), gt_mesh)
    d_g2p = point_to_mesh_dist(Pointclouds(gt_pts), pred_mesh)

    nn_p2g = knn_points(pred_pts, gt_pts, K=1).dists[..., 0].clamp_min(0).sqrt()
    nn_g2p = knn_points(gt_pts, pred_pts, K=1).dists[..., 0].clamp_min(0).sqrt()

    return {
        "ASSD_mm": float(0.5 * (d_p2g.mean().item() + d_g2p.mean().item())),
        "HD90_mm": float(max(torch.quantile(d_p2g, 0.90).item(),
                             torch.quantile(d_g2p, 0.90).item())),
        "ChamferPCL1_mm": float(0.5 * (nn_p2g.mean().item() + nn_g2p.mean().item())),
    }


# --------------------------------------------------------------------------- #
# Self-intersection fraction (SIF)
# --------------------------------------------------------------------------- #
def _sif_candidate_order() -> List[str]:
    """Try the resolved filter first, then all known candidates without repeats."""
    ordered: List[str] = []
    for name in [_SIF_SELECT_FILTER, *_SIF_SELECT_CANDIDATES]:
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def _apply_pymeshlab_filter(ms: "pyml.MeshSet", filter_name: str) -> None:
    """Apply a PyMeshLab filter across old/new API variants."""
    try:
        ms.apply_filter(filter_name)
        return
    except Exception as apply_exc:
        # Some versions expose filters as direct MeshSet methods.
        if hasattr(ms, filter_name):
            try:
                getattr(ms, filter_name)()
                return
            except Exception:
                pass
        raise apply_exc


def _meshset_from_trimesh(mesh: trimesh.Trimesh) -> "pyml.MeshSet":
    ms = pyml.MeshSet()
    ms.add_mesh(
        pyml.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "mesh",
    )
    return ms


def compute_sif(mesh: trimesh.Trimesh) -> Dict[str, object]:
    out: Dict[str, object] = {
        "SIF_pct": float("nan"),
        "SIF_status": "OK",
        "SIF_error": "",
        "SIF_filter": "",
    }

    if not HAS_PYMESHLAB:
        out["SIF_status"] = "no_pymeshlab"
        return out

    if mesh.faces is None or len(mesh.faces) == 0:
        out["SIF_status"] = "no_faces"
        return out

    candidates = _sif_candidate_order()
    if not candidates:
        out["SIF_status"] = "no_sif_filter"
        out["SIF_error"] = "No PyMeshLab self-intersection selection filter candidates configured."
        return out

    tried_errors: List[str] = []
    for filter_name in candidates:
        try:
            ms = _meshset_from_trimesh(mesh)
            n_faces = int(ms.current_mesh().face_number())
            if n_faces <= 0:
                out["SIF_status"] = "no_faces"
                return out

            _apply_pymeshlab_filter(ms, filter_name)

            selected = np.asarray(ms.current_mesh().face_selection_array(), dtype=bool).reshape(-1)
            if selected.size != n_faces:
                raise RuntimeError(
                    f"face_selection_array size {selected.size} does not match face count {n_faces}"
                )

            out["SIF_pct"] = float(selected.sum() / n_faces * 100.0)
            out["SIF_status"] = "OK"
            out["SIF_error"] = ""
            out["SIF_filter"] = filter_name
            return out

        except Exception as exc:
            tried_errors.append(f"{filter_name}: {repr(exc)}")
            LOG.debug("SIF candidate failed (%s): %s", filter_name, exc)

    out["SIF_status"] = "error"
    out["SIF_error"] = " | ".join(tried_errors)
    out["SIF_filter"] = candidates[0] if candidates else ""
    return out


# --------------------------------------------------------------------------- #
# Thickness (nearest-surface estimate)
# --------------------------------------------------------------------------- #
def compute_mean_thickness(white_mesh: Meshes, pial_mesh: Meshes, n_samples: int) -> float:
    """Mean white->pial nearest-surface distance (NOT correspondence thickness)."""
    if white_mesh.faces_packed().shape[0] == 0 or pial_mesh.faces_packed().shape[0] == 0:
        return float("nan")
    white_pts = sample_points_from_meshes(white_mesh, num_samples=int(n_samples))
    d_w2p = point_to_mesh_dist(Pointclouds(white_pts), pial_mesh)
    return float(d_w2p.mean().item())

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_subject_meshes(
    *,
    method: str,
    eval_set: str,
    dataset: str,
    subject: str,
    session: str,
    pred_meshes: Dict[str, MeshData],
    gt_meshes: Dict[str, MeshData],
    pred_paths: Dict[str, Path],
    gt_paths: Dict[str, Path],
    n_samples: int,
    n_thickness_samples: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Evaluate one subject. Surface metrics (ASSD, HD90, ChamferPCL1, SIF) compare pred vs GT.
        Thickness computed for pred and GT.
    """
    surface_rows: List[Dict[str, object]] = []

    for skey in SURFACE_KEYS:
        set_seed(seed, method, dataset, subject, session, skey, "surface")
        metrics = compute_main_surface_metrics(
            pred_meshes[skey].p3d, gt_meshes[skey].p3d, n_samples
        )
        sif = compute_sif(pred_meshes[skey].tri)

        pred_centroid = pred_meshes[skey].verts_np.astype(np.float64).mean(axis=0)
        gt_centroid = gt_meshes[skey].verts_np.astype(np.float64).mean(axis=0)
        centroid_dist = float(np.linalg.norm(pred_centroid - gt_centroid))

        surface_rows.append({
            "schema_version": METRICS_SCHEMA_VERSION,
            "method": method,
            "eval_set": eval_set,
            "dataset": dataset,
            "subject": subject,
            "session": session,
            "surface": skey,
            "ASSD_mm": metrics["ASSD_mm"],
            "HD90_mm": metrics["HD90_mm"],
            "ChamferPCL1_mm": metrics["ChamferPCL1_mm"],
            "SIF_pct": sif["SIF_pct"],
            "SIF_status": sif["SIF_status"],
            "SIF_error": sif["SIF_error"],
            "SIF_filter": sif["SIF_filter"],
            "pred_gt_centroid_dist_mm": centroid_dist,
            "pred_n_vertices": int(len(pred_meshes[skey].verts_np)),
            "pred_n_faces": int(len(pred_meshes[skey].faces_np)),
            "gt_n_vertices": int(len(gt_meshes[skey].verts_np)),
            "gt_n_faces": int(len(gt_meshes[skey].faces_np)),
            "pred_path": str(pred_paths[skey]),
            "gt_path": str(gt_paths[skey]),
        })

    pair_row: Dict[str, object] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "method": method,
        "eval_set": eval_set,
        "dataset": dataset,
        "subject": subject,
        "session": session,
    }

    # Thickness (pred & GT) ------------------------------------------------- #
    for hemi in ("lh", "rh"):
        white = f"{hemi}_white"
        pial = f"{hemi}_pial"
        set_seed(seed, method, dataset, subject, session, hemi, "thickness_pred")
        pred_t = compute_mean_thickness(
            pred_meshes[white].p3d, pred_meshes[pial].p3d, n_thickness_samples
        )
        set_seed(seed, method, dataset, subject, session, hemi, "thickness_gt")
        gt_t = compute_mean_thickness(
            gt_meshes[white].p3d, gt_meshes[pial].p3d, n_thickness_samples
        )
        pair_row[f"{hemi}_thickness_pred_mean_mm"] = pred_t
        pair_row[f"{hemi}_thickness_gt_mean_mm"] = gt_t
        pair_row[f"{hemi}_thickness_abs_error_mm"] = (
            abs(pred_t - gt_t) if np.isfinite(pred_t) and np.isfinite(gt_t) else float("nan")
        )
        pair_row[f"{hemi}_thickness_bias_mm"] = (
            (pred_t - gt_t) if np.isfinite(pred_t) and np.isfinite(gt_t) else float("nan")
        )

    return surface_rows, pair_row
