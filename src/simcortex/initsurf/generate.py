from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Any
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import trimesh
from nibabel.affines import apply_affine
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt as edt,
    generate_binary_structure,
)
from scipy.special import expit
from skimage.filters import gaussian
from skimage.measure import label as compute_cc
from skimage.measure import marching_cubes
from tqdm.auto import tqdm
from trimesh.collision import CollisionManager

from simcortex.initsurf.paths import (
    out_anat_dir,
    out_surf_dir,
    seg9_dseg_path,
    t1_mni_path,
)
from simcortex.utils.tca import topology


log = logging.getLogger("simcortex.initsurf")
_TOPO_CORRECT = None


def setup_logger(log_dir: str, filename: str = "generate_initsurf.log") -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(processName)s - %(message)s",
        force=True,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)


def _close_root_handlers() -> None:
    root = logging.getLogger("")
    for handler in root.handlers[:]:
        try:
            handler.flush()
        finally:
            handler.close()
            root.removeHandler(handler)


def setup_subject_logger(log_dir: str, ds_key: str, subject_id: str) -> None:
    subj_log_dir = os.path.join(log_dir, "subjects")
    os.makedirs(subj_log_dir, exist_ok=True)
    safe_ds = str(ds_key).replace("/", "_")
    safe_sub = str(subject_id).replace("/", "_")
    log_file = os.path.join(subj_log_dir, f"{safe_ds}_{safe_sub}.log")

    _close_root_handlers()
    root = logging.getLogger("")
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s - %(message)s")
    )
    root.addHandler(fh)


def _write_json_atomic(path: str | Path, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(
        f".{out_path.name}.tmp-{os.getpid()}"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: str,
    dtype=np.float32,
) -> None:
    """Save a NIfTI image while preserving qform/sform consistently."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data, dtype=dtype)
    affine = np.asarray(affine, dtype=np.float64)
    img = nib.Nifti1Image(arr, affine)
    img.set_data_dtype(np.dtype(dtype))
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, str(out_p))


def _simcortex_version() -> str:
    try:
        return importlib_metadata.version("simcortex")
    except importlib_metadata.PackageNotFoundError:
        return "2.0.0"


def write_dataset_description(
    root: str,
    name: str = "sc-initsurf",
) -> None:
    path = os.path.join(root, "dataset_description.json")
    dd = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "SimCortex",
                "Version": _simcortex_version(),
                "Description": "Collision-free initial cortical surfaces in MNI152 space.",
            }
        ],
    }
    _write_json_atomic(path, dd)


def _is_affine_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-4) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=atol))


def _validate_seg9_values(seg: np.ndarray, subject_id: str) -> None:
    vals = np.unique(seg)
    bad = vals[(vals < 0) | (vals > 8)]
    if bad.size > 0:
        raise ValueError(
            f"[{subject_id}] seg9 contains invalid labels outside [0,8]: "
            f"{bad[:20].tolist()}"
        )


def expected_output_paths(out_root: str, subject_id: str, ses: str, space: str) -> List[str]:
    anat_dir = out_anat_dir(out_root, subject_id, ses=ses)
    surf_dir = out_surf_dir(out_root, subject_id, ses=ses)
    stem = f"{subject_id}_ses-{ses}_space-{space}"
    return [
        os.path.join(anat_dir, f"{stem}_desc-seg9_dseg_used.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-seg9_dseg_cleaned.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-lh_white_sdf.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-rh_white_sdf.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-lh_pial_sdf.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-rh_pial_sdf.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-ribbon_sdf.nii.gz"),
        os.path.join(anat_dir, f"{stem}_desc-ribbon_prob.nii.gz"),
        os.path.join(surf_dir, f"{stem}_hemi-L_white.surf.ply"),
        os.path.join(surf_dir, f"{stem}_hemi-R_white.surf.ply"),
        os.path.join(surf_dir, f"{stem}_hemi-L_pial.surf.ply"),
        os.path.join(surf_dir, f"{stem}_hemi-R_pial.surf.ply"),
        os.path.join(anat_dir, f"{stem}_desc-initsurf_qc.json"),
    ]


def outputs_complete(out_root: str, subject_id: str, ses: str, space: str) -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in expected_output_paths(out_root, subject_id, ses, space))


def _get_map(cfg_node, keys: Tuple[str, ...]) -> Optional[Dict[str, str]]:
    for k in keys:
        v = getattr(cfg_node, k, None)
        if v is not None and hasattr(v, "items"):
            return {str(kk): str(vv) for kk, vv in v.items()}
    return None


def _normalize_session_label(s: str) -> str:
    s = str(s).strip()
    return s[4:] if s.startswith("ses-") else s


def normalize_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        raise ValueError(f"split_file must contain column 'subject'. Got: {list(df.columns)}")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["subject"] = df["subject"].apply(lambda x: x if x.startswith("sub-") else f"sub-{x}")
    return df


def _validate_base_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")


def _validate_single_split_df(df: pd.DataFrame) -> None:
    _validate_base_split_df(df)
    if "dataset" in df.columns:
        vals = sorted(df["dataset"].dropna().astype(str).str.strip().unique().tolist())
        if len(vals) > 1:
            raise ValueError(
                "Single-dataset InitSurf received a split_file with multiple dataset values. "
                "Please provide a split CSV for one dataset only."
            )


def _validate_multi_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")


def _validate_unique_rows(df: pd.DataFrame, columns: List[str]) -> None:
    duplicates = df.duplicated(subset=columns, keep=False)
    if duplicates.any():
        rows = df.loc[duplicates, columns + ["split"]].head(20)
        raise ValueError(
            "Duplicate split rows found:\n" + rows.to_string(index=False)
        )


def _validate_params(params: Dict[str, Any]) -> None:
    required = {
        "gap_size",
        "sdf_sigma",
        "topo_threshold",
        "n_smooth",
        "wm_start_level",
        "wm_step",
        "wm_min_level",
        "pial_min_level",
        "pial_max_level",
        "pial_grid_step",
    }
    missing = sorted(required - set(params))
    if missing:
        raise KeyError(f"Missing InitSurf parameters: {missing}")

    if int(params["gap_size"]) < 0:
        raise ValueError("params.gap_size must be >= 0")
    if float(params["sdf_sigma"]) < 0:
        raise ValueError("params.sdf_sigma must be >= 0")
    if int(params["n_smooth"]) < 0:
        raise ValueError("params.n_smooth must be >= 0")
    if float(params.get("affine_atol", 1e-4)) < 0:
        raise ValueError("params.affine_atol must be >= 0")

    wm_start = float(params["wm_start_level"])
    wm_step = float(params["wm_step"])
    wm_min = float(params["wm_min_level"])
    wm_inset = float(params.get("wm_inset", 1.0))
    if wm_step >= 0:
        raise ValueError("params.wm_step must be negative")
    if wm_start < wm_min:
        raise ValueError("params.wm_start_level must be >= params.wm_min_level")
    if wm_inset < 0:
        raise ValueError("params.wm_inset must be >= 0")

    pial_min = float(params["pial_min_level"])
    pial_max = float(params["pial_max_level"])
    pial_step = float(params["pial_grid_step"])
    pial_floor = float(params.get("pial_absolute_floor", 0.1))
    if pial_step <= 0:
        raise ValueError("params.pial_grid_step must be positive")
    if pial_min > pial_max:
        raise ValueError("params.pial_min_level must be <= params.pial_max_level")
    if pial_floor < 0 or pial_floor > pial_min:
        raise ValueError(
            "params.pial_absolute_floor must be in [0, params.pial_min_level]"
        )

    numeric_values = [
        float(params["sdf_sigma"]),
        float(params["topo_threshold"]),
        float(params.get("affine_atol", 1e-4)),
        wm_start,
        wm_step,
        wm_min,
        wm_inset,
        pial_min,
        pial_max,
        pial_step,
        pial_floor,
    ]
    if not np.isfinite(numeric_values).all():
        raise ValueError("InitSurf numeric parameters must be finite")


def _validate_runtime_dependencies() -> None:
    try:
        CollisionManager()
    except Exception as exc:
        raise RuntimeError(
            "InitSurf collision checking requires python-fcl. "
            "Install the package with the InitSurf extra."
        ) from exc

    # Validate the packaged topology lookup table before launching workers.
    get_topo_correct()


def _detect_generate_mode(cfg):
    single_preproc_root = OmegaConf.select(cfg, "dataset.path", default=None)
    single_seg_root = OmegaConf.select(cfg, "dataset.seg_root", default=None)
    single_out_root = OmegaConf.select(cfg, "outputs.out_root", default=None)

    single_preproc_root = None if single_preproc_root in (None, "") else str(single_preproc_root)
    single_seg_root = None if single_seg_root in (None, "") else str(single_seg_root)
    single_out_root = None if single_out_root in (None, "") else str(single_out_root)

    roots_map = _get_map(cfg.dataset, ("roots",))
    seg_roots_map = _get_map(cfg.dataset, ("seg_roots",))
    out_roots_map = _get_map(cfg.outputs, ("out_roots",))

    if single_preproc_root is not None:
        log.info("InitSurf mode: SINGLE-DATASET")
        log.info(f"dataset.path = {single_preproc_root}")
        log.info(f"dataset.seg_root = {single_seg_root}")
        log.info(f"outputs.out_root = {single_out_root}")

        if single_seg_root is None:
            raise ValueError("Single-dataset InitSurf requires dataset.seg_root")
        if single_out_root is None:
            raise ValueError("Single-dataset InitSurf requires outputs.out_root")

        if roots_map is not None:
            log.warning(
                "Both dataset.path and dataset.roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.roots."
            )
        if seg_roots_map is not None:
            log.warning(
                "Both dataset.seg_root and dataset.seg_roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.seg_roots."
            )
        if out_roots_map is not None:
            log.warning(
                "Both outputs.out_root and outputs.out_roots are present. "
                "Using SINGLE-DATASET mode and ignoring outputs.out_roots."
            )

        return "single", single_preproc_root, single_seg_root, single_out_root, None, None, None

    if roots_map is not None:
        log.info("InitSurf mode: MULTI-DATASET")
        log.info(f"dataset.roots keys = {list(roots_map.keys())}")

        if seg_roots_map is None:
            raise ValueError("Multi-dataset InitSurf requires dataset.seg_roots")
        if out_roots_map is None:
            raise ValueError("Multi-dataset InitSurf requires outputs.out_roots")

        missing_seg = sorted(set(roots_map.keys()) - set(seg_roots_map.keys()))
        if missing_seg:
            raise KeyError(f"dataset.seg_roots missing keys required by dataset.roots: {missing_seg}")

        missing_out = sorted(set(roots_map.keys()) - set(out_roots_map.keys()))
        if missing_out:
            raise KeyError(f"outputs.out_roots missing keys required by dataset.roots: {missing_out}")

        extra_seg = sorted(set(seg_roots_map.keys()) - set(roots_map.keys()))
        if extra_seg:
            log.warning(f"dataset.seg_roots has extra keys not present in dataset.roots: {extra_seg}")

        extra_out = sorted(set(out_roots_map.keys()) - set(roots_map.keys()))
        if extra_out:
            log.warning(f"outputs.out_roots has extra keys not present in dataset.roots: {extra_out}")

        return "multi", None, None, None, roots_map, seg_roots_map, out_roots_map

    raise ValueError(
        "Could not determine InitSurf mode. Provide either:\n"
        " - dataset.path + dataset.seg_root + outputs.out_root (single-dataset)\n"
        "or\n"
        " - dataset.roots + dataset.seg_roots + outputs.out_roots (multi-dataset)"
    )

def separate_hemispheres(seg_mask: np.ndarray, gap_size: int = 1) -> np.ndarray:
    """Open a small midline gap between left/right WM seed masks.

    Only WM-seed labels are zeroed in the collision zone. This prevents the
    cleanup step from accidentally erasing cortical/subcortical labels near the
    midline if the cleaned seg9 volume is inspected or reused downstream.

    InitSurf currently builds WM masks from labels {1, 7} and {2, 8}; therefore
    the behavior for surface generation is the same as the older implementation,
    but the saved ``desc-seg9_dseg_cleaned`` file is anatomically safer.
    """
    if gap_size <= 0:
        return seg_mask.copy()

    lh_wm_mask = (seg_mask == 1) | (seg_mask == 7)
    rh_wm_mask = (seg_mask == 2) | (seg_mask == 8)

    struct = generate_binary_structure(3, 2)
    dilated_left = binary_dilation(lh_wm_mask, structure=struct, iterations=gap_size)
    dilated_right = binary_dilation(rh_wm_mask, structure=struct, iterations=gap_size)

    collision_zone = dilated_left & dilated_right
    wm_seed_labels = np.isin(seg_mask, [1, 2, 7, 8])

    new_seg = seg_mask.copy()
    new_seg[collision_zone & wm_seed_labels] = 0
    return new_seg


def build_wm_masks_from_labels(seg_npy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lh = (seg_npy == 1) | (seg_npy == 7)
    rh = (seg_npy == 2) | (seg_npy == 8)
    return lh.astype(np.uint8), rh.astype(np.uint8)


def compute_sdf(
    binary_seg: np.ndarray,
    sigma: float = 0.5,
    keep_largest: bool = True,
) -> np.ndarray:
    binary_seg = (binary_seg > 0).astype(np.uint8)

    cc, nc = compute_cc(binary_seg, connectivity=2, return_num=True)
    if nc == 0:
        raise ValueError("No connected components found")

    if keep_largest:
        volumes = np.bincount(cc.ravel())[1:]
        cc_id = 1 + int(np.argmax(volumes))
        seg = (cc == cc_id).astype(np.uint8)
    else:
        seg = (cc > 0).astype(np.uint8)

    sdf = (-edt(seg) + edt(1 - seg)).astype(np.float32)
    sdf = gaussian(sdf, sigma=sigma, preserve_range=True).astype(np.float32)
    return sdf


def sdf_to_probability(
    sdf: np.ndarray,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    prob = expit(-beta * sdf)
    return np.clip(prob, eps, 1.0 - eps).astype(np.float32)

def laplacian_smooth(
    verts: torch.Tensor,
    faces: torch.Tensor,
    lambd: float = 1.0,
) -> torch.Tensor:

    if not (0.0 <= float(lambd) <= 1.0):
        raise ValueError(f"lambd must be in [0, 1], got {lambd}")

    v = verts[0]
    f = faces[0]

    with torch.no_grad():
        V = int(v.shape[0])
        undirected_edges = torch.cat(
            [
                f[:, [0, 1]], f[:, [1, 0]],
                f[:, [1, 2]], f[:, [2, 1]],
                f[:, [2, 0]], f[:, [0, 2]],
            ],
            dim=0,
        ).long()

        idx = undirected_edges.t().contiguous()
        values = torch.ones(idx.shape[1], dtype=v.dtype, device=v.device)
        L = torch.sparse_coo_tensor(idx, values, (V, V), device=v.device).coalesce()

        deg = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1).clamp_min(1.0)
        v_bar = torch.sparse.mm(L, v) / deg

    return ((1.0 - float(lambd)) * v + float(lambd) * v_bar).unsqueeze(0)


def meshes_collide(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> bool:
    cm = CollisionManager()
    cm.add_object("a", mesh_a)
    cm.add_object("b", mesh_b)
    return bool(cm.in_collision_internal())


def pial_vs_wm_collide(pial_mesh: trimesh.Trimesh, wm_mesh: trimesh.Trimesh) -> bool:
    cm = CollisionManager()
    cm.add_object("wm", wm_mesh)
    cm.add_object("pial", pial_mesh)
    return bool(cm.in_collision_internal())


def get_topo_correct():
    global _TOPO_CORRECT
    if _TOPO_CORRECT is None:
        _TOPO_CORRECT = topology()
    return _TOPO_CORRECT


def prepare_topo_sdf(sdf: np.ndarray, topo_threshold: float) -> np.ndarray:
    sdf = np.asarray(sdf, dtype=np.float32)
    sdf_topo = get_topo_correct().apply(sdf, threshold=np.float32(topo_threshold))
    return np.asarray(sdf_topo, dtype=np.float32)


def mesh_from_topo_sdf(
    sdf_topo: np.ndarray,
    level: float,
    brain_affine: np.ndarray,
    n_smooth: int,
) -> trimesh.Trimesh:
    v_mc, f_mc, _, _ = marching_cubes(
        -sdf_topo,
        level=-float(level),
        method="lorensen",
    )

    v_mc = torch.tensor(v_mc.copy(), dtype=torch.float32).unsqueeze(0)
    f_mc = torch.tensor(f_mc.copy(), dtype=torch.long).unsqueeze(0)

    for _ in range(n_smooth):
        v_mc = laplacian_smooth(v_mc, f_mc, lambd=1.0)

    v_np = v_mc[0].cpu().numpy()
    f_np = f_mc[0].cpu().numpy()

    verts_world = apply_affine(brain_affine, v_np)

    mesh = trimesh.Trimesh(vertices=verts_world, faces=f_np, process=False)
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception as e:
        log.warning(f"Mesh normal/winding repair failed at level={level}: {e}")
    return mesh


def level_key(level: float) -> float:
    return round(float(level), 6)


def make_mesh_getter(
    sdf_topo: np.ndarray,
    brain_affine: np.ndarray,
    n_smooth: int,
) -> Callable[[float], trimesh.Trimesh]:
    cache: Dict[float, trimesh.Trimesh] = {}

    def get_mesh(level: float) -> trimesh.Trimesh:
        key = level_key(level)
        if key not in cache:
            cache[key] = mesh_from_topo_sdf(
                sdf_topo=sdf_topo,
                level=key,
                brain_affine=brain_affine,
                n_smooth=n_smooth,
            )
        return cache[key]

    return get_mesh


def make_levels(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError(f"step must be positive. Got {step}")
    if start > stop:
        return []
    n_steps = int(np.round((stop - start) / step))
    levels = [round(float(start + i * step), 6) for i in range(n_steps + 1)]
    return [x for x in levels if start - 1e-6 <= x <= stop + 1e-6]

def free_collision_wm_from_topo(
    lh_wm_topo: np.ndarray,
    rh_wm_topo: np.ndarray,
    brain_affine: np.ndarray,
    start_level: float,
    step: float,
    min_level: float,
    n_smooth: int,
    wm_inset: float = 1.0,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, float]:

    if step >= 0:
        raise ValueError(f"WM step must be negative for inward search. Got step={step}")

    if start_level < min_level:
        raise ValueError(
            f"start_level must be >= min_level for inward search. "
            f"Got start_level={start_level}, min_level={min_level}"
        )

    if wm_inset < 0:
        raise ValueError(f"wm_inset must be non-negative. Got wm_inset={wm_inset}")

    get_l = make_mesh_getter(lh_wm_topo, brain_affine, n_smooth)
    get_r = make_mesh_getter(rh_wm_topo, brain_affine, n_smooth)

    n_steps = int(np.floor((start_level - min_level) / abs(step)))
    levels = [
        round(float(start_level + i * step), 6)
        for i in range(n_steps + 1)
    ]
    levels = [lvl for lvl in levels if lvl >= min_level - 1e-6]

    if not levels:
        raise RuntimeError(
            f"[WM] Empty level search grid. "
            f"start_level={start_level}, min_level={min_level}, step={step}"
        )

    first_collision_free_level = None

    for level in levels:
        mesh_l = get_l(level)
        mesh_r = get_r(level)

        if not meshes_collide(mesh_l, mesh_r):
            first_collision_free_level = level
            break

    if first_collision_free_level is None:
        raise RuntimeError(
            f"[WM] No collision-free level found in "
            f"{start_level:.3f}..{min_level:.3f} with step={step:.3f}."
        )

    wm_final_level = float(first_collision_free_level - wm_inset)

    if wm_final_level < min_level:
        log.warning(
            f"[WM] Requested inset level {wm_final_level:.3f} is below "
            f"min_level={min_level:.3f}; clamping to min_level."
        )
        wm_final_level = float(min_level)

    mesh_l_final = get_l(wm_final_level)
    mesh_r_final = get_r(wm_final_level)

    if meshes_collide(mesh_l_final, mesh_r_final):
        log.warning(
            f"[WM] inset level={wm_final_level:.3f} still collides; "
            f"falling back to first_collision_free_level={first_collision_free_level:.3f}."
        )
        wm_final_level = float(first_collision_free_level)
        mesh_l_final = get_l(wm_final_level)
        mesh_r_final = get_r(wm_final_level)

    log.info(
        f"[WM] first_collision_free_level={first_collision_free_level:.3f}, "
        f"inset={wm_inset:.3f} -> wm_final_level={wm_final_level:.3f}"
    )

    return mesh_l_final, mesh_r_final, wm_final_level

def free_collision_pial_offset_grid(
    lh_pial_topo: np.ndarray,
    rh_pial_topo: np.ndarray,
    wm_l_mesh: trimesh.Trimesh,
    wm_r_mesh: trimesh.Trimesh,
    brain_affine: np.ndarray,
    min_level: float,
    max_level: float,
    step: float,
    n_smooth: int,
    absolute_floor: float = 0.1,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, float, float]:

    if step <= 0:
        raise ValueError(f"pial offset grid step must be positive. Got step={step}")
    if min_level > max_level:
        raise ValueError(f"min_level must be <= max_level. Got {min_level} > {max_level}")
    if absolute_floor < 0:
        raise ValueError(f"absolute_floor must be non-negative. Got {absolute_floor}")

    get_l = make_mesh_getter(lh_pial_topo, brain_affine, n_smooth)
    get_r = make_mesh_getter(rh_pial_topo, brain_affine, n_smooth)

    primary_levels = make_levels(min_level, max_level, step)
    if not primary_levels:
        raise RuntimeError(
            f"[Pial-offset] Empty primary level grid. "
            f"min_level={min_level}, max_level={max_level}, step={step}"
        )

    log.info(
        f"[Pial-offset] strict symmetric inside-out search: "
        f"min={min_level:.3f}, max={max_level:.3f}, step={step:.3f}, "
        f"n_primary_levels={len(primary_levels)}, absolute_floor={absolute_floor:.3f}"
    )

    wm_safe_l: Dict[float, bool] = {}
    wm_safe_r: Dict[float, bool] = {}

    def is_l_wm_safe(lvl: float) -> bool:
        lvl = level_key(lvl)
        if lvl not in wm_safe_l:
            wm_safe_l[lvl] = not pial_vs_wm_collide(get_l(lvl), wm_l_mesh)
        return wm_safe_l[lvl]

    def is_r_wm_safe(lvl: float) -> bool:
        lvl = level_key(lvl)
        if lvl not in wm_safe_r:
            wm_safe_r[lvl] = not pial_vs_wm_collide(get_r(lvl), wm_r_mesh)
        return wm_safe_r[lvl]

    def check_symmetric_level(level: float):
        level = level_key(level)

        l_safe = is_l_wm_safe(level)
        r_safe = is_r_wm_safe(level)

        if not l_safe or not r_safe:
            log.info(
                f"[Pial-offset] frontier={level:.3f} WM collision: "
                f"L_safe={l_safe}, R_safe={r_safe}"
            )
            return "wm_collision", None, None

        mesh_l = get_l(level)
        mesh_r = get_r(level)

        if meshes_collide(mesh_l, mesh_r):
            log.info(f"[Pial-offset] frontier={level:.3f} LR collision")
            return "lr_collision", mesh_l, mesh_r

        log.debug(f"[Pial-offset] frontier={level:.3f} valid symmetric pair")
        return "valid", mesh_l, mesh_r

    best_pair = None 

    checked_primary = 0
    wm_rejected_primary = 0
    lr_rejected_primary = 0

    for frontier in primary_levels:
        frontier = level_key(frontier)
        status, mesh_l, mesh_r = check_symmetric_level(frontier)
        checked_primary += 1

        if status == "valid":
            best_pair = (frontier, frontier, mesh_l, mesh_r)
            continue

        if status == "wm_collision":
            wm_rejected_primary += 1
            if best_pair is not None:
                log.info(
                    f"[Pial-offset] stopping primary search at frontier={frontier:.3f} "
                    f"because WM collision appeared after valid levels."
                )
                break
            continue

        if status == "lr_collision":
            lr_rejected_primary += 1
            log.info(
                f"[Pial-offset] stopping primary search at frontier={frontier:.3f} "
                f"because LR collision appeared."
            )
            break

    if best_pair is not None:
        lvl_l, lvl_r, mesh_l, mesh_r = best_pair
        log.info(
            f"[Pial-offset] best symmetric pair from primary search: "
            f"L={lvl_l:.3f}, R={lvl_r:.3f}, "
            f"checked_primary={checked_primary}, "
            f"wm_rejected_primary={wm_rejected_primary}, "
            f"lr_rejected_primary={lr_rejected_primary}, "
            f"wm_cache L={len(wm_safe_l)}, R={len(wm_safe_r)}"
        )
        return mesh_l, mesh_r, lvl_l, lvl_r

    fallback_start = level_key(min_level - step)
    fallback_levels = []
    if fallback_start >= absolute_floor - 1e-6:
        n_fb = int(np.round((fallback_start - absolute_floor) / step))
        fallback_levels = [
            round(float(fallback_start - i * step), 6)
            for i in range(n_fb + 1)
        ]
        fallback_levels = [
            v for v in fallback_levels
            if absolute_floor - 1e-6 <= v <= fallback_start + 1e-6
        ]

    log.warning(
        f"[Pial-offset] no valid pair found in primary range "
        f"{min_level:.3f}..{max_level:.3f}. "
        f"Starting fallback downward search to absolute_floor={absolute_floor:.3f} "
        f"with {len(fallback_levels)} levels."
    )

    checked_fallback = 0
    wm_rejected_fallback = 0
    lr_rejected_fallback = 0

    for frontier in fallback_levels:
        frontier = level_key(frontier)
        status, mesh_l, mesh_r = check_symmetric_level(frontier)
        checked_fallback += 1

        if status == "valid":
            log.info(
                f"[Pial-offset] fallback found symmetric pair: "
                f"L={frontier:.3f}, R={frontier:.3f}, "
                f"checked_fallback={checked_fallback}, "
                f"wm_rejected_fallback={wm_rejected_fallback}, "
                f"lr_rejected_fallback={lr_rejected_fallback}, "
                f"wm_cache L={len(wm_safe_l)}, R={len(wm_safe_r)}"
            )
            return mesh_l, mesh_r, frontier, frontier

        if status == "wm_collision":
            wm_rejected_fallback += 1
        elif status == "lr_collision":
            lr_rejected_fallback += 1

    raise RuntimeError(
        f"[Pial-offset] No collision-free symmetric pial pair found. "
        f"Primary range={min_level:.3f}..{max_level:.3f}, "
        f"fallback_floor={absolute_floor:.3f}, step={step:.3f}. "
        f"Checked primary={checked_primary}, fallback={checked_fallback}. "
        f"Consider lowering pial_absolute_floor or checking WM/pial field quality."
    )

def _generate_subject(
    subject_id: str,
    ds_key: str,
    preproc_root: str,
    seg_root: str,
    out_root: str,
    ses: str,
    space: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.time()

    gap_size = int(params["gap_size"])
    sdf_sigma = float(params["sdf_sigma"])
    topo_thr = float(params["topo_threshold"])
    n_smooth = int(params["n_smooth"])

    wm_start_level = float(params["wm_start_level"])
    wm_step = float(params["wm_step"])
    wm_min_level = float(params["wm_min_level"])
    wm_inset = float(params.get("wm_inset", 1.0))

    pial_min_level = float(params["pial_min_level"])
    pial_max_level = float(params["pial_max_level"])
    pial_grid_step = float(params["pial_grid_step"])
    pial_absolute_floor = float(params.get("pial_absolute_floor", 0.1))

    overwrite = bool(params.get("overwrite", False))
    validate_affine = bool(params.get("validate_affine", True))
    affine_atol = float(params.get("affine_atol", 1e-4))

    if (not overwrite) and outputs_complete(out_root, subject_id, ses, space):
        msg = "All expected InitSurf outputs already exist; skipping because overwrite=false."
        log.info(f"[{ds_key}][{subject_id}] {msg}")
        return {
            "status": "skipped_existing",
            "reason": msg,
            "subject_id": subject_id,
            "ds_key": ds_key,
        }

    brain_path = t1_mni_path(preproc_root, subject_id, ses=ses, space=space)
    pred_path = seg9_dseg_path(seg_root, subject_id, ses=ses, space=space)

    if not os.path.exists(brain_path):
        raise FileNotFoundError(f"Missing preprocessed T1: {brain_path}")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing seg9 prediction: {pred_path}")

    brain = nib.load(brain_path)
    seg_img = nib.load(pred_path)

    if brain.shape[:3] != seg_img.shape[:3]:
        raise ValueError(
            f"[{ds_key}][{subject_id}] T1/seg shape mismatch: "
            f"T1={brain.shape[:3]}, seg={seg_img.shape[:3]}"
        )

    if validate_affine and not _is_affine_close(brain.affine, seg_img.affine, atol=affine_atol):
        raise ValueError(
            f"[{ds_key}][{subject_id}] T1/seg affine mismatch beyond atol={affine_atol}"
        )

    affine = np.asarray(brain.affine, dtype=np.float64)
    seg_pred = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.uint8)
    _validate_seg9_values(seg_pred, subject_id)

    anat_dir = out_anat_dir(out_root, subject_id, ses=ses)
    surf_dir = out_surf_dir(out_root, subject_id, ses=ses)
    os.makedirs(anat_dir, exist_ok=True)
    os.makedirs(surf_dir, exist_ok=True)

    save_nifti(
        seg_pred,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-seg9_dseg_used.nii.gz"),
        dtype=np.uint8,
    )

    seg_clean = separate_hemispheres(seg_pred, gap_size=gap_size)
    save_nifti(
        seg_clean,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-seg9_dseg_cleaned.nii.gz"),
        dtype=np.uint8,
    )

    lh_mask, rh_mask = build_wm_masks_from_labels(seg_clean)

    lh_wm_sdf_raw = compute_sdf(lh_mask, sigma=sdf_sigma, keep_largest=True)
    rh_wm_sdf_raw = compute_sdf(rh_mask, sigma=sdf_sigma, keep_largest=True)

    lh_wm_topo = prepare_topo_sdf(lh_wm_sdf_raw, topo_threshold=topo_thr)
    rh_wm_topo = prepare_topo_sdf(rh_wm_sdf_raw, topo_threshold=topo_thr)

    mesh_l_wm, mesh_r_wm, wm_final_level = free_collision_wm_from_topo(
        lh_wm_topo=lh_wm_topo,
        rh_wm_topo=rh_wm_topo,
        brain_affine=affine,
        start_level=wm_start_level,
        step=wm_step,
        min_level=wm_min_level,
        n_smooth=n_smooth,
        wm_inset=wm_inset,
    )

    shift = -float(wm_final_level)
    lh_wm_sdf = (lh_wm_topo + np.float32(shift)).astype(np.float32)
    rh_wm_sdf = (rh_wm_topo + np.float32(shift)).astype(np.float32)

    lh_pial_topo = prepare_topo_sdf(lh_wm_sdf, topo_threshold=topo_thr)
    rh_pial_topo = prepare_topo_sdf(rh_wm_sdf, topo_threshold=topo_thr)

    mesh_l_pial, mesh_r_pial, pial_offset_l, pial_offset_r = free_collision_pial_offset_grid(
        lh_pial_topo=lh_pial_topo,
        rh_pial_topo=rh_pial_topo,
        wm_l_mesh=mesh_l_wm,
        wm_r_mesh=mesh_r_wm,
        brain_affine=affine,
        min_level=pial_min_level,
        max_level=pial_max_level,
        step=pial_grid_step,
        n_smooth=n_smooth,
        absolute_floor=pial_absolute_floor,
    )

    lh_pial_sdf = (lh_pial_topo - np.float32(pial_offset_l)).astype(np.float32)
    rh_pial_sdf = (rh_pial_topo - np.float32(pial_offset_r)).astype(np.float32)

    wm_lr_col = meshes_collide(mesh_l_wm, mesh_r_wm)
    pial_l_wm_col = pial_vs_wm_collide(mesh_l_pial, mesh_l_wm)
    pial_r_wm_col = pial_vs_wm_collide(mesh_r_pial, mesh_r_wm)
    pial_lr_col = meshes_collide(mesh_l_pial, mesh_r_pial)

    all_collision_free = not (wm_lr_col or pial_l_wm_col or pial_r_wm_col or pial_lr_col)
    if not all_collision_free:
        raise RuntimeError(
            f"Final InitSurf collision check failed: "
            f"WM_L-WM_R={wm_lr_col}, "
            f"Pial_L-WM_L={pial_l_wm_col}, "
            f"Pial_R-WM_R={pial_r_wm_col}, "
            f"Pial_L-Pial_R={pial_lr_col}"
        )

    mesh_l_wm.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-L_white.surf.ply"))
    mesh_r_wm.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-R_white.surf.ply"))
    mesh_l_pial.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-L_pial.surf.ply"))
    mesh_r_pial.export(os.path.join(surf_dir, f"{subject_id}_ses-{ses}_space-{space}_hemi-R_pial.surf.ply"))

    save_nifti(
        lh_wm_sdf,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-lh_white_sdf.nii.gz"),
    )
    save_nifti(
        rh_wm_sdf,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-rh_white_sdf.nii.gz"),
    )
    save_nifti(
        lh_pial_sdf,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-lh_pial_sdf.nii.gz"),
    )
    save_nifti(
        rh_pial_sdf,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-rh_pial_sdf.nii.gz"),
    )

    lh_ribbon = (lh_pial_sdf <= 0) & (~(lh_wm_sdf <= 0))
    rh_ribbon = (rh_pial_sdf <= 0) & (~(rh_wm_sdf <= 0))
    ribbon_mask = (lh_ribbon | rh_ribbon).astype(np.uint8)

    if ribbon_mask.sum() == 0:
        raise RuntimeError(f"[{ds_key}][{subject_id}] Empty cortical ribbon mask")

    ribbon_sdf = compute_sdf(ribbon_mask, sigma=sdf_sigma, keep_largest=False)
    ribbon_prob = sdf_to_probability(ribbon_sdf, beta=1.0)

    save_nifti(
        ribbon_sdf,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-ribbon_sdf.nii.gz"),
    )
    save_nifti(
        ribbon_prob,
        affine,
        os.path.join(anat_dir, f"{subject_id}_ses-{ses}_space-{space}_desc-ribbon_prob.nii.gz"),
    )

    elapsed = time.time() - t0

    qc_path = os.path.join(
        anat_dir,
        f"{subject_id}_ses-{ses}_space-{space}_desc-initsurf_qc.json",
    )
    _write_json_atomic(
        qc_path,
        {
            "Status": "ok",
            "Dataset": ds_key,
            "Subject": subject_id,
            "Session": f"ses-{ses}",
            "Space": space,
            "SimCortexVersion": _simcortex_version(),
            "Inputs": {
                "PreprocessedT1": brain_path,
                "Segmentation": pred_path,
            },
            "Parameters": params,
            "Levels": {
                "White": float(wm_final_level),
                "PialLeft": float(pial_offset_l),
                "PialRight": float(pial_offset_r),
            },
            "CollisionChecks": {
                "WhiteLeftRight": bool(wm_lr_col),
                "PialLeftWhiteLeft": bool(pial_l_wm_col),
                "PialRightWhiteRight": bool(pial_r_wm_col),
                "PialLeftRight": bool(pial_lr_col),
            },
            "ElapsedSeconds": float(elapsed),
        },
    )

    log.info(
        f"[{ds_key}][{subject_id}] OK | "
        f"wm_final_level={wm_final_level:.3f} "
        f"pialL_offset={pial_offset_l:.3f} "
        f"pialR_offset={pial_offset_r:.3f} "
        f"elapsed={elapsed:.1f}s"
    )

    return {
        "status": "ok",
        "subject_id": subject_id,
        "ds_key": ds_key,
        "elapsed": elapsed,
        "wm_final_level": float(wm_final_level),
        "pial_l": float(pial_offset_l),
        "pial_r": float(pial_offset_r),
        "wm_lr_col": bool(wm_lr_col),
        "pial_l_wm_col": bool(pial_l_wm_col),
        "pial_r_wm_col": bool(pial_r_wm_col),
        "pial_lr_col": bool(pial_lr_col),
    }


def _generate_subject_from_job(job: Dict[str, Any]) -> Dict[str, Any]:
    try:

        if mp.current_process().name != "MainProcess":
            setup_subject_logger(job["log_dir"], job["ds_key"], job["subject_id"])

        result = _generate_subject(
            subject_id=job["subject_id"],
            ds_key=job["ds_key"],
            preproc_root=job["preproc_root"],
            seg_root=job["seg_root"],
            out_root=job["out_root"],
            ses=job["ses"],
            space=job["space"],
            params=job["params"],
        )
        return result
    except Exception as e:
        return {
            "status": "failed",
            "subject_id": job.get("subject_id", "UNKNOWN"),
            "ds_key": job.get("ds_key", "UNKNOWN"),
            "reason": repr(e),
            "traceback": traceback.format_exc(),
        }


def _log_result_summary(result: Dict[str, Any]) -> None:
    ds = result.get("ds_key", "UNKNOWN")
    sub = result.get("subject_id", "UNKNOWN")
    status = result.get("status", "unknown")

    if status == "ok":
        log.info(
            f"[{ds}][{sub}] OK | "
            f"elapsed={result.get('elapsed', -1):.1f}s "
            f"wm={result.get('wm_final_level', float('nan')):.3f} "
            f"pialL={result.get('pial_l', float('nan')):.3f} "
            f"pialR={result.get('pial_r', float('nan')):.3f}"
        )
    elif status == "skipped_existing":
        log.info(f"[{ds}][{sub}] SKIPPED_EXISTING | {result.get('reason', '')}")
    else:
        log.error(
            f"[{ds}][{sub}] FAILED | {result.get('reason', '')}\n"
            f"{result.get('traceback', '')}"
        )


def _status_counts(results: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped_existing = sum(
        1 for r in results if r.get("status") == "skipped_existing"
    )
    failed = sum(
        1
        for r in results
        if r.get("status") not in {"ok", "skipped_existing"}
    )
    return ok, skipped_existing, failed


def _run_jobs(jobs: List[Dict[str, Any]], n_workers: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if n_workers <= 1:
        log.info("InitSurf execution: serial")
        with tqdm(total=len(jobs), desc="InitSurf", unit="subj") as pbar:
            for job in jobs:
                result = _generate_subject_from_job(job)
                _log_result_summary(result)
                results.append(result)

                ok, skipped_existing, failed = _status_counts(results)
                pbar.set_postfix_str(
                    f"ok={ok} skipped_existing={skipped_existing} failed={failed}"
                )
                pbar.update(1)
        return results

    log.info(f"InitSurf execution: multiprocessing with n_workers={n_workers}")
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_generate_subject_from_job, job): job
            for job in jobs
        }

        with tqdm(total=len(futures), desc="InitSurf", unit="subj") as pbar:
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "subject_id": job.get("subject_id", "UNKNOWN"),
                        "ds_key": job.get("ds_key", "UNKNOWN"),
                        "reason": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                _log_result_summary(result)
                results.append(result)

                ok, skipped_existing, failed = _status_counts(results)
                pbar.set_postfix_str(
                    f"ok={ok} skipped_existing={skipped_existing} failed={failed}"
                )
                pbar.update(1)

    return results


def _write_run_summary(results: List[Dict[str, Any]], out_path: str) -> None:
    if not results:
        return
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)


# ------------------------------- main ------------------------------- #

@hydra.main(
    config_path="pkg://simcortex.configs.initsurf",
    config_name="generate",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    setup_logger(str(cfg.outputs.log_dir), "generate_initsurf.log")
    log.info("=== InitSurf generate config ===")
    log.info("\n" + OmegaConf.to_yaml(cfg))

    mode, single_preproc_root, single_seg_root, single_out_root, roots_map, seg_roots_map, out_roots_map = _detect_generate_mode(cfg)

    split_file = to_absolute_path(str(cfg.dataset.split_file))
    if single_preproc_root is not None:
        single_preproc_root = to_absolute_path(str(single_preproc_root))
    if single_seg_root is not None:
        single_seg_root = to_absolute_path(str(single_seg_root))
    if single_out_root is not None:
        single_out_root = to_absolute_path(str(single_out_root))
    if roots_map is not None:
        roots_map = {k: to_absolute_path(str(v)) for k, v in roots_map.items()}
    if seg_roots_map is not None:
        seg_roots_map = {k: to_absolute_path(str(v)) for k, v in seg_roots_map.items()}
    if out_roots_map is not None:
        out_roots_map = {k: to_absolute_path(str(v)) for k, v in out_roots_map.items()}

    df = pd.read_csv(split_file)
    df = normalize_subject_column(df)
    _validate_base_split_df(df)

    split_name = str(cfg.dataset.split_name).strip()
    split_name_l = split_name.lower()
    if split_name_l not in {"all", "*", "any"}:
        wanted = {p.strip().lower() for p in split_name_l.replace("+", ",").split(",") if p.strip()}
        df = df[df["split"].astype(str).str.strip().str.lower().isin(wanted)].copy()

    if df.empty:
        raise ValueError(f"No rows found in split_file for split='{split_name}'")

    ses = _normalize_session_label(str(cfg.dataset.session_label))
    space = str(cfg.dataset.space).strip()
    params = OmegaConf.to_container(cfg.params, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("cfg.params must resolve to a mapping")
    _validate_params(params)

    n_workers = int(OmegaConf.select(cfg, "n_workers", default=1))
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    _validate_runtime_dependencies()

    jobs: List[Dict[str, Any]] = []

    if mode == "single":
        _validate_single_split_df(df)

        write_dataset_description(
            str(single_out_root),
            name="sc-initsurf",
        )

        _validate_unique_rows(df, ["subject"])
        entries = df[["subject"]].drop_duplicates().reset_index(drop=True)
        log.info(f"InitSurf: {len(entries)} subjects (mode=single, split={split_name})")

        for _, row in entries.iterrows():
            subject_id = str(row["subject"])
            jobs.append({
                "subject_id": subject_id,
                "ds_key": "SINGLE",
                "preproc_root": str(single_preproc_root),
                "seg_root": str(single_seg_root),
                "out_root": str(single_out_root),
                "ses": ses,
                "space": space,
                "params": params,
                "log_dir": str(cfg.outputs.log_dir),
            })

    else:
        _validate_multi_split_df(df)
        df["dataset"] = df["dataset"].astype(str).str.strip()

        dataset_keys = sorted(df["dataset"].unique().tolist())

        missing_preproc = sorted(k for k in dataset_keys if k not in roots_map)
        missing_seg = sorted(k for k in dataset_keys if k not in seg_roots_map)
        missing_out = sorted(k for k in dataset_keys if k not in out_roots_map)

        if missing_preproc:
            raise KeyError(f"dataset.roots missing keys from split_file: {missing_preproc}")
        if missing_seg:
            raise KeyError(f"dataset.seg_roots missing keys from split_file: {missing_seg}")
        if missing_out:
            raise KeyError(f"outputs.out_roots missing keys from split_file: {missing_out}")

        for ds_key in dataset_keys:
            write_dataset_description(
                str(out_roots_map[ds_key]),
                name="sc-initsurf",
            )

        _validate_unique_rows(df, ["dataset", "subject"])
        entries = df[["dataset", "subject"]].drop_duplicates().reset_index(drop=True)
        log.info(f"InitSurf: {len(entries)} subject-dataset pairs (mode=multi, split={split_name})")

        for _, row in entries.iterrows():
            ds_key = str(row["dataset"])
            subject_id = str(row["subject"])

            jobs.append({
                "subject_id": subject_id,
                "ds_key": ds_key,
                "preproc_root": str(roots_map[ds_key]),
                "seg_root": str(seg_roots_map[ds_key]),
                "out_root": str(out_roots_map[ds_key]),
                "ses": ses,
                "space": space,
                "params": params,
                "log_dir": str(cfg.outputs.log_dir),
            })

    t0 = time.time()
    results = _run_jobs(jobs, n_workers=n_workers)
    elapsed = time.time() - t0

    ok, skipped_existing, failed = _status_counts(results)

    summary_path = os.path.join(str(cfg.outputs.log_dir), "generate_initsurf_summary.csv")
    _write_run_summary(results, summary_path)

    log.info(
        f"InitSurf generation finished in {elapsed / 60:.2f} min. "
        f"OK={ok}, SKIPPED_EXISTING={skipped_existing}, FAILED={failed}. "
        f"Summary: {summary_path}"
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
