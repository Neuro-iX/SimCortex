#!/usr/bin/env python
"""
Evaluation script for unified cortical surface reconstruction predictions.
Compares predicted PLY surfaces in a fixed folder against ground truth PLYs in a fixed base directory.
Computes per-surface and overall metrics:
  - Chamfer Distance
  - Average Symmetric Surface Distance (ASSD)
  - Hausdorff Distance (HD)
  - Self-Intersection Fraction (SIF)
  - Pairwise surface collisions
Configuration via Hydra (eval.yaml).
"""
import os
import glob
import logging

import torch
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.loss import chamfer_distance
from trimesh.collision import CollisionManager

# Try importing pymeshlab for SIF calculation
try:
    import pymeshlab as pyml
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False
    logging.warning("pymeshlab not found. SIF calculation will be disabled.")

# Define collision pairs globally
collision_pairs = [
    ('lh_pial','rh_pial'),
    ('lh_white','lh_pial'),
    ('rh_white','rh_pial'),
    ('lh_white','rh_white'),
]

# --- Utility functions --------------------------------

def load_mesh_from_ply(path):
    m = trimesh.load(path, process=False)
    return torch.tensor(m.vertices, dtype=torch.float32), torch.tensor(m.faces, dtype=torch.int64)


def to_p3d_mesh(verts, faces, device=None):
    if device:
        verts = verts.to(device)
        faces = faces.to(device)
    return Meshes(verts=[verts], faces=[faces])


def point_to_mesh_dist(pcls, mesh):
    pts = pcls.points_packed()
    first_idx = pcls.cloud_to_packed_first_idx()
    max_pts = pcls.num_points_per_cloud().max().item()
    tris = mesh.verts_packed()[mesh.faces_packed()]
    tri_first = mesh.mesh_to_faces_packed_first_idx()
    # Use .apply for autograd function
    d2 = _PointFaceDistance.apply(pts, first_idx, tris, tri_first, max_pts)
    return torch.sqrt(d2)


def compute_chamfer(mesh_p, mesh_g, n_pts):
    ppts = sample_points_from_meshes(mesh_p, num_samples=n_pts)
    gpts = sample_points_from_meshes(mesh_g, num_samples=n_pts)
    loss, _ = chamfer_distance(ppts, gpts)
    return loss.item()


def compute_assd_hd(mesh_p, mesh_g, n_pts):
    ppts = sample_points_from_meshes(mesh_p, num_samples=n_pts)
    gpts = sample_points_from_meshes(mesh_g, num_samples=n_pts)
    pcl_p = Pointclouds(ppts)
    pcl_g = Pointclouds(gpts)
    d_p = point_to_mesh_dist(pcl_p, mesh_g)
    d_g = point_to_mesh_dist(pcl_g, mesh_p)
    assd = (d_p.mean().item() + d_g.mean().item()) / 2.0
    hd = max(d_p.max().item(), d_g.max().item())
    return assd, hd


def compute_sif(verts, faces):
    if not HAS_PYMESHLAB:
        return float('nan')
    v_np = verts.cpu().numpy().astype(np.float64)
    f_np = faces.cpu().numpy().astype(np.int32)
    ms = pyml.MeshSet()
    ms.add_mesh(pyml.Mesh(vertex_matrix=v_np, face_matrix=f_np), 'm')
    total = ms.current_mesh().face_number()
    if total == 0:
        return float('nan')
    ms.apply_filter('compute_selection_by_self_intersections_per_face')
    ms.apply_filter('meshing_remove_selected_faces')
    remaining = ms.current_mesh().face_number()
    return (total - remaining) / total * 100.0


def check_collision(file1, file2):
    m1 = trimesh.load(file1, process=False)
    m2 = trimesh.load(file2, process=False)
    cm = CollisionManager()
    cm.add_object('m1', m1)
    cm.add_object('m2', m2)
    _, contacts = cm.in_collision_internal(return_data=True)
    idxs1 = [c.index('m1') for c in contacts]
    idxs2 = [c.index('m2') for c in contacts]
    return {
        'total_faces': (len(m1.faces), len(m2.faces)),
        'intersecting_faces': (len(set(idxs1)), len(set(idxs2))),
        'num_intersections': len(contacts)
    }


def evaluate_subject(preds, gts, surface_names, device, n_chamfer, n_assd_hd):
    subj_res = {}
    # geometric & topological metrics per surface
    for surf in surface_names:
        pv, pf = load_mesh_from_ply(preds[surf])
        gv, gf = load_mesh_from_ply(gts[surf])
        mesh_p = to_p3d_mesh(pv, pf, device)
        mesh_g = to_p3d_mesh(gv, gf, device)
        cham = compute_chamfer(mesh_p, mesh_g, n_chamfer)
        assd, hd = compute_assd_hd(mesh_p, mesh_g, n_assd_hd)
        sif = compute_sif(pv, pf)
        subj_res[surf] = {"Chamfer": cham, "ASSD": assd, "HD": hd, "SIF": sif}
    # pairwise collisions
    for a, b in collision_pairs:
        key = f'collision_{a}_{b}'
        subj_res[key] = check_collision(preds[a], preds[b])
    return subj_res


@hydra.main(config_path="../../../configs/surf_recon", config_name="eval")
def eval_app(cfg: DictConfig) -> None:
    # setup
    logging.basicConfig(level=cfg.eval.log_level)
    logging.info("Starting evaluation with config:\n%s", OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.eval.device if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)

    # config params
    PRED_DIR = cfg.dataset.pred_dir
    GT_BASE_DIR = cfg.dataset.gt_base_dir
    surface_names = cfg.dataset.surface_names
    n_chamfer = cfg.eval.n_chamfer
    n_assd_hd = cfg.eval.n_assd_hd

    # gather files
    pred_files = glob.glob(os.path.join(PRED_DIR, "*.ply"))
    subs_pred = {}
    for f in pred_files:
        name = os.path.basename(f).replace('pred_','')
        sub, hemi, surf_ext = name.split('_')
        key = f"{hemi}_{surf_ext.split('.')[0]}"
        subs_pred.setdefault(sub, {})[key] = f

    subs_gt = {}
    for sub, pdict in subs_pred.items():
        gdict = {}
        subdir = os.path.join(GT_BASE_DIR, sub)
        for key, pf in pdict.items():
            hemi, surf = key.split('_')
            fn = f"{sub}_{hemi}_{surf}_mni.ply"
            p = os.path.join(subdir, fn)
            if os.path.exists(p):
                gdict[key] = p
        if len(gdict) == len(surface_names):
            subs_gt[sub] = gdict

    subjects = sorted(set(subs_pred) & set(subs_gt))
    logging.info("Found %d subjects for evaluation", len(subjects))

    # evaluate
    all_results = {}
    coll_stats = {f'collision_{a}_{b}': [] for a, b in collision_pairs}
    for sub in tqdm(subjects, desc="Evaluating subjects"):
        res = evaluate_subject(subs_pred[sub], subs_gt[sub], surface_names, device, n_chamfer, n_assd_hd)
        all_results[sub] = res
        for k in coll_stats:
            coll_stats[k].append(res[k]['num_intersections'])

    # output
    out_dir = cfg.eval.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # surface metrics table
    surf_rows = []
    for sub, res in all_results.items():
        row = {'subject': sub}
        for surf in surface_names:
            for m in ('Chamfer','ASSD','HD','SIF'):
                row[f'{surf}_{m}'] = res[surf][m]
        surf_rows.append(row)
    df_surf = pd.DataFrame(surf_rows)
    df_surf.to_excel(os.path.join(out_dir, 'surface_metrics.xlsx'), index=False)
    logging.info("Surface metrics written to %s", out_dir)

    # collision metrics table
    coll_rows = []
    for sub, res in all_results.items():
        row = {'subject': sub}
        for k in coll_stats:
            info = res[k]
            row[f'{k}_num_intersections'] = info['num_intersections']
            row[f'{k}_intersecting_faces'] = str(info['intersecting_faces'])
            row[f'{k}_total_faces'] = str(info['total_faces'])
        coll_rows.append(row)
    df_coll = pd.DataFrame(coll_rows)
    df_coll.to_excel(os.path.join(out_dir, 'collision_metrics.xlsx'), index=False)
    logging.info("Collision metrics written to %s", out_dir)

    # print averages
    logging.info("\n=== Average Metrics per Surface ===")
    for surf in surface_names:
        vals = [all_results[sub][surf] for sub in subjects]
        cham = [v['Chamfer'] for v in vals]
        assd = [v['ASSD'] for v in vals]
        hd = [v['HD'] for v in vals]
        sif = [v['SIF'] for v in vals if not np.isnan(v['SIF'])]
        logging.info(f"{surf}: Chamfer={np.mean(cham):.4f}±{np.std(cham):.4f}, ASSD={np.mean(assd):.4f}±{np.std(assd):.4f}, HD={np.mean(hd):.4f}±{np.std(hd):.4f}, SIF={np.mean(sif):.2f}±{np.std(sif):.2f}%")

    # print collision summary
    logging.info("\n=== Collision Summary ===")
    for k, ints in coll_stats.items():
        logging.info(f"{k}: avg intersections={np.mean(ints):.1f}±{np.std(ints):.1f}")

if __name__ == "__main__":
    eval_app()
