# src/simcortex/initial_surf/generate.py

import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import trimesh
from tqdm import tqdm
from trimesh.collision import CollisionManager

from simcortex.initial_surf.mesh import seg2surf, voxel_to_world
from simcortex.segmentation.model import Unet

def save_surface_mask_and_mesh(
    seg_pred_npy, out_dir, subject_id, brain_affine,
    left_white_level, right_white_level, pial_level,
    sigma, alpha, n_smooth, target_surface="all"
):
    os.makedirs(out_dir, exist_ok=True)
    if target_surface == "all":
        surfaces = {
            "lh_pial":  (seg_pred_npy==1)|(seg_pred_npy==3)|(seg_pred_npy==7),
            "rh_pial":  (seg_pred_npy==2)|(seg_pred_npy==4)|(seg_pred_npy==8),
            "lh_white": (seg_pred_npy==1)|(seg_pred_npy==7),
            "rh_white": (seg_pred_npy==2)|(seg_pred_npy==8),
        }
    else:
        surfaces = { target_surface: seg_pred_npy==int(target_surface.split("_")[0][-1]) }

    for name, mask in surfaces.items():
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.sum()==0:
            print(f"[{subject_id}] {name} empty → skip")
            continue

        # save mask
        nib.save(
          nib.Nifti1Image(mask_u8, brain_affine),
          os.path.join(out_dir, f"{subject_id}_mask_{name}.nii.gz")
        )

        # pick level
        lvl = {
          "lh_white": left_white_level,
          "rh_white": right_white_level
        }.get(name, pial_level)

        # marching cubes
        verts, faces = seg2surf(
          mask_u8,
          sigma=sigma,
          alpha=alpha,
          level=lvl,
          n_smooth=n_smooth
        )
        verts_world = voxel_to_world(verts, brain_affine)
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces)
        mesh.export(os.path.join(out_dir, f"{subject_id}_{name}.ply"))
        print(f"[{subject_id}] wrote {name}.ply")

def run_generation(cfg):
    # load model
    dev = torch.device(cfg.device)
    ck = cfg.model_seg_dir
    if os.path.isdir(ck):
        ck = os.path.join(ck, next(f for f in os.listdir(ck) if f.endswith(".pt")))
    sd = torch.load(ck, map_location=dev)

    model = Unet(1,9).to(dev)
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    stripped = {k.replace("module.",""):v for k,v in sd.items()}
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    target.load_state_dict(stripped)
    model.eval()

    # read CSV, pick subjects
    df = pd.read_csv(cfg.csv_split_path)
    subs = df[df["split"]=="test"]["subject"].astype(str).tolist()

    for sub in tqdm(subs, desc="generating surfaces"):
        img  = nib.load(os.path.join(cfg.data_dir, sub, f"{sub}_t1w_mni.nii.gz"))
        vol  = img.get_fdata().astype(np.float32)
        if vol.max()>1: vol /= vol.max()
        inp  = torch.from_numpy(vol[None,None]).to(dev)

        with torch.no_grad():
            seg = model(inp).argmax(1).cpu().squeeze(0).numpy()

        # find collision‐free pial
        p = cfg.levels.pial_start
        while True:
            lv, lf = seg2surf((seg==1)|(seg==3)|(seg==7),
                              sigma=cfg.mesh.sigma,
                              alpha=cfg.mesh.alpha,
                              level=p,
                              n_smooth=cfg.mesh.n_smooth)
            rv, rf = seg2surf((seg==2)|(seg==4)|(seg==8),
                              sigma=cfg.mesh.sigma,
                              alpha=cfg.mesh.alpha,
                              level=p,
                              n_smooth=cfg.mesh.n_smooth)
            cm = CollisionManager()
            cm.add_object("L", trimesh.Trimesh(voxel_to_world(lv,img.affine), lf))
            cm.add_object("R", trimesh.Trimesh(voxel_to_world(rv,img.affine), rf))
            if cm.in_collision_internal():
                p -= cfg.levels.pial_step
            else:
                break

        wl = p - cfg.levels.white_offset
        # left‐white vs pial
        while True:
            wv, wf = seg2surf((seg==1)|(seg==7),
                              sigma=cfg.mesh.sigma,
                              alpha=cfg.mesh.alpha,
                              level=wl,
                              n_smooth=cfg.mesh.n_smooth)
            pv, pf = seg2surf((seg==1)|(seg==3)|(seg==7),
                              sigma=cfg.mesh.sigma,
                              alpha=cfg.mesh.alpha,
                              level=p,
                              n_smooth=cfg.mesh.n_smooth)
            cm = CollisionManager()
            cm.add_object("W", trimesh.Trimesh(voxel_to_world(wv,img.affine), wf))
            cm.add_object("P", trimesh.Trimesh(voxel_to_world(pv,img.affine), pf))
            if cm.in_collision_internal():
                wl -= cfg.levels.pial_step
            else:
                break

        wr = p - cfg.levels.white_offset
        # right‐white vs pial
        while True:
            wv, wf = seg2surf((seg==2)|(seg==8),
                              sigma=cfg.mesh.sigma,
                              alpha=cfg.mesh.alpha,
                              level=wr,
                              n_smooth=cfg.mesh.n_smooth)
            pv, pf = seg2surf((seg==2)|(seg==4)|(seg==8),
                              sigma=cfg.mesh.sigma,
                              alpha=cfg.mesh.alpha,
                              level=p,
                              n_smooth=cfg.mesh.n_smooth)
            cm = CollisionManager()
            cm.add_object("W", trimesh.Trimesh(voxel_to_world(wv,img.affine), wf))
            cm.add_object("P", trimesh.Trimesh(voxel_to_world(pv,img.affine), pf))
            if cm.in_collision_internal():
                wr -= cfg.levels.pial_step
            else:
                break

        print(f"[{sub}] pial={p:.3f}, wl={wl:.3f}, wr={wr:.3f}")

        outd = os.path.join(cfg.result_dir, sub)
        save_surface_mask_and_mesh(
            seg, outd, sub, img.affine,
            left_white_level=wl,
            right_white_level=wr,
            pial_level=p,
            sigma=cfg.mesh.sigma,
            alpha=cfg.mesh.alpha,
            n_smooth=cfg.mesh.n_smooth,
            target_surface=cfg.target_surface
        )
