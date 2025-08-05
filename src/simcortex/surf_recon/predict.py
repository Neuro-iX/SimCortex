# src/simcortex/surf_recon/predict.py

import os
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from simcortex.surf_recon.data  import csr_dataset_factory, NormalizeMRIVoxels, collate_CSRData_fn
from simcortex.surf_recon.model import SurfDeform
from simcortex.surf_recon.utils import merge_multiple_meshes, pad_to_multiple_of_8, world_to_voxel

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import trimesh
import numpy as np

logger = logging.getLogger(__name__)

# Order must match merge logic
surface_names = ["lh_pial", "lh_white", "rh_pial", "rh_white"]


@hydra.main(config_path="../../../configs/surf_recon", config_name="predict")
def predict_app(cfg: DictConfig) -> None:
    logger.info("Unified Cortical Surface Reconstruction – Predict\n%s", OmegaConf.to_yaml(cfg))

    out_dir = cfg.outputs.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # build dataset + loader
    transforms = {"mri": NormalizeMRIVoxels("mean_std")}
    predict_ds = csr_dataset_factory(
        None, transforms,
        dataset_path=cfg.dataset.path,
        split_file=cfg.dataset.split_file,
        split_name=cfg.dataset.test_split_name,
        surface_name=cfg.dataset.surface_name,
        initial_surface_path=cfg.dataset.initial_surface_path,
    )
    loader = DataLoader(
        predict_ds,
        batch_size=cfg.predict.img_batch_size,
        collate_fn=collate_CSRData_fn,
        shuffle=False,
        num_workers=cfg.predict.num_workers,
        pin_memory=True,
    )
    logger.info("Loaded %d subjects for prediction", len(predict_ds))

    # model
    device = torch.device(cfg.predict.device)
    model = SurfDeform(
        C_hid=cfg.model.c_hid,
        C_in=1,
        inshape=cfg.model.inshape,
        sigma=cfg.model.sigma,
        device=device,
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # load checkpoint
    ckpt = cfg.predict.checkpoint_path
    sd = torch.load(ckpt, map_location=device)
    clean_sd = {
        k.replace("module.", ""):v
        for k,v in sd.items() if not k.startswith("n_averaged")
    }
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    target.load_state_dict(clean_sd)
    model.eval()
    logger.info("Loaded checkpoint %s", ckpt)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            # 1) preprocess MRI
            mri = batch["mri_vox"].to(device)
            if mri.ndim == 4:      mri = mri.unsqueeze(1)
            if mri.shape[1] > 1:   mri = mri.mean(1, keepdim=True)
            vol = pad_to_multiple_of_8(mri)

            # 2) build per-subject merged init meshes
            B = vol.shape[0]
            merged_verts, counts, faces, affines = [], [], [], []
            for i in range(B):
                meshes, cnts, fcs = [], [], []
                A = batch["mri_affine"][i].to(device)
                affines.append(A.cpu().numpy())
                for surf in surface_names:
                    iv = batch["py3d_init_meshes"][surf].verts_list()[i]
                    ifa= batch["py3d_init_meshes"][surf].faces_list()[i]
                    vox = world_to_voxel(iv.to(device), A)
                    meshes.append(Meshes(verts=[vox], faces=[ifa.to(device)]))
                    cnts.append(vox.shape[0])
                    fcs.append(ifa.cpu().numpy())
                counts.append(cnts)
                faces.append(fcs)
                mv, _ = merge_multiple_meshes(meshes)
                merged_verts.append(mv)

            # 3) forward
            padded = pad_sequence(merged_verts, batch_first=True).to(device)
            lengths = torch.tensor([v.shape[0] for v in merged_verts], device=device)
            pred = model(padded, vol, cfg.predict.n_steps)

            # 4) split back & save
            for i in range(B):
                subj = batch["subject"][i]
                # take only first `lengths[i]` verts
                p = pred[i, :lengths[i]].cpu().numpy()
                splits = torch.split(torch.from_numpy(p), counts[i], dim=0)
                A = affines[i]

                for j, surf in enumerate(surface_names):
                    verts_vox = splits[j].numpy()
                    # to world mm
                    ones = np.ones((verts_vox.shape[0],1), np.float32)
                    vox_h = np.concatenate([verts_vox, ones], axis=1)
                    world = (A @ vox_h.T).T[:, :3]

                    hemi, tissue = surf.split('_')
                    fname = f"pred_{subj}_{hemi}_{tissue}.ply"
                    outp = os.path.join(out_dir, fname)
                    trimesh.Trimesh(vertices=world, faces=faces[i][j], process=False).export(outp)
                    logger.info("Saved %s", outp)

    logger.info("Prediction done.")


if __name__ == "__main__":
    predict_app()
