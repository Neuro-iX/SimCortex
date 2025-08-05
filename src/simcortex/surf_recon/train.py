# src/simcortex/surf_recon/train.py
"""
Training script for unified cortical surface reconstruction
"""
import os
import gc
import logging

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency

from simcortex.surf_recon.data import (
    csr_dataset_factory,
    NormalizeMRIVoxels,
    collate_CSRData_fn
) 
from simcortex.surf_recon.model import SurfDeform
from simcortex.surf_recon.utils import (
    merge_multiple_meshes,
    pad_to_multiple_of_8,
    world_to_voxel
)

logger = logging.getLogger(__name__)
surface_names = ["lh_pial", "lh_white", "rh_pial", "rh_white"]


def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def compute_loss_per_batch(
    pred, lengths,
    init_vs, init_faces,
    gt_vs,   gt_faces,
    counts_i, counts_g,
    cfg, device
):
    total_c = total_e = total_n = 0.0
    n_surfs = 0

    for b in range(pred.shape[0]):
        out_vs = pred[b, : lengths[b]]
        chunks_p = torch.split(out_vs, counts_i[b], dim=0)
        chunks_g = torch.split(gt_vs[b],    counts_g[b], dim=0)

        for j, surf in enumerate(surface_names):
            pv = chunks_p[j]
            gv = chunks_g[j]

            off_i = sum(counts_i[b][:j])
            off_g = sum(counts_g[b][:j])
            fi = init_faces[b][j].to(device) - off_i
            fg = gt_faces[b][j].to(device)   - off_g

            # Quick skip if raw arrays are zero length:
            if pv.shape[0] == 0 or fi.numel() == 0 or gv.shape[0] == 0 or fg.numel() == 0:
                logger.warning(f"Skipping empty surface {surf} in batch {b}: "
                               f"pv={pv.shape[0]} verts, fi={fi.numel()} faces, "
                               f"gv={gv.shape[0]} verts, fg={fg.numel()} faces")
                continue

            mesh_p = Meshes(verts=[pv], faces=[fi])
            mesh_g = Meshes(verts=[gv], faces=[fg])

            # Now double-check after creating the Meshes object:
            nv = mesh_p.num_verts_per_mesh()[0].item()
            nf = mesh_p.num_faces_per_mesh()[0].item()
            if nv == 0 or nf == 0:
                logger.warning(f"Skipping surface {surf} because merged mesh is empty "
                               f"(verts={nv}, faces={nf})")
                continue

            # And similarly for the GT mesh if you sample from it directly:
            # (uncomment if you ever sample p_g from mesh_g before chamfer)
            # ng = mesh_g.num_verts_per_mesh()[0].item()
            # nfg = mesh_g.num_faces_per_mesh()[0].item()
            # if ng == 0 or nfg == 0:
            #     logger.warning(f"Skipping GT surface {surf} because ground‐truth mesh is empty")
            #     continue

            # Safe to sample now
            ppts = sample_points_from_meshes(mesh_p,
                                             num_samples=cfg.trainer.points_per_image)
            gpts = sample_points_from_meshes(mesh_g,
                                             num_samples=cfg.trainer.points_per_image)
            c, _ = chamfer_distance(ppts, gpts)
            e    = mesh_edge_loss(mesh_p)
            n    = mesh_normal_consistency(mesh_p)

            total_c += c
            total_e += e
            total_n += n
            n_surfs += 1

    if n_surfs == 0:
        raise RuntimeError("All surfaces empty in this batch — something’s wrong with your meshes!")

    loss_c = total_c / n_surfs
    loss_e = total_e / n_surfs
    loss_n = total_n / n_surfs

    total_loss = (
        cfg.objective.chamfer_weight  * loss_c
      + cfg.objective.edge_loss_weight    * loss_e
      + cfg.objective.normal_weight       * loss_n
    )
    return total_loss, (loss_c, loss_e, loss_n)


@hydra.main(config_path="../../../configs/surf_recon", config_name="train")
def train_app(cfg: DictConfig) -> None:
    # merge any user override
    if cfg.user_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

    # set up TensorBoard
    os.makedirs(cfg.outputs.output_dir, exist_ok=True)
    tb = SummaryWriter(os.path.join(cfg.outputs.output_dir, "tb_logs"))

    # MRI normalization
    transforms = {"mri": T.Compose([NormalizeMRIVoxels("mean_std")])}

    # build datasets & loaders
    train_ds = csr_dataset_factory(
        None, transforms,
        dataset_path=cfg.dataset.path,
        split_file=cfg.dataset.split_file,
        split_name=cfg.dataset.train_split_name,
        surface_name=cfg.dataset.surface_name,
        initial_surface_path=cfg.dataset.initial_surface_path,
    )
    val_ds = csr_dataset_factory(
        None, transforms,
        dataset_path=cfg.dataset.path,
        split_file=cfg.dataset.split_file,
        split_name=cfg.dataset.val_split_name,
        surface_name=cfg.dataset.surface_name,
        initial_surface_path=cfg.dataset.initial_surface_path,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.trainer.img_batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        collate_fn=collate_CSRData_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.trainer.img_batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        collate_fn=collate_CSRData_fn,
    )

    logger.info("Loaded %d training / %d validation subjects",
                len(train_ds), len(val_ds))

    # model + optimizer
    device = torch.device(cfg.trainer.device)
    model = SurfDeform(
        C_hid=cfg.model.c_hid,
        C_in=1, 
        inshape=cfg.model.inshape,
        sigma=cfg.model.sigma,
        device=device
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.trainer.learning_rate,
        weight_decay=1e-3
    )

    best_ch = float("inf")

    # --- main loop ---
    for epoch in range(1, cfg.trainer.num_epochs + 1):
        model.train()
        clean_gpu()

        sum_loss = sum_c = sum_e = sum_n = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            # 1) MRI → padded volume
            mri = batch["mri_vox"].to(device)
            if mri.ndim == 4:    mri = mri.unsqueeze(1)
            if mri.shape[1] > 1: mri = mri.mean(dim=1, keepdim=True)
            vol = pad_to_multiple_of_8(mri)

            # 2) per-subject merge
            init_vs, gt_vs = [], []
            counts_i, counts_g = [], []
            init_faces, gt_faces = [], []

            for i in range(vol.shape[0]):
                meshes_i, meshes_g = [], []
                ci, cg = [], []
                fis, fgs = [], []

                A = batch["mri_affine"][i].to(device)
                for surf in surface_names:
                    iv  = batch["py3d_init_meshes"][surf].verts_list()[i]
                    ifa = batch["py3d_init_meshes"][surf].faces_list()[i]
                    gv  = batch["py3d_meshes"][surf].verts_list()[i]
                    gfa = batch["py3d_meshes"][surf].faces_list()[i]

                    vox_iv = world_to_voxel(iv.to(device), A)
                    vox_gv = world_to_voxel(gv.to(device), A)

                    ci.append(vox_iv.shape[0])
                    cg.append(vox_gv.shape[0])

                    meshes_i.append(Meshes(verts=[vox_iv],
                                            faces=[ifa.to(device)]))
                    meshes_g.append(Meshes(verts=[vox_gv],
                                            faces=[gfa.to(device)]))

                    fis.append(ifa)
                    fgs.append(gfa)

                mvi, _ = merge_multiple_meshes(meshes_i)
                mgv, _ = merge_multiple_meshes(meshes_g)

                init_vs.append(mvi)
                gt_vs.append(mgv)
                counts_i.append(ci)
                counts_g.append(cg)
                init_faces.append(fis)
                gt_faces.append(fgs)

            padded_i = pad_sequence(init_vs, batch_first=True).to(device)
            lengths = torch.tensor([v.shape[0] for v in init_vs],
                                   device=device)

            # 3) forward deformation
            pred = model(padded_i, vol, cfg.model.n_steps)

            # 4) compute losses
            loss, (lc, le, ln) = compute_loss_per_batch(
                pred, lengths,
                init_vs, init_faces,
                gt_vs,   gt_faces,
                counts_i, counts_g,
                cfg, device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_c    += lc.item()
            sum_e    += le.item()
            sum_n    += ln.item()
            n_batches += 1

        # log training scalars
        tb.add_scalar("train/total",   sum_loss / n_batches, epoch)
        tb.add_scalar("train/chamfer", sum_c    / n_batches, epoch)
        tb.add_scalar("train/edge",    sum_e    / n_batches, epoch)
        tb.add_scalar("train/normal",  sum_n    / n_batches, epoch)
        logger.info(f"Epoch {epoch} TRAIN → total={sum_loss/n_batches:.6f}, chamfer={sum_c/n_batches:.6f}")

        # validation every N epochs
        if epoch % cfg.trainer.validation_interval == 0:
            model.eval()
            val_c = 0.0; val_sf = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                    # same preprocess + merge + forward as above,
                    # then only Chamfer:
                    mri = batch["mri_vox"].to(device)
                    if mri.ndim==4:    mri=mri.unsqueeze(1)
                    if mri.shape[1]>1: mri=mri.mean(dim=1,keepdim=True)
                    vol = pad_to_multiple_of_8(mri)

                    init_vs, gt_vs, counts_i = [], [], []
                    init_faces = []
                    for i in range(vol.shape[0]):
                        meshes_i, meshes_g, ci = [], [], []
                        fis = []
                        A = batch["mri_affine"][i].to(device)
                        for surf in surface_names:
                            iv  = batch["py3d_init_meshes"][surf].verts_list()[i]
                            ifa = batch["py3d_init_meshes"][surf].faces_list()[i]
                            gv  = batch["py3d_meshes"][surf].verts_list()[i]
                            gfa = batch["py3d_meshes"][surf].faces_list()[i]
                            vox_iv = world_to_voxel(iv.to(device), A)
                            vox_gv = world_to_voxel(gv.to(device), A)
                            ci.append(vox_iv.shape[0])
                            meshes_i.append(Meshes(verts=[vox_iv],
                                                   faces=[ifa.to(device)]))
                            meshes_g.append(Meshes(verts=[vox_gv],
                                                   faces=[gfa.to(device)]))
                            fis.append(ifa)
                        mvi,_ = merge_multiple_meshes(meshes_i)
                        mgv,_ = merge_multiple_meshes(meshes_g)
                        init_vs.append(mvi)
                        gt_vs.append(mgv)
                        counts_i.append(ci)
                        init_faces.append(fis)

                    padded_i = pad_sequence(init_vs, batch_first=True).to(device)
                    lengths = torch.tensor([v.shape[0] for v in init_vs], device=device)
                    pred = model(padded_i, vol, cfg.model.n_steps)

                    for b in range(pred.shape[0]):
                        out_vs = pred[b, :lengths[b]]
                        chunks = torch.split(out_vs, counts_i[b], dim=0)
                        for j in range(len(surface_names)):
                            pv = chunks[j]
                            ifa = init_faces[b][j].to(device)
                            if pv.shape[0]==0 or ifa.numel()==0:
                                continue
                            mesh_p = Meshes(verts=[pv], faces=[ifa - sum(counts_i[b][:j])])
                            ppts   = sample_points_from_meshes(mesh_p,
                                                               num_samples=cfg.trainer.points_per_image)
                            c,_    = chamfer_distance(ppts, ppts)  # same for Pts vs Pts
                            val_c += c
                            val_sf += 1

            avg_val = (val_c / val_sf).item()
            tb.add_scalar("val/chamfer", avg_val, epoch)
            logger.info(f"Epoch {epoch} VAL → chamfer={avg_val:.6f}")

            if avg_val < best_ch:
                best_ch = avg_val
                ckpt_dir = os.path.join(cfg.outputs.output_dir, "model")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    (model.module if hasattr(model, "module") else model).state_dict(),
                    os.path.join(ckpt_dir, "best_model.pth")
                )
                logger.info(f"New best model @ epoch {epoch}: Chamfer={best_ch:.6f}")

    logger.info("Training complete.")


if __name__ == "__main__":
    train_app()
