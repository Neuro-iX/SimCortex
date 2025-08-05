# src/simcortex/segmentation/predict.py

import os
import torch
import nibabel as nib
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf

from simcortex.segmentation.data import PredictSegDataset
from simcortex.segmentation.model import Unet


def predict_one(cfg):
    """
    Run segmentation inference on the specified split and save NIfTI predictions.
    """
    # Device setup
    device = torch.device(cfg.trainer.device)

    # DataLoader for inference
    ds = PredictSegDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
    )

    # Model instantiation
    model = Unet(c_in=cfg.model.in_channels, c_out=cfg.model.out_channels).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load checkpoint and strip DataParallel prefix if needed
    raw_state = torch.load(cfg.outputs.model_path, map_location=device)
    state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in raw_state.items()
    }
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    model.eval()

    os.makedirs(cfg.outputs.pred_save_dir, exist_ok=True)

    # Inference loop
    for batch in loader:
        vol, subid, affine = batch
        # Unbatch subject and affine
        sub = subid[0] if isinstance(subid, (list, tuple)) else subid
        if isinstance(affine, torch.Tensor):
            aff = affine.squeeze(0).cpu().numpy()
        else:
            aff = affine.squeeze(0) if hasattr(affine, 'ndim') and affine.ndim == 3 else affine

        vol = vol.to(device)  # [1,1,D,H,W]
        with torch.no_grad():
            logits = model(vol)
            pred = logits.argmax(dim=1).cpu().squeeze(0).numpy().astype("int32")

        # Save as NIfTI
        out_path = os.path.join(cfg.outputs.pred_save_dir, f"{sub}_prediction_mni.nii.gz")
        nib.save(nib.Nifti1Image(pred, aff), out_path)
        print(f"Saved prediction → {out_path}")


@hydra.main(
    version_base="1.1",
    config_path="../../../configs/segmentation",
    config_name="predict"
)
def predict_app(cfg):
    predict_one(cfg)


if __name__ == "__main__":
    predict_app()
