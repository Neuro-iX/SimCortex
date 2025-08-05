# src/simcortex/segmentation/evaluate.py

import os
import csv
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from simcortex.segmentation.data import map_labels, EvalSegDataset
from simcortex.segmentation.loss import compute_dice


def evaluate_one(cfg):
    """
    Evaluate segmentation predictions against ground truth on the specified split.
    Writes metrics.csv to cfg.outputs.eval_save_dir
    """
    # Prepare device
    device = torch.device(cfg.trainer.device)

    # DataLoader for evaluation
    ds = EvalSegDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.trainer.img_batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
    )

    # Ensure output directory exists
    os.makedirs(cfg.outputs.eval_save_dir, exist_ok=True)
    csv_path = os.path.join(cfg.outputs.eval_save_dir, 'metrics.csv')

    all_dice = []
    all_iou  = []
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['subject', 'dice', 'iou'])

        for gt_arr, filled_arr, pred_arr, sub in loader:
            # Each is batch of size 1
            gt = gt_arr.squeeze(0).numpy().astype(np.int32)
            filled = filled_arr.squeeze(0).numpy().astype(np.int32)
            pred = pred_arr.squeeze(0).numpy().astype(np.int32)
            subj = sub[0] if isinstance(sub, (list, tuple)) else sub

            # Map labels to 0..8 classes
            gt_map = map_labels(gt, filled)

            # One-hot (drop background)
            gt_oh = torch.from_numpy(gt_map).long()
            gt_oh = torch.nn.functional.one_hot(gt_oh, num_classes=9)  # [D,H,W,9]
            gt_oh = gt_oh.permute(3,0,1,2)[1:].unsqueeze(0).float()  # [1,8,D,H,W]

            pred_idx = torch.from_numpy(pred).long()
            pred_oh = torch.nn.functional.one_hot(pred_idx, num_classes=9)
            pred_oh = pred_oh.permute(3,0,1,2)[1:].unsqueeze(0).float()

            # Dice
            d = compute_dice(pred_oh, gt_oh, dim='3d')
            all_dice.append(d)

            # IoU
            inter = (pred_oh.bool() & gt_oh.bool()).float().sum(dim=(2,3,4))
            union = (pred_oh.bool() | gt_oh.bool()).float().sum(dim=(2,3,4)).clamp_min(1e-8)
            iou = (inter / union).mean().item()
            all_iou.append(iou)

            writer.writerow([subj, f"{d:.4f}", f"{iou:.4f}"])

    print(f"Average Dice: {np.mean(all_dice):.4f}, Average IoU: {np.mean(all_iou):.4f}")


@hydra.main(
    version_base="1.1",
    config_path="../../../configs/segmentation",
    config_name="eval"
)
def evaluate_app(cfg):
    evaluate_one(cfg)


if __name__ == '__main__':
    evaluate_app()
