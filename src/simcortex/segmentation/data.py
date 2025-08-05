# src/simcortex/segmentation/data.py

import os
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# Define the label groups for  9 classes
LABEL_GROUPS = {
    1: {2, 5, 10, 11, 12, 13, 26, 28, 30, 31},                      # lh white matter
    2: {41, 44, 49, 50, 51, 52, 58, 60, 62, 63},                    # rh white matter
    3: set(range(1000, 1004)) | set(range(1005, 1036)),            # lh pial
    4: set(range(2000, 2004)) | set(range(2005, 2036)),            # rh pial
    5: {17, 18},                                                    # lh amyg/hip
    6: {53, 54},                                                    # rh amyg/hip
    7: {4},                                                        # lh vertical
    8: {43},                                                       # rh vertical
}

def map_labels(seg_arr: np.ndarray, filled_arr: np.ndarray) -> np.ndarray:
    """
    Map FreeSurfer labels (aparc+aseg) to 9-class segmentation.
    """
    seg_mapped = np.zeros_like(seg_arr, dtype=np.int32)
    # assign primary labels
    for cls, labels in LABEL_GROUPS.items():
        mask = np.isin(seg_arr, list(labels))
        seg_mapped[mask] = cls
    # resolve ambiguous 77/80 using filled image
    ambiguous = np.isin(seg_arr, [77, 80])
    seg_mapped[ambiguous & (filled_arr == 255)] = 1
    seg_mapped[ambiguous & (filled_arr == 127)] = 2
    return seg_mapped

class SegDataset(Dataset):
    """
    PyTorch Dataset for 3D MRI + 9-class segmentation.
    Expects:
      - cfg.dataset.path: base folder containing one subfolder per subject
      - cfg.dataset.split_file: CSV with 'subject' & 'split' columns
    """
    def __init__(self, cfg, split: str):
        self.data_dir = cfg.dataset.path
        df = pd.read_csv(cfg.dataset.split_file)
        # filter to requested split (train/val/test)
        self.subjects = df[df['split'] == split]['subject'].astype(str).tolist()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        subj_dir = os.path.join(self.data_dir, sub)
        # file paths
        t1_path    = os.path.join(subj_dir, f"{sub}_t1w_mni.nii.gz")
        seg_path   = os.path.join(subj_dir, f"{sub}_aparc_aseg_mni.nii.gz")
        fill_path  = os.path.join(subj_dir, f"{sub}_filled_mni.nii.gz")

        # load MRI volume and normalize to [0,1]
        vol = nib.load(t1_path).get_fdata().astype(np.float32)
        if vol.max() > 1:
            vol /= vol.max()
        # add channel dim: [1, D, H, W]
        vol = torch.from_numpy(vol[None])

        # load segmentation and filled arrays
        seg_arr   = nib.load(seg_path).get_fdata().astype(np.int32)
        filled_arr= nib.load(fill_path).get_fdata().astype(np.int32)
        # map to 9-class 
        seg9 = map_labels(seg_arr, filled_arr)
        seg9 = torch.from_numpy(seg9).long()  # [D, H, W]

        return vol, seg9

class PredictSegDataset(Dataset):
    """
    For inference on the 'test' split: returns (vol_tensor, subject_id, affine).
    """
    def __init__(self, cfg):
        df = pd.read_csv(cfg.dataset.split_file)
        # Allow overriding split name if you like
        split_name = getattr(cfg.dataset, "predict_split_name", "test")
        self.subjects = df[df["split"] == split_name]["subject"].astype(str).tolist()
        self.data_dir = cfg.dataset.path

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        subj_dir = os.path.join(self.data_dir, sub)
        t1_path = os.path.join(subj_dir, f"{sub}_t1w_mni.nii.gz")
        img = nib.load(t1_path)
        vol = img.get_fdata().astype(np.float32)
        if vol.max() > 1:
            vol = vol / vol.max()
        vol = torch.from_numpy(vol[None])  # [1, D, H, W]
        return vol, sub, img.affine



class EvalSegDataset(Dataset):
    """
    Dataset for evaluation: returns (gt_array, filled_array, pred_array, subject_id).
    Expects:
      - cfg.dataset.path        : base folder of MRI / aparc_aseg / filled NIfTIs
      - cfg.dataset.split_file  : CSV with 'subject','split'
      - cfg.outputs.pred_save_dir : folder where your predict step wrote *_prediction_mni.nii.gz
    """
    def __init__(self, cfg):
        self.data_dir     = cfg.dataset.path
        self.split_file   = cfg.dataset.split_file
        self.pred_dir     = cfg.outputs.pred_save_dir
        df = pd.read_csv(self.split_file)
        # you can override split name if desired:
        self.split_name  = getattr(cfg.dataset, "eval_split_name", "test")
        self.subjects    = df[df["split"] == self.split_name]["subject"].astype(str).tolist()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sub = self.subjects[idx]
        subdir = os.path.join(self.data_dir, sub)
        # load ground truth & filled
        gt_img     = nib.load(os.path.join(subdir, f"{sub}_aparc_aseg_mni.nii.gz"))
        filled_img = nib.load(os.path.join(subdir, f"{sub}_filled_mni.nii.gz"))
        # load prediction
        pred_img   = nib.load(os.path.join(self.pred_dir,   f"{sub}_prediction_mni.nii.gz"))

        gt_arr     = gt_img.get_fdata().astype(np.int32)
        filled_arr = filled_img.get_fdata().astype(np.int32)
        pred_arr   = pred_img.get_fdata().astype(np.int32)

        return gt_arr, filled_arr, pred_arr, sub