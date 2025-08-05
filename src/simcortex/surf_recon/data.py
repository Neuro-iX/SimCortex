# src/simcortex/surf_recon/data.py

import os
import logging
import numpy as np
import pandas as pd
import torch
import trimesh
import nibabel
from torch.utils.data import Dataset
from pytorch3d.structures import Meshes

logger = logging.getLogger(__name__)

def mri_reader(path, hemisphere=None):
    """
    Load a NIfTI; optionally crop to one hemisphere.
    Returns (header, voxels, affine).
    """
    img = nibabel.load(path)
    if hemisphere:
        img = {
            'lh': img.slicer[75:171, 12:204, 10:170],
            'rh': img.slicer[ 9:105, 12:204, 10:170]
        }[hemisphere]
    data   = img.get_fdata().astype(np.float32)
    affine = img.affine.astype(np.float32)
    return img.header, data, affine

def mesh_reader(path):
    """
    Load a mesh (PLY/OBJ) via trimesh.
    Returns (verts, faces).
    """
    m = trimesh.load(path, process=False)
    verts = np.array(m.vertices, dtype=np.float32)
    faces = np.array(m.faces,    dtype=np.int64)
    return verts, faces

class NormalizeMRIVoxels:
    """
    In‐place normalization of data['mri_vox'].
    """
    def __init__(self, norm_type='mean_std', **kwargs):
        self.norm_type = norm_type
        self.params    = kwargs

    def __call__(self, item):
        vox = item['mri_vox']
        if self.norm_type == 'mean_std':
            m = self.params.get('mean', vox.mean())
            s = self.params.get('std',  vox.std())
            item['mri_vox'] = (vox - float(m)) / float(s)
        elif self.norm_type == 'min_max':
            vmin, vmax = vox.min(), vox.max()
            scale = float(self.params.get('scale', 1.0))
            item['mri_vox'] = ((vox - vmin) / (vmax - vmin + 1e-8)) * scale
        else:
            raise ValueError(f"Unknown norm_type '{self.norm_type}'")
        return item

class CSRDataset(Dataset):
    """
    Dataset for unified cortical surface reconstruction.
    """
    def __init__(self,
                 subjects: list,
                 mris: list,
                 surfaces: list = None,
                 hemisphere: str = None,
                 transforms: dict = None,
                 initial_surfaces: list = None):
        assert len(subjects) == len(mris)
        if surfaces is not None:
            assert len(subjects) == len(surfaces)
        if initial_surfaces is not None:
            assert len(subjects) == len(initial_surfaces)

        self.subjects          = subjects
        self.mris               = mris
        self.surfaces           = surfaces
        self.initial_surfaces   = initial_surfaces
        self.hemisphere         = hemisphere
        self.transforms         = transforms or {}

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        mri_path = self.mris[idx]

        item = {'subject': subj}
        _, vox, aff = mri_reader(mri_path, hemisphere=self.hemisphere)
        item['mri_vox']    = vox
        item['mri_affine'] = aff

        # GT surfaces
        if self.surfaces:
            vs, fs = {}, {}
            for key, p in self.surfaces[idx].items():
                v,f = mesh_reader(p)
                vs[key], fs[key] = v, f
            item['surf_verts'] = vs
            item['surf_faces'] = fs

        # Initial surfaces
        if self.initial_surfaces:
            ivs, ifs = {}, {}
            for key, p in self.initial_surfaces[idx].items():
                v,f = mesh_reader(p)
                ivs[key], ifs[key] = v, f
            item['init_surf_verts'] = ivs
            item['init_surf_faces'] = ifs

        # Apply MRI‐only transforms
        if 'mri' in self.transforms:
            item = self.transforms['mri'](item)

        return item

def csr_dataset_factory(hemisphere, transforms, **kwargs):
    """
    Build CSRDataset via:
      - split_file (csv with 'subject','split')
      - dataset_path
      - split_name
      - surface_name (list of keys)
      - initial_surface_path (optional)
    """
    df = pd.read_csv(kwargs['split_file'])
    df = df[df['split'] == kwargs['split_name']]
    subs = df['subject'].astype(str).tolist()

    mris = [os.path.join(kwargs['dataset_path'], s, f"{s}_t1w_mni.nii.gz")
            for s in subs]

    surfaces = []
    if kwargs.get('surface_name'):
        for s in subs:
            d = {}
            for key in kwargs['surface_name']:
                fn = f"{s}_{key}_mni.ply"
                d[key] = os.path.join(kwargs['dataset_path'], s, fn)
            surfaces.append(d)

    initials = None
    if kwargs.get('initial_surface_path'):
        initials = []
        for s in subs:
            d = {}
            for key in kwargs['surface_name']:
                fn = f"ini_{s}_{key}_mni.ply"
                d[key] = os.path.join(kwargs['initial_surface_path'], s, fn)
            initials.append(d)

    return CSRDataset(
        subjects=subs,
        mris=mris,
        surfaces=surfaces or None,
        initial_surfaces=initials,
        hemisphere=hemisphere,
        transforms=transforms
    )

def collate_CSRData_fn(batch):
    """
    Collate a list of CSRDataset items into batched tensors + PyTorch3D Meshes.
    """
    out = {
        'subject':    [x['subject'] for x in batch],
        'mri_vox':    torch.from_numpy(np.stack([x['mri_vox'] for x in batch],0)).float(),
        'mri_affine': torch.from_numpy(np.stack([x['mri_affine'] for x in batch],0)).float()
    }

    # GT
    if 'surf_verts' in batch[0]:
        md = {}
        for key in batch[0]['surf_verts'].keys():
            verts = [torch.from_numpy(x['surf_verts'][key]) for x in batch]
            faces = [torch.from_numpy(x['surf_faces'][key]).long() for x in batch]
            md[key] = Meshes(verts=verts, faces=faces)
        out['py3d_meshes'] = md

    # Initial
    if batch[0].get('init_surf_verts') is not None:
        im = {}
        for key in batch[0]['init_surf_verts'].keys():
            verts = [torch.from_numpy(x['init_surf_verts'][key]) for x in batch]
            faces = [torch.from_numpy(x['init_surf_faces'][key]).long() for x in batch]
            im[key] = Meshes(verts=verts, faces=faces)
        out['py3d_init_meshes'] = im

    return out
