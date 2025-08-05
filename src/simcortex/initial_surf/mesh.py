# src/simcortex/initial_surf/mesh.py

import numpy as np
import torch
from scipy.ndimage import distance_transform_cdt as cdt
from skimage.filters import gaussian
from skimage.measure import marching_cubes, label as compute_cc
from nibabel.affines import apply_affine
from tca import topology
from nibabel.affines import apply_affine


# one shared topology correction instance
topo_correct = topology()

def laplacian_smooth(verts, faces, method="uniform", lambd=1.):
    v = verts[0]
    f = faces[0]

    with torch.no_grad():
        if method == "uniform":
            V = v.shape[0]
            edge = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0).T
            L = torch.sparse_coo_tensor(edge, torch.ones_like(edge[0]).float(), (V, V))
            norm_w = 1.0 / torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
    v_bar = L.mm(v) * norm_w  # new vertices    
    return ((1 - lambd) * v + lambd * v_bar).unsqueeze(0)

def seg2surf(seg,
             data_name='hcp-oasis',
             sigma=0.5,
             alpha=16,
             level=0.8,
             n_smooth=2):
    """
    Extract the surface based on the segmentation.
    
    seg: input segmentation
    sigma: standard deviation of guassian blurring
    alpha: threshold for obtaining boundary of topology correction
    level: extracted surface level for Marching Cubes
    n_smooth: iteration of Laplacian smoothing
    """
    
    # ------ connected components checking ------ 
    cc, nc = compute_cc(seg, connectivity=2, return_num=True)
    cc_id = 1 + np.argmax(np.array([np.count_nonzero(cc == i)\
                                    for i in range(1, nc+1)]))
    seg = (cc==cc_id).astype(np.float64)

    # ------ generate signed distance function ------       
    sdf = -cdt(seg) + cdt(1-seg)
    sdf = sdf.astype(float)         
    sdf = gaussian(sdf, sigma=sigma)

     # ------ topology correction ------
    sdf_topo= topo_correct.apply(sdf, threshold=alpha)

    # ------ marching cubes ------
    v_mc, f_mc, _, _ = marching_cubes(-sdf_topo, level=-level, method='lorensen')
    # ------ mesh smoothing ------
    v_mc = torch.Tensor(v_mc.copy()).unsqueeze(0)
    f_mc = torch.LongTensor(f_mc.copy()).unsqueeze(0)
    for j in range(n_smooth):  # Smooth and inflate the mesh
        v_mc = laplacian_smooth(v_mc, f_mc, 'uniform', lambd=1)
    v_mc = v_mc[0].cpu().numpy()
    f_mc = f_mc[0].cpu().numpy()

    return v_mc, f_mc


def voxel_to_world(verts, affine):
    return apply_affine(affine, verts)