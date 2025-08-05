import torch
import torch.nn.functional as F

def merge_multiple_meshes(mesh_list):
    merged_verts, merged_faces, offset = [], [], 0
    for m in mesh_list:
        v, f = m.verts_packed(), m.faces_packed()
        merged_verts.append(v)
        merged_faces.append(f + offset)
        offset += v.shape[0]
    return torch.cat(merged_verts,0), torch.cat(merged_faces,0)

def pad_to_multiple_of_8(x: torch.Tensor) -> torch.Tensor:
    _,_,D,H,W = x.shape
    pd, ph, pw = [(8 - d%8)%8 for d in (D,H,W)]
    # pad format (Wl,Wr,Hl,Hr,Dl,Dr)
    return F.pad(x, (0,pw,0,ph,0,pd), 'constant', 0)

def world_to_voxel(verts_mm: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    invA = torch.inverse(affine)
    N = verts_mm.shape[0]
    ones = torch.ones(N,1, device=verts_mm.device)
    homog = torch.cat([verts_mm, ones], dim=1)  # (N,4)
    vox4 = (invA @ homog.t()).t()
    return vox4[:,:3]
