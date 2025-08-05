# src/simcortex/segmentation/loss.py

import torch


def compute_dice(x: torch.Tensor, y: torch.Tensor, dim: str = '3d') -> float:
    """
    Compute the mean Dice score between prediction and ground truth.

    Args:
        x: one-hot prediction tensor of shape
           - 2D: [B, C, H, W] or [C, H, W]
           - 3D: [B, C, D, H, W] or [C, D, H, W]
        y: one-hot ground-truth tensor of same shape as x
        dim: '2d' or '3d' indicating spatial dimensionality

    Returns:
        float: average Dice score over batch and channels
    """
    # Add batch dimension if needed
    if dim == '3d' and x.dim() == 4:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    if dim == '2d' and x.dim() == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    if dim == '2d':
        # now [B, C, H, W]
        intersection = (2 * (x * y).sum(dim=(2, 3)))
        denominator  = (x.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1e-8)
        dice_per_batch = (intersection / denominator).mean(dim=-1)
    else:
        # now [B, C, D, H, W]
        intersection = (2 * (x * y).sum(dim=(2, 3, 4)))
        denominator  = (x.sum(dim=(2, 3, 4)) + y.sum(dim=(2, 3, 4)) + 1e-8)
        dice_per_batch = (intersection / denominator).mean(dim=-1)

    # Return the mean over the batch
    return dice_per_batch.mean().item()
