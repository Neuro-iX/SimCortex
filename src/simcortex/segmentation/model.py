# src/simcortex/segmentation/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    """
    3D U-Net for multi-class MRI segmentation.
    
    Args:
      - c_in: number of input channels (e.g., 1 for T1)
      - c_out: number of output classes
    """

    def __init__(self, c_in: int = 1, c_out: int = 9):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv3d(c_in, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1)

        # Decoder
        self.deconv4 = nn.Conv3d(128 + 128, 64,  kernel_size=3, padding=1)
        self.deconv3 = nn.Conv3d(64  + 64,  32,  kernel_size=3, padding=1)
        self.deconv2 = nn.Conv3d(32  + 32,  16,  kernel_size=3, padding=1)
        self.deconv1 = nn.Conv3d(16  + 16,  16,  kernel_size=3, padding=1)

        # Final conv layers
        self.lastconv1 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.lastconv2 = nn.Conv3d(16, c_out, kernel_size=3, padding=1)

        # Upsample module
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = F.leaky_relu(self.conv1(x),  negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2)

        # Decoder stage 1
        x = self.up(x5)
        x = self._resize_to(x, x4)
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), negative_slope=0.2)

        # Decoder stage 2
        x = self.up(x)
        x = self._resize_to(x, x3)
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), negative_slope=0.2)

        # Decoder stage 3
        x = self.up(x)
        x = self._resize_to(x, x2)
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), negative_slope=0.2)

        # Decoder stage 4
        x = self.up(x)
        x = self._resize_to(x, x1)
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.2)

        # Final convs
        x = F.leaky_relu(self.lastconv1(x), negative_slope=0.2)
        x = self.lastconv2(x)
        return x

    @staticmethod
    def _resize_to(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Center-crop or interpolate src tensor to match spatial dims of tgt.
        """
        if src.shape[2:] != tgt.shape[2:]:
            return F.interpolate(src, size=tgt.shape[2:], mode='trilinear', align_corners=False)
        return src
