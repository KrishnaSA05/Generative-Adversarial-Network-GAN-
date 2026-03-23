"""
Generator Network — Radar View DCGAN.

Fixes vs. notebook:
  - Output activation: ReLU  →  Tanh  (correct for DCGAN with [-1,1] normalisation)
  - All magic numbers replaced with constructor args
"""
import torch.nn as nn


class Generator(nn.Module):
    """Maps a latent noise vector to a synthetic radar image."""

    def __init__(self, latent_dim: int, channels: int, image_size: tuple):
        super().__init__()
        H, W = image_size
        self.init_h, self.init_w = H // 4, W // 4   # 16×16

        self.model = nn.Sequential(
            # Project + reshape
            nn.Linear(latent_dim, 128 * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, self.init_h, self.init_w)),

            # 16×16 → 32×32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, momentum=0.8),
            nn.ReLU(inplace=True),

            # 32×32 → 64×64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=0.8),
            nn.ReLU(inplace=True),

            # Output  ← FIX: Tanh replaces ReLU
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)
