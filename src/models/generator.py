"""Vanilla DCGAN Generator."""
import torch.nn as nn

class Generator(nn.Module):
    """Maps a latent noise vector to a synthetic radar image."""
    def __init__(self, latent_dim: int, channels: int, image_size: tuple):
        super().__init__()
        H, W = image_size
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * (H//4) * (W//4)),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, H//4, W//4)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128, momentum=0.8), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64,  3, 1, 1), nn.BatchNorm2d(64,  momentum=0.8), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, z):
        return self.model(z)
