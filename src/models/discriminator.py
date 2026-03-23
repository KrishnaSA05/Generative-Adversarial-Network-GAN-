"""Discriminator Network — Radar View DCGAN."""
import torch.nn as nn


class Discriminator(nn.Module):
    """Classifies radar images as real or fake."""

    def __init__(self, channels: int, image_size: tuple):
        super().__init__()
        H, W = image_size

        self.features = nn.Sequential(
            nn.Conv2d(channels, 32,  3, 2, 1), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(32,       64,  3, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(64,       128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(128,      256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
        )
        flat_dim = (H // 8) * (W // 8) * 256   # 8×8×256 = 16384

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
