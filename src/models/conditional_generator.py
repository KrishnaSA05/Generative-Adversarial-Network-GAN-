"""
Conditional DCGAN Generator.
Input  : noise z (B, latent_dim) + class label (B,)
Output : class-specific radar image (B, 1, 64, 64)
"""
import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    """Maps (noise, class_label) → class-specific synthetic radar image."""
    def __init__(self, latent_dim: int, num_classes: int,
                 embed_dim: int, channels: int, image_size: tuple):
        super().__init__()
        H, W = image_size
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        in_features    = latent_dim + embed_dim
        self.model = nn.Sequential(
            nn.Linear(in_features, 128 * (H//4) * (W//4)),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, H//4, W//4)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128, momentum=0.8), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64,  3, 1, 1), nn.BatchNorm2d(64,  momentum=0.8), nn.ReLU(inplace=True),
            nn.Conv2d(64, channels, 3, 1, 1), nn.Tanh(),
        )
    def forward(self, z, labels):
        emb = self.label_emb(labels)
        x   = torch.cat([z, emb], dim=1)
        return self.model(x)
