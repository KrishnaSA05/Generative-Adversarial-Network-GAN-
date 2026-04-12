"""
Conditional DCGAN Discriminator.
Input  : image (B, 1, 64, 64) + class label (B,)
Output : P(image is real AND belongs to claimed class) (B, 1)
"""
import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    """Verifies image is both realistic and the correct class."""
    def __init__(self, channels: int, num_classes: int, image_size: tuple):
        super().__init__()
        H, W = image_size
        self.label_emb = nn.Embedding(num_classes, H * W)
        self.H, self.W = H, W
        in_ch = channels + 1
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32,  3, 2, 1), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(32,    64,  3, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(64,    128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
            nn.Conv2d(128,   256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear((H//8)*(W//8)*256, 1), nn.Sigmoid()
        )
    def forward(self, img, labels):
        emb = self.label_emb(labels).view(-1, 1, self.H, self.W)
        x   = torch.cat([img, emb], dim=1)
        return self.classifier(self.features(x))
