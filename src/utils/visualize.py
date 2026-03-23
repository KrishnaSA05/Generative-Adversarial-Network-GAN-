"""Visualisation helpers — image grids and loss curves."""
import os, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)


def save_image_grid(generator, latent_dim, device, epoch,
                    output_dir, n=25, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    side = int(n ** 0.5)
    generator.eval()
    with torch.inference_mode():
        torch.manual_seed(seed)
        z    = torch.randn(n, latent_dim, device=device)
        imgs = generator(z).squeeze(1).cpu().numpy()
    imgs = np.clip((imgs + 1.0) / 2.0, 0, 1)   # [-1,1] → [0,1]

    fig, axes = plt.subplots(side, side, figsize=(side*1.5, side*1.5))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.axis("off")
    fig.suptitle(f"Generated Radar Images — Epoch {epoch}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"samples_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=100); plt.close(fig)
    logger.info(f"Sample grid saved → {path}")
    generator.train()
    return path


def save_loss_curves(d_losses, g_losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(d_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, d_losses, label="Discriminator Loss", color="#E74C3C", lw=2)
    ax.plot(epochs, g_losses, label="Generator Loss",     color="#2E86C1", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("DCGAN Training Loss — Radar View Generator")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=120); plt.close(fig)
    logger.info(f"Loss curves saved → {path}")
