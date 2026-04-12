"""Visualisation — image grids, per-class grids, and loss curves."""
import os, logging
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)

CLASS_COLORS = {0: "#AED6F1", 1: "#A9DFBF", 2: "#FAD7A0", 3: "#D2B4DE"}


def save_image_grid(generator, latent_dim, device, epoch,
                    output_dir, n=25, seed=42):
    """Save n generated images as a square PNG grid (vanilla DCGAN)."""
    os.makedirs(output_dir, exist_ok=True)
    side = int(n ** 0.5)
    generator.eval()
    with torch.inference_mode():
        torch.manual_seed(seed)
        imgs = generator(torch.randn(n, latent_dim, device=device))
        imgs = np.clip((imgs.squeeze(1).cpu().numpy() + 1.0) / 2.0, 0, 1)
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


def save_conditional_image_grid(generator, latent_dim, num_classes,
                                 class_names, device, epoch,
                                 output_dir, n_per_class=8, seed=42):
    """
    Save a per-class image grid for Conditional DCGAN.
    Rows = classes, Columns = generated samples.
    First column is a colour-coded class label box.
    """
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()

    fig, axes = plt.subplots(
        num_classes, n_per_class + 1,
        figsize=((n_per_class + 1) * 1.8, num_classes * 2.0),
        gridspec_kw={"width_ratios": [1.2] + [1] * n_per_class}
    )

    with torch.inference_mode():
        for cls_id in range(num_classes):
            torch.manual_seed(seed)
            z      = torch.randn(n_per_class, latent_dim, device=device)
            labels = torch.full((n_per_class,), cls_id, dtype=torch.long, device=device)
            imgs   = generator(z, labels).squeeze(1).cpu().numpy()
            imgs   = np.clip((imgs + 1.0) / 2.0, 0, 1)

            # Label box (column 0)
            ax_lbl = axes[cls_id, 0]
            ax_lbl.set_facecolor(CLASS_COLORS.get(cls_id, "#EEEEEE"))
            ax_lbl.text(0.5, 0.5, f"Class {cls_id}\n\n{class_names.get(cls_id, str(cls_id))}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", transform=ax_lbl.transAxes, color="#1A1A2E")
            ax_lbl.axis("off")

            for col, img in enumerate(imgs):
                ax = axes[cls_id, col + 1]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.axis("off")

    fig.suptitle(f"Conditional DCGAN — Generated Radar Images per Class  [Epoch {epoch}]",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"samples_epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    logger.info(f"Conditional sample grid saved → {path}")
    generator.train()
    return path


def save_loss_curves(d_losses, g_losses, output_dir, title="DCGAN Training Loss"):
    """Save generator vs discriminator loss curve PNG."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(d_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, d_losses, label="Discriminator Loss", color="#E74C3C", lw=2)
    ax.plot(epochs, g_losses, label="Generator Loss",     color="#2E86C1", lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=120); plt.close(fig)
    logger.info(f"Loss curves saved → {path}")
