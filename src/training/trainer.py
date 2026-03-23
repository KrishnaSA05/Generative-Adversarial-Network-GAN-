"""
DCGAN Training Loop.

Fixes vs. notebook:
  1.  Discriminator trained FIRST  (standard DCGAN order)
  2.  Checkpoint saved every cfg[training][save_every] epochs
  3.  Sample images saved every cfg[training][sample_every] epochs
  4.  Python logging  (no bare print)
  5.  TensorBoard — scalars, histograms, image grids
"""
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.checkpoint import save_checkpoint
from src.utils.visualize   import save_image_grid

logger = logging.getLogger(__name__)


def train(generator, discriminator, train_dataloader: DataLoader,
          cfg: dict, device: torch.device):
    """
    Full DCGAN training run.

    Returns:
        d_losses, g_losses  (per-epoch float lists)
    """
    latent_dim   = cfg["model"]["latent_dim"]
    epochs       = cfg["training"]["epochs"]
    lr           = cfg["training"]["lr"]
    beta1        = cfg["training"]["beta1"]
    beta2        = cfg["training"]["beta2"]
    save_every   = cfg["training"]["save_every"]
    sample_every = cfg["training"]["sample_every"]
    ckpt_dir     = cfg["paths"]["checkpoints"]
    img_dir      = cfg["paths"]["generated_images"]
    log_dir      = cfg["paths"]["logs"]
    n_samples    = cfg["training"]["num_samples"]

    criterion   = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(),     lr=lr, betas=(beta1, beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    generator.to(device).train()
    discriminator.to(device).train()

    # ── TensorBoard writer ────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs → {log_dir}  |  run: tensorboard --logdir {log_dir}")

    d_losses, g_losses = [], []

    for epoch in range(1, epochs + 1):
        epoch_d, epoch_g = 0.0, 0.0

        for real_imgs, _ in train_dataloader:
            batch     = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_lbl  = torch.ones(batch,  1, device=device)
            fake_lbl  = torch.zeros(batch, 1, device=device)

            # ── Step 1 : Train Discriminator FIRST ────────────────────────
            optimizer_D.zero_grad()
            z         = torch.randn(batch, latent_dim, device=device)
            fake_imgs = generator(z).detach()
            d_loss    = (criterion(discriminator(real_imgs), real_lbl) +
                         criterion(discriminator(fake_imgs), fake_lbl)) / 2.0
            d_loss.backward()
            optimizer_D.step()

            # ── Step 2 : Train Generator SECOND ───────────────────────────
            optimizer_G.zero_grad()
            z         = torch.randn(batch, latent_dim, device=device)
            fake_imgs = generator(z)
            g_loss    = criterion(discriminator(fake_imgs), real_lbl)
            g_loss.backward()
            optimizer_G.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()

        avg_d = epoch_d / len(train_dataloader)
        avg_g = epoch_g / len(train_dataloader)
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        # ── Logging ───────────────────────────────────────────────────────
        logger.info(f"Epoch [{epoch:03d}/{epochs}]  "
                    f"D_Loss: {avg_d:.5f}  G_Loss: {avg_g:.5f}")

        # ── TensorBoard scalars ───────────────────────────────────────────
        writer.add_scalar("Loss/Discriminator", avg_d,  epoch)
        writer.add_scalar("Loss/Generator",     avg_g,  epoch)
        writer.add_scalars("Loss/Combined", {"D": avg_d, "G": avg_g}, epoch)

        # ── TensorBoard weight histograms (every 5 epochs) ────────────────
        if epoch % 5 == 0:
            for name, param in generator.named_parameters():
                writer.add_histogram(f"Generator/{name}",     param.data, epoch)
            for name, param in discriminator.named_parameters():
                writer.add_histogram(f"Discriminator/{name}", param.data, epoch)

        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % save_every == 0:
            save_checkpoint(generator, discriminator,
                            optimizer_G, optimizer_D,
                            epoch, avg_d, avg_g, ckpt_dir)

        # ── Sample images ─────────────────────────────────────────────────
        if epoch % sample_every == 0:
            img_path = save_image_grid(generator, latent_dim, device,
                                       epoch, img_dir, n=n_samples)
            # TensorBoard image grid
            import torchvision
            with torch.inference_mode():
                torch.manual_seed(42)
                z      = torch.randn(n_samples, latent_dim, device=device)
                grid   = generator(z)                       # (N,1,64,64)
                grid   = (grid + 1.0) / 2.0                # [-1,1]→[0,1]
                grid   = torchvision.utils.make_grid(grid, nrow=5, normalize=False)
            writer.add_image("Generated/RadarImages", grid, epoch)

    writer.flush()
    writer.close()
    logger.info("Training complete.  "
                f"Start TensorBoard:  tensorboard --logdir {log_dir}")
    return d_losses, g_losses
