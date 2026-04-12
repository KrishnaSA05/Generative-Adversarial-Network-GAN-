"""Conditional DCGAN training loop with TensorBoard + logging."""
import logging, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.utils.checkpoint import save_checkpoint
from src.utils.visualize   import save_conditional_image_grid
logger = logging.getLogger(__name__)

def train_conditional(generator, discriminator, train_dataloader: DataLoader, cfg, device):
    latent_dim   = cfg["model"]["latent_dim"]
    num_classes  = cfg["data"]["num_classes"]
    class_names  = {int(k): v for k, v in cfg["data"]["class_names"].items()}
    epochs       = cfg["training"]["epochs"]
    lr, b1, b2   = cfg["training"]["lr"], cfg["training"]["beta1"], cfg["training"]["beta2"]
    criterion    = nn.BCELoss()
    optimizer_G  = torch.optim.Adam(generator.parameters(),     lr=lr, betas=(b1, b2))
    optimizer_D  = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    generator.to(device).train(); discriminator.to(device).train()
    writer       = SummaryWriter(log_dir=cfg["paths"]["logs"])
    logger.info(f"TensorBoard → tensorboard --logdir {cfg['paths']['logs']}")
    d_losses, g_losses = [], []

    for epoch in range(1, epochs + 1):
        ed, eg = 0.0, 0.0
        for real_imgs, real_labels in train_dataloader:
            batch       = real_imgs.size(0)
            real_imgs   = real_imgs.to(device)
            real_labels = real_labels.to(device)
            valid = torch.ones(batch,  1, device=device)
            fake  = torch.zeros(batch, 1, device=device)
            # Discriminator
            optimizer_D.zero_grad()
            fake_labels = torch.randint(0, num_classes, (batch,), device=device)
            z           = torch.randn(batch, latent_dim, device=device)
            d_loss      = (criterion(discriminator(real_imgs, real_labels), valid) +
                           criterion(discriminator(generator(z, fake_labels).detach(), fake_labels), fake)) / 2
            d_loss.backward(); optimizer_D.step()
            # Generator
            optimizer_G.zero_grad()
            fake_labels = torch.randint(0, num_classes, (batch,), device=device)
            z           = torch.randn(batch, latent_dim, device=device)
            g_loss      = criterion(discriminator(generator(z, fake_labels), fake_labels), valid)
            g_loss.backward(); optimizer_G.step()
            ed += d_loss.item(); eg += g_loss.item()

        avg_d, avg_g = ed/len(train_dataloader), eg/len(train_dataloader)
        d_losses.append(avg_d); g_losses.append(avg_g)
        logger.info(f"Epoch [{epoch:03d}/{epochs}]  D_Loss: {avg_d:.5f}  G_Loss: {avg_g:.5f}")
        writer.add_scalars("Loss", {"Discriminator": avg_d, "Generator": avg_g}, epoch)
        if epoch % 5 == 0:
            save_checkpoint(generator, discriminator, optimizer_G, optimizer_D,
                            epoch, avg_d, avg_g, cfg["paths"]["checkpoints"])
            save_conditional_image_grid(generator, latent_dim, num_classes, class_names,
                                        device, epoch, cfg["paths"]["generated_images"],
                                        n_per_class=cfg["training"]["num_samples"] // num_classes)
    writer.flush(); writer.close()
    logger.info("Conditional training complete.")
    return d_losses, g_losses
