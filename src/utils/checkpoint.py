"""Checkpoint save / load utilities."""
import os, logging, torch
logger = logging.getLogger(__name__)


def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D,
                    epoch, d_loss, g_loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"dcgan_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch, "d_loss": d_loss, "g_loss": g_loss,
        "generator_state_dict":     generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_G_state_dict":   optimizer_G.state_dict(),
        "optimizer_D_state_dict":   optimizer_D.state_dict(),
    }, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path, generator, discriminator,
                    optimizer_G=None, optimizer_D=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    if optimizer_G: optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
    if optimizer_D: optimizer_D.load_state_dict(ckpt["optimizer_D_state_dict"])
    logger.info(f"Loaded checkpoint ← {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["d_loss"], ckpt["g_loss"]
