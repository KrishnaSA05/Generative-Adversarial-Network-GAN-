"""
train.py — Train Vanilla DCGAN.

Usage:
    python train.py
    python train.py --config configs/config.yaml
"""
import argparse, logging, os, sys
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from src.models   import Generator, Discriminator
from src.data     import load_datasets
from src.training import train
from src.utils    import save_loss_curves

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "train.log")),
                  logging.StreamHandler(sys.stdout)])

def main():
    parser = argparse.ArgumentParser(description="Train Vanilla DCGAN")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    setup_logging(cfg["paths"]["logs"])
    logger = logging.getLogger("train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    train_ds, test_ds, labels = load_datasets(cfg)
    logger.info(f"Train: {len(train_ds)}  |  Test: {len(test_ds)}  |  Classes: {labels}")
    loader   = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, drop_last=True)
    G        = Generator(cfg["model"]["latent_dim"], cfg["data"]["channels"], tuple(cfg["data"]["image_size"]))
    D        = Discriminator(cfg["data"]["channels"], tuple(cfg["data"]["image_size"]))
    logger.info(f"TensorBoard: tensorboard --logdir {cfg['paths']['logs']}")
    d_l, g_l = train(G, D, loader, cfg, device)
    save_loss_curves(d_l, g_l, cfg["paths"]["generated_images"],
                     title="DCGAN Training Loss — Radar View Generator")
    logger.info("Done.")

if __name__ == "__main__": main()
