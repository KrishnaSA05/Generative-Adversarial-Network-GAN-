"""
train_conditional.py — Train Conditional DCGAN.

Usage:
    python train_conditional.py
    python train_conditional.py --config configs/conditional_config.yaml
"""
import argparse, logging, os, sys
import torch, yaml
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from src.models   import ConditionalGenerator, ConditionalDiscriminator
from src.data     import load_datasets
from src.training import train_conditional
from src.utils    import save_loss_curves

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "train_conditional.log")),
                  logging.StreamHandler(sys.stdout)])

def main():
    parser = argparse.ArgumentParser(description="Train Conditional DCGAN")
    parser.add_argument("--config", default="configs/conditional_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    setup_logging(cfg["paths"]["logs"])
    logger = logging.getLogger("train_conditional")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    train_ds, test_ds, labels = load_datasets(cfg)
    class_names = {int(k): v for k, v in cfg["data"]["class_names"].items()}
    logger.info(f"Train: {len(train_ds)}  |  Classes: {class_names}")
    loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, drop_last=True)
    G = ConditionalGenerator(cfg["model"]["latent_dim"], cfg["data"]["num_classes"],
                              cfg["model"]["embed_dim"], cfg["data"]["channels"],
                              tuple(cfg["data"]["image_size"]))
    D = ConditionalDiscriminator(cfg["data"]["channels"], cfg["data"]["num_classes"],
                                  tuple(cfg["data"]["image_size"]))
    logger.info(f"Generator     params: {sum(p.numel() for p in G.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    logger.info(f"TensorBoard: tensorboard --logdir {cfg['paths']['logs']}")
    d_l, g_l = train_conditional(G, D, loader, cfg, device)
    save_loss_curves(d_l, g_l, cfg["paths"]["generated_images"],
                     title="Conditional DCGAN Training Loss — Radar View Generator")
    logger.info("Done.")

if __name__ == "__main__": main()
