"""generate.py — Vanilla DCGAN: generate images from a checkpoint.

Usage:
    python generate.py --checkpoint outputs/dcgan/checkpoints/dcgan_epoch_0030.pt
"""
import argparse, os, sys
import torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from src.models import Generator
from src.utils  import save_image_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--n",          type=int, default=25)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/dcgan/generated_images")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G      = Generator(cfg["model"]["latent_dim"], cfg["data"]["channels"], tuple(cfg["data"]["image_size"]))
    ckpt   = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["generator_state_dict"])
    G.to(device).eval()
    save_image_grid(G, cfg["model"]["latent_dim"], device,
                    ckpt.get("epoch", 0), args.output_dir, n=args.n, seed=args.seed)
    print(f"Generated {args.n} images → {args.output_dir}")

if __name__ == "__main__": main()
