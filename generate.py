"""
generate.py — Generate radar images from a saved checkpoint.

Usage:
    python generate.py --checkpoint outputs/checkpoints/dcgan_epoch_0030.pt
    python generate.py --checkpoint outputs/checkpoints/dcgan_epoch_0030.pt --n 25 --seed 7
"""
import argparse, os, sys
import torch, yaml

sys.path.insert(0, os.path.dirname(__file__))
from src.models import Generator
from src.utils  import save_image_grid


def main():
    parser = argparse.ArgumentParser(description="Generate radar images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--n",          type=int, default=25)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/generated_images")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = cfg["model"]["latent_dim"]
    channels   = cfg["data"]["channels"]
    img_size   = tuple(cfg["data"]["image_size"])

    generator = Generator(latent_dim, channels, img_size)
    ckpt      = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.to(device).eval()

    epoch = ckpt.get("epoch", 0)
    save_image_grid(generator, latent_dim, device, epoch,
                    args.output_dir, n=args.n, seed=args.seed)
    print(f"Generated {args.n} images → {args.output_dir}")


if __name__ == "__main__":
    main()
