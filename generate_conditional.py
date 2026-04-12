"""generate_conditional.py — Conditional DCGAN: generate class-specific images.

Usage:
    python generate_conditional.py --checkpoint outputs/cdcgan/checkpoints/dcgan_epoch_0030.pt
    python generate_conditional.py --checkpoint ... --target_class 2   # Cars only
"""
import argparse, os, sys
import torch, yaml
sys.path.insert(0, os.path.dirname(__file__))
from src.models import ConditionalGenerator
from src.utils  import save_conditional_image_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--config",       default="configs/conditional_config.yaml")
    parser.add_argument("--target_class", type=int, default=None,
                        help="If set, generate only this class. Else generate all classes.")
    parser.add_argument("--n",            type=int, default=8)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--output_dir",   default="outputs/cdcgan/generated_images")
    args = parser.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]
    class_names = {int(k): v for k, v in cfg["data"]["class_names"].items()}
    G = ConditionalGenerator(cfg["model"]["latent_dim"], num_classes,
                              cfg["model"]["embed_dim"], cfg["data"]["channels"],
                              tuple(cfg["data"]["image_size"]))
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["generator_state_dict"])
    G.to(device).eval()

    if args.target_class is not None:
        # Generate one class only
        import numpy as np, matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.output_dir, exist_ok=True)
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            z      = torch.randn(args.n, cfg["model"]["latent_dim"], device=device)
            labels = torch.full((args.n,), args.target_class, dtype=torch.long, device=device)
            imgs   = np.clip((G(z, labels).squeeze(1).cpu().numpy() + 1) / 2, 0, 1)
        side = int(args.n ** 0.5)
        fig, axes = plt.subplots(side, side, figsize=(side*2, side*2))
        for ax, img in zip(axes.flat, imgs): ax.imshow(img, cmap="gray"); ax.axis("off")
        name = class_names.get(args.target_class, str(args.target_class))
        fig.suptitle(f"Generated: {name}  (Class {args.target_class})")
        path = os.path.join(args.output_dir, f"class_{args.target_class}_{name}.png")
        plt.tight_layout(); plt.savefig(path, dpi=120); print(f"Saved → {path}")
    else:
        save_conditional_image_grid(G, cfg["model"]["latent_dim"], num_classes,
                                     class_names, device, ckpt.get("epoch", 0),
                                     args.output_dir, n_per_class=args.n, seed=args.seed)
        print(f"All-class grid saved → {args.output_dir}")

if __name__ == "__main__": main()
