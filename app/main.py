"""
app/main.py — Streamlit interactive demo for the Radar View Generator.

Run:
    streamlit run app/main.py
"""
import os, sys
import numpy as np
import torch, yaml
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import Generator

st.set_page_config(page_title="Radar View Generator — DCGAN", page_icon="📡", layout="wide")
st.title("📡 Radar View Generator — DCGAN")
st.markdown("> **Synthetic radar image generation** | PyTorch DCGAN | "
            "Deep Learning Project — *Krishna Sanjay Ambekar*")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    config_path  = st.text_input("Config file", value="configs/config.yaml")
    ckpt_file    = st.file_uploader("Upload checkpoint (.pt)", type=["pt"])
    st.subheader("Generation controls")
    n_images     = st.slider("Number of images", 1, 25,  9)
    seed         = st.slider("Random seed",       0, 9999, 42)
    generate_btn = st.button("🚀 Generate", use_container_width=True)
    st.divider()
    st.caption("Model: DCGAN | Output: Tanh | Device: auto (CUDA/CPU)")

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

try:
    cfg = load_config(config_path)
except FileNotFoundError:
    st.error(f"Config not found at `{config_path}`."); st.stop()

latent_dim = cfg["model"]["latent_dim"]
channels   = cfg["data"]["channels"]
img_size   = tuple(cfg["data"]["image_size"])
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Info panels ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Model Info")
    st.json({"Architecture":"DCGAN","Latent Dim":latent_dim,
             "Image Size":f"{img_size[0]}×{img_size[1]}",
             "Channels":channels,"Device":str(device),
             "Dataset":"Radar (19 200 train / 4 800 test)"})
with col2:
    st.subheader("🧠 Generator Layers")
    st.code("Input      : (B, 100)\n"
            "Linear     : 100 → 32768\n"
            "Reshape    : (128, 16, 16)\n"
            "Upsample + Conv2d → (128, 32, 32)\n"
            "Upsample + Conv2d → ( 64, 64, 64)\n"
            "Conv2d + Tanh     → (  1, 64, 64)\n"
            "Output     : (B, 1, 64, 64)", language="text")

st.divider()

# ── Load generator from uploaded checkpoint ───────────────────────────────────
@st.cache_resource
def build_generator(ckpt_bytes):
    import io
    gen  = Generator(latent_dim, channels, img_size)
    ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
    gen.load_state_dict(ckpt["generator_state_dict"])
    gen.eval()
    return gen

if ckpt_file is None:
    st.info("👈 Upload a trained `.pt` checkpoint from the sidebar to start.")
else:
    generator = build_generator(ckpt_file.read())
    st.success(f"✅ Checkpoint loaded — `{ckpt_file.name}`")

    if generate_btn:
        with st.spinner("Generating…"):
            torch.manual_seed(seed)
            with torch.inference_mode():
                z    = torch.randn(n_images, latent_dim)
                imgs = generator(z).squeeze(1).numpy()
            imgs = np.clip((imgs + 1.0) / 2.0, 0, 1)

        side = int(np.ceil(n_images ** 0.5))
        fig, axes = plt.subplots(side, side, figsize=(side*2, side*2))
        for i, ax in enumerate(axes.flat):
            if i < n_images: ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.suptitle(f"Generated Radar Images  (seed={seed})", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption(f"Generated {n_images} synthetic radar images | seed = {seed} | device = {device}")
