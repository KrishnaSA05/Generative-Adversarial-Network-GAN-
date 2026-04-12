"""
app/main.py — Streamlit demo supporting BOTH Vanilla DCGAN and Conditional DCGAN.

Run:
    streamlit run app/main.py
"""
import os, sys, io
import numpy as np
import torch, yaml
import streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import Generator, ConditionalGenerator

st.set_page_config(page_title="Radar View Generator", page_icon="📡", layout="wide")
st.title("📡 Radar View Generator")
st.markdown("> **Synthetic radar image generation** | DCGAN & Conditional DCGAN | "
            "*Krishna Sanjay Ambekar*")
st.divider()

CLASS_NAMES  = {0: "🚶 Pedestrian", 1: "🛴 Electric Scooter", 2: "🚗 Car", 3: "🌫️ Background"}
CLASS_COLORS = {0: "#AED6F1", 1: "#A9DFBF", 2: "#FAD7A0", 3: "#D2B4DE"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    model_type   = st.radio("Model", ["🎲 Vanilla DCGAN", "🎯 Conditional DCGAN"], index=0)
    config_file  = "configs/config.yaml" if "Vanilla" in model_type else "configs/conditional_config.yaml"
    ckpt_file    = st.file_uploader("Upload checkpoint (.pt)", type=["pt"])
    st.subheader("Generation")
    n_images     = st.slider("Number of images", 1, 25, 9)
    seed         = st.slider("Random seed", 0, 9999, 42)
    if "Conditional" in model_type:
        cls_choice = st.selectbox("Target class", list(CLASS_NAMES.values()))
        cls_id     = [k for k, v in CLASS_NAMES.items() if v == cls_choice][0]
    generate_btn = st.button("🚀 Generate", use_container_width=True)
    st.divider()
    st.caption(f"Config: `{config_file}`")

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

try:
    cfg = load_config(config_file)
except FileNotFoundError:
    st.error(f"Config not found: `{config_file}`"); st.stop()

latent_dim  = cfg["model"]["latent_dim"]
channels    = cfg["data"]["channels"]
image_size  = tuple(cfg["data"]["image_size"])
device      = torch.device("cpu")

# ── Info columns ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Model Info")
    info = {"Architecture": "Vanilla DCGAN" if "Vanilla" in model_type else "Conditional DCGAN",
            "Latent Dim": latent_dim, "Image Size": f"{image_size[0]}×{image_size[1]}",
            "Channels": channels, "Device": str(device)}
    if "Conditional" in model_type:
        info["Num Classes"] = cfg["data"]["num_classes"]
        info["Embed Dim"]   = cfg["model"]["embed_dim"]
    st.json(info)
with col2:
    if "Conditional" in model_type:
        st.subheader("🏷️ Class Legend")
        for k, v in CLASS_NAMES.items():
            st.markdown(
                f'<div style="background:{list(CLASS_COLORS.values())[k]};'
                f'padding:6px 12px;border-radius:6px;margin:3px 0;">'
                f'<b>Class {k}</b> — {v}</div>', unsafe_allow_html=True)
    else:
        st.subheader("🧠 Generator Architecture")
        st.code("z (100,) → Linear → Reshape → Upsample×2 → Conv2d×2 → Tanh", language="text")

st.divider()

# ── Build generator from checkpoint ──────────────────────────────────────────
@st.cache_resource
def build_vanilla_generator(ckpt_bytes, latent_dim, channels, image_size):
    G    = Generator(latent_dim, channels, image_size)
    ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
    G.load_state_dict(ckpt["generator_state_dict"]); G.eval()
    return G

@st.cache_resource
def build_conditional_generator(ckpt_bytes, latent_dim, num_classes, embed_dim, channels, image_size):
    G    = ConditionalGenerator(latent_dim, num_classes, embed_dim, channels, image_size)
    ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu")
    G.load_state_dict(ckpt["generator_state_dict"]); G.eval()
    return G

if ckpt_file is None:
    st.info("👈 Upload a trained `.pt` checkpoint from the sidebar to start generating.")
else:
    ckpt_bytes = ckpt_file.read()
    if "Vanilla" in model_type:
        generator = build_vanilla_generator(ckpt_bytes, latent_dim, channels, image_size)
    else:
        num_classes = cfg["data"]["num_classes"]
        embed_dim   = cfg["model"]["embed_dim"]
        generator   = build_conditional_generator(ckpt_bytes, latent_dim, num_classes,
                                                   embed_dim, channels, image_size)
    st.success(f"✅ Checkpoint loaded — `{ckpt_file.name}`")

    if generate_btn:
        with st.spinner("Generating..."):
            torch.manual_seed(seed)
            with torch.inference_mode():
                z = torch.randn(n_images, latent_dim)
                if "Conditional" in model_type:
                    labels = torch.full((n_images,), cls_id, dtype=torch.long)
                    imgs   = generator(z, labels).squeeze(1).numpy()
                else:
                    imgs   = generator(z).squeeze(1).numpy()
            imgs = np.clip((imgs + 1.0) / 2.0, 0, 1)

        side = int(np.ceil(n_images ** 0.5))
        fig, axes = plt.subplots(side, side, figsize=(side*2, side*2))
        for i, ax in enumerate(axes.flat):
            if i < n_images: ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        title = (f"Generated: {CLASS_NAMES[cls_id]}  (seed={seed})"
                 if "Conditional" in model_type
                 else f"Generated Radar Images  (seed={seed})")
        plt.suptitle(title, fontsize=11); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
