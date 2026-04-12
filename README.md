<div align="center">

# 📡 Radar View Generator
## Comparative Study: DCGAN vs Conditional DCGAN

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Enabled-orange?logo=tensorflow)](https://www.tensorflow.org/tensorboard)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **A complete, production-grade deep learning study on synthetic radar image generation.**
> Two GAN architectures trained, evaluated, and compared on a real radar dataset of
> **19,200 training samples** across 4 target classes.

[📖 Study Overview](#-study-overview) •
[🧠 Architectures](#-architectures) •
[📊 Comparative Results](#-comparative-results) •
[🖼️ Generated Images](#️-generated-images) •
[🚀 Quick Start](#-quick-start) •
[👤 Author](#-author)

</div>

---

## 📖 Study Overview

This repository presents a **side-by-side experimental comparison** between two
GAN variants for synthetic radar image generation:

| | Model | Question Answered |
|---|---|---|
| **Experiment 1** | DCGAN | *Can a GAN learn the general distribution of radar images?* |
| **Experiment 2** | Conditional DCGAN | *Can a GAN generate radar images of a specific target class on demand?* |

Both models were trained from scratch on the **same dataset**, with the **same
hyperparameters**, for **30 epochs** — ensuring a fair, controlled comparison.

### 🎯 Why Radar Image Synthesis?

Radar image synthesis has direct real-world impact in:
- 🛡️ **Defence & Surveillance** — data augmentation for radar target recognition
- 🚗 **Autonomous Vehicles** — simulating sensor inputs for ADAS safety testing
- 🌦️ **Meteorology** — generating synthetic weather radar training data
- 🔬 **Research** — creating labelled datasets where real radar data is scarce or classified

---

## 📂 Repository Structure

```
radar-dcgan/
│
├── 📓 DCGAN_Radar.ipynb                  ← Experiment 1: Vanilla DCGAN
├── 📓 Conditional_DCGAN_Radar.ipynb      ← Experiment 2: Conditional DCGAN
│
├── 📂 app/
│   └── main.py                           ← Streamlit interactive demo
│
├── 📂 src/                               ← Production source package
│   ├── models/
│   │   ├── generator.py                  ← Vanilla Generator
│   │   └── discriminator.py              ← Vanilla Discriminator
│   ├── data/dataset.py                   ← PyTorch Dataset class
│   ├── training/trainer.py               ← Training loop + TensorBoard
│   └── utils/
│       ├── checkpoint.py                 ← Fault-tolerant save/load
│       └── visualize.py                  ← Loss curves + image grids
│
├── 📂 configs/
│   └── config.yaml                       ← All hyperparameters (single source of truth)
│
├── 📂 outputs/
│   ├── dcgan/                            ← Vanilla DCGAN results
│   │   ├── checkpoints/
│   │   ├── generated_images/
│   │   └── logs/                         ← TensorBoard logs
│   └── cdcgan/                           ← Conditional DCGAN results
│       ├── checkpoints/
│       ├── generated_images/
│       └── logs/
│
├── train.py                              ← CLI: train vanilla DCGAN
├── generate.py                           ← CLI: generate from checkpoint
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Format | `.npz` (NumPy compressed archive) |
| Train samples | **19,200** |
| Test samples | **4,800** |
| Image size | 64 × 64 pixels (cropped from 100 × 160) |
| Channels | 1 (grayscale) |
| Normalisation | `[-1, 1]` (standard for DCGAN with Tanh output) |

### 🏷️ Target Classes

| Label | Class | Description |
|:---:|---|---|
| 0 | 🚶 **Pedestrian** | Small, diffuse horizontal radar cross-section |
| 1 | 🛴 **Electric Scooter** | Wider horizontal pattern with vertical micro-structure |
| 2 | 🚗 **Car** | Tall, narrow, high-intensity vertical reflection |
| 3 | 🌫️ **Background** | Scattered low-intensity clutter |

---

## 🧠 Architectures

### Experiment 1 — DCGAN (Vanilla)

```
Random Noise z ~ N(0,1)                     Real/Fake Image
        │                                          │
        ▼                                          ▼
┌──────────────────────┐              ┌─────────────────────────┐
│     GENERATOR        │              │     DISCRIMINATOR       │
│                      │              │                         │
│  Linear(100→32768)   │              │  Conv2d(1→32, s=2)      │
│  Reshape(128,16,16)  │   Fake Img   │  Conv2d(32→64, s=2)     │
│  Upsample+Conv2d×2   │ ──────────→ │  Conv2d(64→128, s=2)    │
│  Conv2d + Tanh       │              │  Conv2d(128→256, s=1)   │
│                      │              │  Linear → Sigmoid       │
│  Params: 3,531,905   │              │  Params: 405,121        │
└──────────────────────┘              └─────────────────────────┘

Input  : z (B, 100)              Input  : Image (B, 1, 64, 64)
Output : Image (B, 1, 64, 64)    Output : P(real) (B, 1)
```

### Experiment 2 — Conditional DCGAN

```
Noise z + Class Label                       Image + Class Label
        │                                          │
        ▼                                          ▼
┌──────────────────────────┐        ┌──────────────────────────────┐
│   CONDITIONAL GENERATOR  │        │   CONDITIONAL DISCRIMINATOR  │
│                          │        │                              │
│  Embedding(4 → 50)       │        │  Embedding(4 → 64×64)        │
│  concat(z, emb) → (150,) │        │  Reshape → (1, 64, 64)       │
│  Linear(150→32768)       │  Fake  │  concat(img, label_map)      │
│  Reshape(128,16,16)      │  Img   │  → (2, 64, 64)               │
│  Upsample+Conv2d×2       │ ─────→ │  Conv2d blocks × 4           │
│  Conv2d + Tanh           │        │  Linear → Sigmoid            │
│                          │        │                              │
│  Params: 3,587,505       │        │  Params: 409,217             │
└──────────────────────────┘        └──────────────────────────────┘

Input  : z (B,100) + label (B,)     Input  : Image (B,1,64,64) + label (B,)
Output : Class-specific image       Output : P(real AND correct class)
```

### 🔑 Key Architectural Difference

| Component | DCGAN | Conditional DCGAN |
|---|---|---|
| Generator input | `z` (100,) | `concat(z, class_emb)` (150,) |
| Discriminator input | image `(1, 64, 64)` | `concat(image, label_map)` **(2, 64, 64)** |
| Class conditioning | ❌ None | ✅ `nn.Embedding(num_classes, embed_dim)` |
| Output class control | ❌ Random | ✅ **On demand** |
| New hyperparameter | — | `embed_dim = 50` |

---

## 📊 Comparative Results

### Training Configuration (Identical for Both)

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Learning rate | 0.0002 |
| Adam β₁ | 0.5 |
| Adam β₂ | 0.999 |
| Batch size | 100 |
| Loss function | Binary Cross-Entropy (BCELoss) |
| Training order | Discriminator first, Generator second |

---

### 📉 Loss Comparison — Epoch by Epoch

| Epoch | DCGAN D_Loss | DCGAN G_Loss | cDCGAN D_Loss | cDCGAN G_Loss |
|:---:|:---:|:---:|:---:|:---:|
| 1  | 0.539 | 1.337 | 0.702 | 0.843 |
| 5  | 0.615 | 1.073 | 0.709 | 0.791 |
| 10 | 0.538 | 1.255 | 0.674 | 0.869 |
| 15 | 0.433 | 1.382 | 0.662 | 0.997 |
| 20 | 0.338 | 1.327 | 0.589 | 1.210 |
| 25 | 0.280 | 1.211 | 0.591 | 1.337 |
| **30** | **0.207** | **0.881** | **0.448** | **1.964** |

### 🔍 Key Findings from Loss Analysis

**Finding 1 — cDCGAN achieves near-ideal Discriminator balance**
> The theoretically perfect D_Loss for a GAN at equilibrium is **0.5** (the point where
> D cannot distinguish real from fake — random chance). cDCGAN's final D_Loss of **0.448**
> is remarkably close to this ideal, whereas vanilla DCGAN's **0.207** indicates
> the Discriminator became overpowered.

**Finding 2 — Class conditioning acts as a natural regulariser**
> By forcing the Discriminator to verify both image realism AND class correctness,
> conditioning prevents D from "memorising" and overfit to the real distribution.
> This is why cDCGAN maintains better D/G balance throughout training.

**Finding 3 — cDCGAN G_Loss rising after Epoch 20**
> G_Loss climbing 1.0 → 1.96 indicates the Generator faces a harder task
> in cDCGAN — it must satisfy both a realism constraint AND a class-correctness
> constraint simultaneously. This is expected and can be stabilised with
> label smoothing (see Improvements section).

---

## 🖼️ Generated Images

### Experiment 1 — Vanilla DCGAN Results

The grid below shows **25 samples** generated by the vanilla DCGAN after 30 epochs of
training. The model has learned the general distribution of radar signatures across all
classes, producing structurally diverse patterns including horizontal streaks, vertical
columns, and diffuse blob signatures — without any class-level control.

![DCGAN Generated Images](DC-GAN.jpg)

> **Observation:** By Epoch 30, the vanilla DCGAN generates sharp, realistic radar
> patterns with clear structural diversity. The variety of shapes (horizontal blobs,
> vertical columns, scattered clutter) confirms the model has captured the multi-modal
> nature of the radar dataset. However, there is no way to request a specific class —
> the output class is determined entirely by the sampled noise vector `z`.

---

### Experiment 2 — Conditional DCGAN Results (Class-Specific Generation)

The figure below demonstrates the **key result** of this study. Each row was generated
by conditioning the model on a different class label. Using the **same architecture
and training duration**, the Conditional DCGAN learns to produce visually and
physically distinct radar signatures per class — entirely on demand.

![Conditional DCGAN Class-Specific Results](conditional-gan.jpg)

**Per-class visual analysis:**

| Class | Label | Visual Pattern Observed | Physical Interpretation |
|:---:|---|---|---|
| 0 | 🚶 **Pedestrian** | Wide, diffuse **horizontal blobs** | Small RCS, distributed body reflection |
| 1 | 🛴 **Electric Scooter** | **Wider horizontal** spread + vertical micro-structure | Larger combined rider + vehicle RCS |
| 2 | 🚗 **Car** | Tall, sharp **vertical columns** | High RCS, strong specular metallic reflection |

> **RCS** = Radar Cross Section — the measure of how detectable an object is by radar.
> The clear visual separation between classes confirms the Conditional DCGAN has
> genuinely learned the underlying radar physics of each target type.

---

## 🧪 Comparative Study — Summary Table

| Evaluation Criterion | DCGAN | Conditional DCGAN | Winner |
|---|---|---|:---:|
| Final D_Loss (closer to 0.5 = better) | 0.207 | **0.448** | cDCGAN ✅ |
| Training stability | Moderate | Good (1 spike, self-healed) | cDCGAN ✅ |
| D / G balance at Epoch 30 | Poor | Near-ideal | cDCGAN ✅ |
| Class-specific generation | ❌ Not possible | ✅ On demand | cDCGAN ✅ |
| Image sharpness (Epoch 30) | High | High | Tie 🤝 |
| Class 0 vs Class 3 separation | N/A | Slight overlap | — |
| Car vs Pedestrian distinction | N/A | ✅ Clear | cDCGAN ✅ |
| Training time per epoch | ~20s | ~22s | DCGAN ✅ |
| Model parameters (G) | 3,531,905 | 3,587,505 | DCGAN ✅ |
| Practical utility | Limited | **High** | cDCGAN ✅ |

---

## 🚀 Quick Start

### 1 — Clone & Install
```bash
git clone https://github.com/krishna-ambekar/radar-dcgan.git
cd radar-dcgan
pip install -r requirements.txt
```

### 2 — Prepare Dataset
```bash
mkdir data
# Place: data/Dataset_19200_train_4800_test.npz
# Keys : X_train, y_train, X_test, y_test
```

### 3 — Run Experiment 1: Vanilla DCGAN
```bash
python train.py
```

### 4 — Run Experiment 2: Conditional DCGAN
Open `Conditional_DCGAN_Radar.ipynb` and run all cells.

### 5 — Monitor with TensorBoard
```bash
tensorboard --logdir outputs/logs
# Open http://localhost:6006
# Scalars: D_Loss, G_Loss | Images: generated grids | Histograms: weights
```

### 6 — Generate Class-Specific Images (cDCGAN)
```python
# Inside notebook — generate 25 Car images
generate_class_grid(target_class=2, n=25, seed=42)   # Class 2 = Car
```

### 7 — Launch Streamlit Demo
```bash
streamlit run app/main.py
# Upload any .pt checkpoint → generate images live in browser
```

---

## 🐳 Docker
```bash
docker build -t radar-dcgan .
docker run -p 8501:8501 radar-dcgan
# Open http://localhost:8501
```

---

## 🔑 Engineering Decisions

| Decision | Reason |
|---|---|
| **Tanh output** (not ReLU) | Correct for data normalised to [-1,1]; ReLU clips negatives |
| **Discriminator trains first** | Provides meaningful gradients to Generator (DCGAN paper) |
| **Adam lr=0.0002, β₁=0.5** | Directly from Radford et al. (2015) DCGAN paper |
| **Label map as extra channel** | Spatial conditioning — D checks class at every pixel location |
| **Embedding dim = 50** | Balances expressiveness vs. overfitting for 4-class problem |
| **config.yaml** | Zero hardcoded values — full reproducibility |
| **Checkpoint every 5 epochs** | Resume training from any saved state |
| **Python logging + TensorBoard** | Structured, searchable logs + real-time visual monitoring |
| **Dropout in Discriminator** | Prevents memorisation, keeps G competitive |

---

## 📈 Recommendations for Future Work

- [ ] **Label smoothing** — use `valid=0.9`, `fake=0.1` to prevent D overconfidence
- [ ] **Train 50–100 epochs** — G_Loss still descending at Epoch 30
- [ ] **Separate LR for D and G** — `lr_D=0.0001`, `lr_G=0.0002`
- [ ] **WGAN-GP loss** — gradient penalty for more stable training
- [ ] **FID score** (Fréchet Inception Distance) — quantitative image quality metric
- [ ] **Class-specific FID** — evaluate quality per radar target class independently
- [ ] **ONNX export** — deploy Generator on edge radar processing hardware

---

## 👤 Author

<div align="center">

**Krishna Sanjay Ambekar**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5?logo=linkedin)](https://www.linkedin.com/in/krishna-ambekar-b4a2641b2)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/KrishnaSA05)

*"This project demonstrates not just the ability to implement deep learning models,
but to conduct structured experiments, analyse results scientifically,
and engineer production-quality systems."*

</div>

---

<div align="center">
⭐ If this study was useful, please consider starring the repository!
</div>
