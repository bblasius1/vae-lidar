# VAE-Based Synthetic LiDAR Scene Generation

> CPSC 440/550 Machine Learning — Course Project (Jan–Apr 2026)  
> University of British Columbia

Can a Variational Autoencoder trained on real LiDAR scans produce synthetic point clouds that are **geometrically plausible** and **useful as augmentation data** for a downstream 3D object detector?

This project investigates that question using the KITTI 3D Object Detection benchmark, with the goal of addressing data scarcity in autonomous vehicle perception systems.

---

## Overview

Autonomous vehicle (AV) perception systems rely on 3D LiDAR point clouds to detect and localize objects such as cars, pedestrians, and cyclists. Collecting and annotating sufficient real-world data is extremely costly, and existing datasets are heavily skewed toward common scenarios.

We train a **VAE on Bird's-Eye View (BEV) projections** of LiDAR scenes and evaluate whether synthetic scenes can augment real training data to improve a downstream 3D object detector's mean Average Precision (mAP).

---

## Method

### Preprocessing: Point Cloud → BEV

Raw KITTI point clouds (~120,000 points per scan) are projected onto a 512×512 Bird's-Eye View height map encoding three channels:

- **Height** — maximum z-value per cell
- **Density** — log-normalized point count per cell  
- **Intensity** — maximum return intensity per cell

### Model: Convolutional VAE

The VAE consists of:
- **Encoder** — convolutional layers compressing BEV maps into a latent distribution `q(z|x)`
- **Latent space** — Gaussian prior `p(z) = N(0, I)`
- **Decoder** — transposed-convolutional layers reconstructing BEV maps from `z`

Training minimizes the **ELBO** (Evidence Lower Bound):

```
L = E[log p(x|z)] - β · KL(q(z|x) || p(z))
```

The β coefficient is annealed during training to prevent posterior collapse.

### Evaluation

| Axis | Metric |
|---|---|
| Generative quality | Fréchet Point Cloud Distance (FPD) |
| Downstream utility | mAP on KITTI val set (3D, IoU ≥ 0.7) |

We compare three detector training conditions:
1. Real data only (baseline)
2. Real + VAE-synthesized scenes
3. Real + classical augmentation (random flip / rotation)

---

## Repository Structure

```
vae-lidar/
├── data/
│   └── kitti/              ← raw KITTI downloads (not tracked)
├── src/
│   ├── dataset.py          ← KITTIBEVDataset + DataLoader
│   ├── preprocess.py       ← point cloud → BEV conversion
│   ├── model.py            ← VAE encoder / decoder
│   └── train.py            ← training loop + ELBO loss
├── notebooks/
│   └── 01_explore.ipynb    ← data exploration
├── outputs/
│   └── samples/            ← generated BEV images
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install Miniconda

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't already have it.

### 2. Create environment

```bash
conda create -n vae-lidar python=3.10
conda activate vae-lidar
pip install -r requirements.txt
```

### 3. Download KITTI

Register at [https://www.cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and download:

| File | Size | Required |
|---|---|---|
| Velodyne point clouds | ~29 GB | ✅ Yes |
| Training labels | ~5 MB | ✅ Yes |
| Left color images | ~12 GB | Optional |

Place the downloaded files under `data/kitti/`.

### 4. Run preprocessing

```bash
python src/preprocess.py --input data/kitti/velodyne --output data/kitti/bev
```

### 5. Train the VAE

```bash
python src/train.py --epochs 100 --latent-dim 256 --beta 1.0
```

---

## Results

*To be updated as experiments complete.*

| Condition | Car mAP | Pedestrian mAP | Cyclist mAP |
|---|---|---|---|
| Real only (baseline) | — | — | — |
| Real + VAE augmentation | — | — | — |
| Real + classical augmentation | — | — | — |

---

## Course Connection

This project directly applies the following CPSC 440/550 topics:

- **Variational inference & VAEs** (Lecture 14) — core model and ELBO objective
- **Transposed convolutions & representation learning** (Lectures 15–16) — encoder/decoder architecture
- **Bayesian learning & MAP estimation** (Lectures 2, 6) — Gaussian prior/posterior
- **Gaussian mixture models** (Lectures 10–11) — latent space analysis baseline

---

## Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1 | Mar 29 – Apr 4 | Setup, KITTI download, BEV preprocessing |
| 2 | Apr 4 – Apr 11 | DataLoader, train/val/test split |
| 3 | Apr 11 – Apr 18 | VAE implementation + first training run |
| 4 | Apr 18 – Apr 23 | Generation, FPD evaluation, qualitative inspection |
| 5 | Apr 23 – Apr 25 | Downstream detector experiment + writeup |

---
