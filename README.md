# VAE-Based Synthetic LiDAR Scene Generation


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

### 1. Create environment

```bash
pip install -r requirements.txt
```

### 2. Download KITTI

Register at [https://www.cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and download:

| File | Size | Required |
|---|---|---|
| Velodyne point clouds | ~29 GB | ✅ Yes |
| Training labels | ~5 MB | ✅ Yes |

Place the downloaded files under `data/kitti/`.

### 3. Run preprocessing

```bash
python src/preprocess.py --input data/kitti/velodyne --output data/kitti/bev
```

### 4. Train the VAE

```bash
python src/train.py --epochs 100 --latent-dim 256 --beta 1.0
```

---

## Results

| Condition | Car mAP | Pedestrian mAP | Cyclist mAP |
|---|---|---|---|
| Real only (baseline) | 0.0873 | 0.1332 | 0.0671 |
| Real + VAE augmentation | 0.0996 | 0.1479 | 0.0732 |
| Real + classical augmentation | 0.0918 | 0.1159 | 0.0520 |

---
