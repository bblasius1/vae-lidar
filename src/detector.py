"""
Simple BEV CNN Detector for KITTI 3D Object Detection
------------------------------------------------------
Trains a lightweight convolutional detector on BEV maps.
Supports 3 training conditions:
  1. Real data only (baseline)
  2. Real + VAE augmented scenes
  3. Real + classical augmentation (flip/rotate)

Usage:
  python src/detector.py --condition real
  python src/detector.py --condition vae
  python src/detector.py --condition classical
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from collections import defaultdict

sys.path.append('src')
from model import VAE


# ─────────────────────────────────────────────
# 1. LABEL CONVERTER
#    Reads KITTI .txt annotations and converts
#    object positions into BEV heatmaps
# ─────────────────────────────────────────────

CLASS_MAP = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

def parse_kitti_label(label_path):
    """Parse a KITTI label file → list of (class_id, x, y) in BEV."""
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_name = parts[0]
            if cls_name not in CLASS_MAP:
                continue
            # KITTI format: x3d=parts[11], y3d=parts[12], z3d=parts[13]
            x3d = float(parts[11])
            z3d = float(parts[13])  # z in camera = forward = x in BEV
            objects.append((CLASS_MAP[cls_name], x3d, z3d))
    return objects


def make_heatmap(objects, x_range=(-40, 40), y_range=(-40, 40),
                 resolution=32, num_classes=3):
    """
    Convert object list to BEV heatmap.
    Returns (num_classes, resolution, resolution) float tensor.
    Gaussian blob placed at each object location.
    """
    heatmap = np.zeros((num_classes, resolution, resolution), dtype=np.float32)
    x_min, x_max = x_range
    y_min, y_max = y_range

    for cls_id, x, y in objects:
        # Map world coords to grid indices
        px = int((x - x_min) / (x_max - x_min) * resolution)
        py = int((y - y_min) / (y_max - y_min) * resolution)
        if 0 <= px < resolution and 0 <= py < resolution:
            # Place a small Gaussian blob (radius 1 cell)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < resolution and 0 <= ny < resolution:
                        dist = dx**2 + dy**2
                        heatmap[cls_id, ny, nx] = max(
                            heatmap[cls_id, ny, nx],
                            np.exp(-dist / 0.5)
                        )
    return heatmap


# ─────────────────────────────────────────────
# 2. DATASETS
# ─────────────────────────────────────────────

def point_cloud_to_bev(scan, x_range=(-40, 40), y_range=(-40, 40),
                        z_range=(-3, 1), resolution=512):
    """Convert raw point cloud to 3-channel BEV map."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    mask = (
        (scan[:, 0] >= x_min) & (scan[:, 0] < x_max) &
        (scan[:, 1] >= y_min) & (scan[:, 1] < y_max) &
        (scan[:, 2] >= z_range[0]) & (scan[:, 2] < z_range[1])
    )
    scan = scan[mask]
    if len(scan) == 0:
        return np.zeros((3, resolution, resolution), dtype=np.float32)

    px = np.clip(
        ((scan[:, 0] - x_min) / (x_max - x_min) * resolution).astype(int),
        0, resolution - 1
    )
    py = np.clip(
        ((scan[:, 1] - y_min) / (y_max - y_min) * resolution).astype(int),
        0, resolution - 1
    )
    height_map    = np.zeros((resolution, resolution), dtype=np.float32)
    density_map   = np.zeros((resolution, resolution), dtype=np.float32)
    intensity_map = np.zeros((resolution, resolution), dtype=np.float32)

    np.maximum.at(height_map,    (py, px), scan[:, 2])
    np.add.at(density_map,       (py, px), 1)
    np.maximum.at(intensity_map, (py, px), scan[:, 3])

    height_map    = (height_map - z_range[0]) / (z_range[1] - z_range[0])
    density_map   = np.log1p(density_map) / np.log1p(density_map.max() + 1e-6)
    intensity_map = intensity_map / (intensity_map.max() + 1e-6)

    return np.stack([height_map, density_map, intensity_map], axis=0)


class KITTIDetectionDataset(Dataset):
    """Real KITTI scenes with BEV maps + heatmap labels."""
    def __init__(self, velodyne_dir, label_dir, resolution=512,
                 augment=False):
        self.velodyne_dir = Path(velodyne_dir)
        self.label_dir    = Path(label_dir)
        self.resolution   = resolution
        self.augment      = augment  # classical augmentation flag

        # Only use files that have both scan + label
        self.scan_files = sorted([
            f for f in self.velodyne_dir.glob('*.bin')
            if (self.label_dir / f.stem).with_suffix('.txt').exists()
        ])
        print(f"Found {len(self.scan_files)} labelled scenes")

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        scan_path  = self.scan_files[idx]
        label_path = (self.label_dir / scan_path.stem).with_suffix('.txt')

        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)

        # Classical augmentation: random horizontal flip
        if self.augment and np.random.rand() > 0.5:
            scan[:, 1] = -scan[:, 1]  # flip y axis

        bev     = point_cloud_to_bev(scan, resolution=self.resolution)
        objects = parse_kitti_label(label_path)
        heatmap = make_heatmap(objects)

        return torch.tensor(bev, dtype=torch.float32), \
               torch.tensor(heatmap, dtype=torch.float32)


class VAEAugmentedDataset(Dataset):
    """
    Generates synthetic BEV scenes on-the-fly from the VAE.
    Used as additional training data alongside real scenes.
    Labels are estimated from VAE-decoded scenes using
    the density channel peaks as pseudo object locations.
    Note: VAE scenes don't have ground truth labels so we
    use them as background augmentation (label = all zeros).
    This tests whether realistic background diversity helps.
    """
    def __init__(self, vae_model, device, n_samples, resolution=512):
        self.vae      = vae_model
        self.device   = device
        self.n        = n_samples
        self.resolution = resolution

        print(f"Pre-generating {n_samples} VAE scenes...")
        self.scenes = []
        self.vae.eval()
        with torch.no_grad():
            for i in range(0, n_samples, 32):
                batch_n = min(32, n_samples - i)
                z = torch.randn(batch_n, 256).to(device)
                generated = self.vae.decoder(z).cpu()
                self.scenes.append(generated)
        self.scenes = torch.cat(self.scenes, dim=0)
        print(f"Generated {len(self.scenes)} VAE scenes")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        bev     = self.scenes[idx]
        # Empty label heatmap for VAE scenes
        heatmap = torch.zeros(3, 32, 32, dtype=torch.float32)
        return bev, heatmap


# ─────────────────────────────────────────────
# 3. DETECTOR MODEL
# ─────────────────────────────────────────────

class SimpleBEVDetector(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
            # 512 → 256
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # 256 → 128
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # 128 → 64
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            # 64 → 32
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Detection head — one score map per class at 32×32 resolution
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, num_classes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
# 4. mAP EVALUATION
# ─────────────────────────────────────────────

def compute_map(pred_maps, gt_maps, threshold=0.5, num_classes=3):
    """
    Compute mean Average Precision across classes.
    pred_maps: (N, C, H, W) predictions in [0,1]
    gt_maps:   (N, C, H, W) ground truth heatmaps
    """
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    aps = []

    for cls in range(num_classes):
        pred_flat = pred_maps[:, cls].reshape(-1).numpy()
        gt_flat   = gt_maps[:, cls].reshape(-1).numpy()

        # Binarize ground truth
        gt_binary = (gt_flat > 0.5).astype(int)

        if gt_binary.sum() == 0:
            continue  # skip class if no positives in val set

        # Sort by prediction confidence descending
        sorted_idx  = np.argsort(-pred_flat)
        gt_sorted   = gt_binary[sorted_idx]
        pred_sorted = pred_flat[sorted_idx]

        # Compute precision-recall curve
        tp = np.cumsum(gt_sorted)
        fp = np.cumsum(1 - gt_sorted)
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (gt_binary.sum() + 1e-6)

        # AP = area under PR curve (trapezoid)
        ap = np.trapz(precision, recall)
        aps.append(ap)
        print(f"  AP {class_names[cls]}: {ap:.4f}")

    map_score = np.mean(aps) if aps else 0.0
    return map_score


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────

def train_detector(condition='real', epochs=30, batch_size=4, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Condition: {condition.upper()}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")

    os.makedirs('outputs/detector', exist_ok=True)

    # ── Build dataset ──
    velodyne_dir = 'data/kitti/data_object_velodyne/training/velodyne'
    label_dir    = 'data/kitti/data_object_label_2/training/label_2'

    real_dataset = KITTIDetectionDataset(
        velodyne_dir, label_dir,
        augment=(condition == 'classical')
    )

    # Train/val split (90/10)
    val_size   = int(0.1 * len(real_dataset))
    train_size = len(real_dataset) - val_size
    train_real, val_dataset = torch.utils.data.random_split(
        real_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    if condition == 'vae':
        # Load VAE and generate synthetic scenes
        vae = VAE(latent_dim=256).to(device)
        vae.load_state_dict(torch.load(
            'outputs/bce_loss/vae_bce_epoch100.pt',
            map_location=device,
            weights_only=True
        ))
        vae_dataset  = VAEAugmentedDataset(vae, device, n_samples=train_size)
        train_dataset = ConcatDataset([train_real, vae_dataset])
        print(f"Training on {len(train_real)} real + {len(vae_dataset)} VAE scenes")
    else:
        train_dataset = train_real
        print(f"Training on {len(train_dataset)} scenes")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── Model + optimizer ──
    model     = SimpleBEVDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_map  = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0
        for bev, heatmap in train_loader:
            bev, heatmap = bev.to(device), heatmap.to(device)
            pred = model(bev)
            loss = F.binary_cross_entropy(pred, heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validate every 5 epochs ──
        if epoch % 5 == 0:
            model.eval()
            all_preds, all_gts = [], []
            with torch.no_grad():
                for bev, heatmap in val_loader:
                    pred = model(bev.to(device)).cpu()
                    all_preds.append(pred)
                    all_gts.append(heatmap)

            all_preds = torch.cat(all_preds, dim=0)
            all_gts   = torch.cat(all_gts,   dim=0)
            map_score = compute_map(all_preds, all_gts)

            print(f"Epoch {epoch:03d} | "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | "
                  f"mAP: {map_score:.4f}")

            if map_score > best_map:
                best_map = map_score
                torch.save(model.state_dict(),
                           f'outputs/detector/best_{condition}.pt')
                print(f"  ✓ Saved best model (mAP={best_map:.4f})")

    print(f"\nFinal best mAP [{condition}]: {best_map:.4f}")
    return best_map


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='real',
                        choices=['real', 'vae', 'classical'],
                        help='Training condition')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()

    map_score = train_detector(
        condition=args.condition,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )