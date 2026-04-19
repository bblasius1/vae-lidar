from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch

from .preprocess import point_cloud_to_bev

def load_kitti_bin(file_path):
    scan = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    bev = point_cloud_to_bev(scan)
    return bev


class KITTIBEVDataset(Dataset):
    def __init__(self, scan_dir, resolution=512):
        self.files = sorted(Path(scan_dir).glob("*.bin"))
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        scan = np.fromfile(self.files[idx], dtype=np.float32).reshape(-1, 4)
        bev = point_cloud_to_bev(scan, resolution=self.resolution)
        return torch.from_numpy(bev).float()