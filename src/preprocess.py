import numpy as np

def point_cloud_to_bev(scan, x_range=(-40, 40), y_range=(-40, 40),
                        z_range=(-3, 1), resolution=512):
    """
    Convert a raw KITTI point cloud to a BEV height map.
    Returns a (3, H, W) tensor: max height, point density, max intensity.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Filter to region of interest
    mask = (
        (scan[:,0] >= x_min) & (scan[:,0] < x_max) &
        (scan[:,1] >= y_min) & (scan[:,1] < y_max) &
        (scan[:,2] >= z_range[0]) & (scan[:,2] < z_range[1])
    )
    scan = scan[mask]

    # Map continuous coords to pixel indices
    px = ((scan[:,0] - x_min) / (x_max - x_min) * resolution).astype(int)
    py = ((scan[:,1] - y_min) / (y_max - y_min) * resolution).astype(int)
    px = np.clip(px, 0, resolution - 1)
    py = np.clip(py, 0, resolution - 1)

    height_map    = np.zeros((resolution, resolution), dtype=np.float32)
    density_map   = np.zeros((resolution, resolution), dtype=np.float32)
    intensity_map = np.zeros((resolution, resolution), dtype=np.float32)

    np.maximum.at(height_map,    (py, px), scan[:,2])
    np.add.at(density_map,       (py, px), 1)
    np.maximum.at(intensity_map, (py, px), scan[:,3])

    # Normalize each channel to [0, 1]
    height_map    = (height_map - z_range[0]) / (z_range[1] - z_range[0])
    density_map   = np.log1p(density_map) / np.log1p(density_map.max() + 1e-6)
    intensity_map = intensity_map / (intensity_map.max() + 1e-6)

    return np.stack([height_map, density_map, intensity_map], axis=0)