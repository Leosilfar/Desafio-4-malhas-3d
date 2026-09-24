from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One deterministic representative per occupied voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return np.sort(indices)


def estimate_spacing(points: np.ndarray, seed: int = 7, count: int = 6000) -> float:
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(count, len(points)), replace=False)]
    distances, _ = cKDTree(points).query(sample, k=2, workers=-1)
    return float(np.median(distances[:, 1]))


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    frame = vt.T
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return center, frame
