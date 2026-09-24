from __future__ import annotations

from itertools import permutations, product
import numpy as np
from scipy.spatial import cKDTree

from .preprocess import pca_frame, voxel_downsample


def identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left @ right


def apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def rigid_from_correspondences(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = identity()
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def coarse_pca(reference: np.ndarray, moving: np.ndarray, sample_size: int = 5000) -> tuple[np.ndarray, dict]:
    rc, rf = pca_frame(reference)
    mc, mf = pca_frame(moving)
    rng = np.random.default_rng(19)
    sample = moving[rng.choice(len(moving), min(sample_size, len(moving)), replace=False)]
    tree = cKDTree(reference)
    candidates: list[tuple[float, np.ndarray]] = []
    for order in permutations(range(3)):
        permutation = np.eye(3)[:, order]
        for signs in product((-1.0, 1.0), repeat=3):
            signed = permutation @ np.diag(signs)
            rotation = rf @ signed @ mf.T
            if np.linalg.det(rotation) < 0:
                continue
            transform = identity()
            transform[:3, :3] = rotation
            transform[:3, 3] = rc - rotation @ mc
            distances, _ = tree.query(apply(sample, transform), workers=-1)
            candidates.append((float(np.median(distances)), transform))
    score, best = min(candidates, key=lambda item: item[0])
    return best, {"method": "PCA axes with 24 proper orientation hypotheses", "candidates": len(candidates), "median_nn_score_native": score}


def _small_angle_transform(parameters: np.ndarray) -> np.ndarray:
    wx, wy, wz, tx, ty, tz = parameters
    vector = np.array([wx, wy, wz])
    angle = np.linalg.norm(vector)
    skew = np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])
    if angle < 1e-12:
        rotation = np.eye(3) + skew
    else:
        rotation = np.eye(3) + (np.sin(angle) / angle) * skew + ((1 - np.cos(angle)) / angle**2) * (skew @ skew)
    delta = identity()
    delta[:3, :3] = rotation
    delta[:3, 3] = (tx, ty, tz)
    return delta


def point_to_plane_icp(reference: np.ndarray, reference_normals: np.ndarray, moving: np.ndarray,
                       initial: np.ndarray, voxel_sizes: list[float], stable_mask: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    transform = initial.copy()
    stages: list[dict] = []
    for voxel in voxel_sizes:
        ri = voxel_downsample(reference, voxel)
        mi = voxel_downsample(moving, voxel)
        ref, ref_normals = reference[ri], reference_normals[ri].copy()
        norm = np.linalg.norm(ref_normals, axis=1)
        ref_normals[norm > 0] /= norm[norm > 0, None]
        movable = moving[mi]
        if stable_mask is not None:
            movable = movable[stable_mask[mi]]
        tree = cKDTree(ref)
        max_distance = voxel * 3.5
        iterations, last_rms = 0, None
        for iteration in range(50):
            transformed = apply(movable, transform)
            distances, nearest = tree.query(transformed, workers=-1)
            valid = distances < max_distance
            if valid.sum() < 30:
                break
            source, targets, normals, residual = transformed[valid], ref[nearest[valid]], ref_normals[nearest[valid]], None
            residual = np.sum(normals * (source - targets), axis=1)
            cutoff = np.quantile(np.abs(residual), 0.80)
            keep = np.abs(residual) <= max(cutoff, voxel * 0.20)
            source, normals, residual = source[keep], normals[keep], residual[keep]
            a = np.column_stack((np.cross(source, normals), normals))
            weights = 1.0 / (1.0 + (residual / max(voxel, 1e-8)) ** 2)  # Cauchy robust kernel
            step, *_ = np.linalg.lstsq(a * weights[:, None], -residual * weights, rcond=None)
            transform = compose(_small_angle_transform(step), transform)
            rms = float(np.sqrt(np.mean(residual ** 2)))
            iterations = iteration + 1
            if last_rms is not None and abs(last_rms - rms) < voxel * 1e-4 and np.linalg.norm(step) < voxel * 1e-4:
                break
            last_rms = rms
        stages.append({"voxel_native": float(voxel), "reference_points": int(len(ref)), "moving_points": int(len(movable)), "iterations": iterations, "point_to_plane_rms_native": last_rms})
    return transform, stages


def stable_region(reference: np.ndarray, moving: np.ndarray, coarse: np.ndarray, fraction: float = 0.65) -> tuple[np.ndarray, dict]:
    distances, _ = cKDTree(reference).query(apply(moving, coarse), workers=-1)
    threshold = float(np.quantile(distances, fraction))
    return distances <= threshold, {"method": "automatic residual consensus after PCA coarse alignment", "fraction": fraction, "threshold_native": threshold, "selected_points": int((distances <= threshold).sum())}
