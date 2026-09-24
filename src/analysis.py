from __future__ import annotations

import time
import numpy as np

from .registration import coarse_pca, point_to_plane_icp, stable_region, apply
from .metrics import directional_metrics


def register(reference, moving, normals, spacing: float):
    started = time.perf_counter()
    coarse, coarse_info = coarse_pca(reference, moving)
    stable, stable_info = stable_region(reference, moving, coarse)
    voxels = [spacing * 12, spacing * 6, spacing * 3]
    final, stages = point_to_plane_icp(reference, normals, moving, coarse, voxels, stable)
    return final, {"coarse": coarse_info, "stable_region": stable_info, "icp_stages": stages, "registration_seconds": time.perf_counter() - started}, stable


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def uncertainty(reference, moving, normals, spacing: float, repeats: int = 6) -> dict:
    rng = np.random.default_rng(2030)
    transforms, rms_values, seconds = [], [], []
    for _ in range(repeats):
        ref = reference[rng.choice(len(reference), int(len(reference) * 0.75), replace=False)]
        mov = moving[rng.choice(len(moving), int(len(moving) * 0.75), replace=False)]
        # Normals follow reference bootstrap indices through an independent paired draw.
        ri = rng.choice(len(reference), int(len(reference) * 0.75), replace=False)
        ref, norm = reference[ri], normals[ri]
        transform, _, _ = register(ref, mov, norm, spacing)
        distances, metric = directional_metrics(apply(moving, transform), reference, spacing * 8, "moving -> reference")
        transforms.append(transform); rms_values.append(metric["rms_native"]); seconds.append(0.0)
    translations = np.array([item[:3, 3] for item in transforms])
    angles = np.array([rotation_angle_degrees(item[:3, :3] @ transforms[0][:3, :3].T) for item in transforms])
    return {"method": f"{repeats} bootstrap repeats with 75% point subsets; fixed seed 2030", "rms_mean_native": float(np.mean(rms_values)), "rms_std_native": float(np.std(rms_values, ddof=1)), "translation_std_native": np.std(translations, axis=0, ddof=1).tolist(), "translation_std_norm_native": float(np.linalg.norm(np.std(translations, axis=0, ddof=1))), "rotation_std_deg": float(np.std(angles, ddof=1))}


def robustness(reference, moving, normals, spacing: float, final: np.ndarray) -> dict:
    rng = np.random.default_rng(40)
    outcomes = []
    tests = [("noise_0.5_spacing", moving + rng.normal(0, spacing * 0.5, moving.shape)), ("missing_highest_25pct_z", moving[moving[:, 2] <= np.quantile(moving[:, 2], .75)])]
    for name, altered in tests:
        transform, details, _ = register(reference, altered, normals, spacing)
        _, metric = directional_metrics(apply(altered, transform), reference, spacing * 8, "moving -> reference")
        outcomes.append({"test": name, "success": metric["coverage_within_threshold_percent"] > 50, "rms_native": metric["rms_native"], "coverage_within_threshold_percent": metric["coverage_within_threshold_percent"], "registration_seconds": details["registration_seconds"]})
    perturb = np.eye(4); angle = np.radians(8); perturb[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]; perturb[:3, 3] = [spacing * 15, -spacing * 10, spacing * 8]
    transform, details, _ = register(reference, apply(moving, perturb), normals, spacing)
    _, metric = directional_metrics(apply(apply(moving, perturb), transform), reference, spacing * 8, "moving -> reference")
    outcomes.append({"test": "initialization_8deg_and_translation", "success": metric["coverage_within_threshold_percent"] > 50, "rms_native": metric["rms_native"], "coverage_within_threshold_percent": metric["coverage_within_threshold_percent"], "registration_seconds": details["registration_seconds"]})
    return {"method": "controlled synthetic perturbations; originals untouched; fixed seed 40", "tests": outcomes}
