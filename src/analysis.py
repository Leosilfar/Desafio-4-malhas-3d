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
    transforms, rms_values = [], []
    fraction = 0.75
    for _ in range(repeats):
        ri = rng.choice(len(reference), int(len(reference) * fraction), replace=False)
        mi = rng.choice(len(moving), int(len(moving) * fraction), replace=False)
        ref, norm = reference[ri], normals[ri]
        mov = moving[mi]
        transform, _, _ = register(ref, mov, norm, spacing)
        distances, metric = directional_metrics(apply(moving, transform), reference, spacing * 8, "moving -> reference")
        transforms.append(transform)
        rms_values.append(metric["rms_native"])
    translations = np.array([item[:3, 3] for item in transforms])
    angles = np.array([rotation_angle_degrees(item[:3, :3] @ transforms[0][:3, :3].T) for item in transforms])
    return {
        "label": "variability of RMS under point-subset perturbations",
        "method": f"{repeats} deterministic repetitions with independent 75% subsets of reference and moving clouds; fixed seed 2030",
        "repetitions": repeats,
        "fraction_of_points_used": fraction,
        "noise_applied_native": 0.0,
        "rms_each_native": [float(value) for value in rms_values],
        "rms_mean_native": float(np.mean(rms_values)),
        "rms_std_native": float(np.std(rms_values, ddof=1)),
        "translation_std_native": np.std(translations, axis=0, ddof=1).tolist(),
        "translation_std_norm_native": float(np.linalg.norm(np.std(translations, axis=0, ddof=1))),
        "rotation_std_deg": float(np.std(angles, ddof=1)),
    }


def robustness(reference, moving, normals, spacing: float, final: np.ndarray) -> dict:
    rng = np.random.default_rng(40)
    outcomes = []
    tests = [
        ("noise_0.5_spacing", moving + rng.normal(0, spacing * 0.5, moving.shape), "Gaussian noise, sigma=0.5 median NN spacing"),
        ("missing_highest_25pct_z", moving[moving[:, 2] <= np.quantile(moving[:, 2], .75)], "spatial absence: highest 25% of moving Z"),
        ("missing_highest_50pct_z", moving[moving[:, 2] <= np.quantile(moving[:, 2], .50)], "low-confidence stress test: highest 50% of moving Z removed"),
    ]
    _, baseline_metric = directional_metrics(apply(moving, final), reference, spacing * 8, "moving -> reference")
    baseline_coverage = baseline_metric["coverage_within_threshold_percent"]
    baseline_rms = baseline_metric["rms_native"]
    criteria = {
        "potentially_unreliable_if": "coverage falls by at least 20 percentage points from the original result OR RMS is at least twice the original RMS",
        "coverage_drop_percentage_points": 20.0,
        "rms_multiplier": 2.0,
        "baseline_coverage_percent": baseline_coverage,
        "baseline_rms_native": baseline_rms,
    }
    for name, altered, condition in tests:
        transform, details, _ = register(reference, altered, normals, spacing)
        _, metric = directional_metrics(apply(altered, transform), reference, spacing * 8, "moving -> reference")
        unreliable = (
            metric["coverage_within_threshold_percent"] <= baseline_coverage - criteria["coverage_drop_percentage_points"]
            or metric["rms_native"] >= baseline_rms * criteria["rms_multiplier"]
        )
        outcomes.append({
            "test": name,
            "condition": condition,
            "potentially_unreliable": unreliable,
            "rms_native": metric["rms_native"],
            "coverage_within_threshold_percent": metric["coverage_within_threshold_percent"],
            "registration_seconds": details["registration_seconds"],
        })
    perturb = np.eye(4); angle = np.radians(8); perturb[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]; perturb[:3, 3] = [spacing * 15, -spacing * 10, spacing * 8]
    transform, details, _ = register(reference, apply(moving, perturb), normals, spacing)
    _, metric = directional_metrics(apply(apply(moving, perturb), transform), reference, spacing * 8, "moving -> reference")
    unreliable = (
        metric["coverage_within_threshold_percent"] <= baseline_coverage - criteria["coverage_drop_percentage_points"]
        or metric["rms_native"] >= baseline_rms * criteria["rms_multiplier"]
    )
    outcomes.append({
        "test": "initialization_8deg_and_translation",
        "condition": "synthetic 8 degree rotation and translation of moving cloud",
        "potentially_unreliable": unreliable,
        "rms_native": metric["rms_native"],
        "coverage_within_threshold_percent": metric["coverage_within_threshold_percent"],
        "registration_seconds": details["registration_seconds"],
    })
    return {
        "method": "controlled perturbations and spatial point absence on temporary subsets; originals untouched; fixed seed 40",
        "coverage_threshold_definition": "8 times the reference median nearest-neighbor spacing",
        "failure_criterion": criteria,
        "tests": outcomes,
    }
