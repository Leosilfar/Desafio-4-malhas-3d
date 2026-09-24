from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def directional_metrics(source_points: np.ndarray, target_points: np.ndarray, coverage_threshold: float,
                        direction: str) -> tuple[np.ndarray, dict]:
    """Evaluate every source point against its Euclidean nearest target point."""
    distances, _ = cKDTree(target_points).query(source_points, workers=-1)
    within_threshold = distances <= coverage_threshold
    return distances, {
        "direction": direction,
        "distance_kind": "Euclidean nearest-neighbor, point cloud to point cloud",
        "points_evaluated": int(len(distances)),
        "sum_squared_distances": float(np.sum(distances ** 2)),
        "rms_native": float(np.sqrt(np.mean(distances ** 2))),
        "mean_native": float(np.mean(distances)),
        "median_native": float(np.quantile(distances, 0.50)),
        "p95_native": float(np.quantile(distances, 0.95)),
        "p99_native": float(np.quantile(distances, 0.99)),
        "maximum_directed_nearest_neighbor_distance_native": float(np.max(distances)),
        "standard_deviation_native": float(np.std(distances)),
        "coverage_threshold_native": float(coverage_threshold),
        "coverage_within_threshold_percent": float(100 * within_threshold.mean()),
        "coverage_count": int(within_threshold.sum()),
    }


def bidirectional_metrics(forward: dict, reverse: dict) -> dict:
    """Combine directional residuals by point count, never by arithmetic mean of RMS values."""
    total_points = forward["points_evaluated"] + reverse["points_evaluated"]
    sum_squared = forward["sum_squared_distances"] + reverse["sum_squared_distances"]
    return {
        "definition": "sqrt((sum(d_moving_to_reference^2) + sum(d_reference_to_moving^2)) / (N_moving + N_reference))",
        "points_evaluated": int(total_points),
        "sum_squared_distances": float(sum_squared),
        "rms_bidirectional_native": float(np.sqrt(sum_squared / total_points)),
        "maximum_bidirectional_nearest_neighbor_distance_native": float(max(
            forward["maximum_directed_nearest_neighbor_distance_native"],
            reverse["maximum_directed_nearest_neighbor_distance_native"],
        )),
    }


def subset_metrics(distances: np.ndarray, label: str) -> dict:
    """Unthresholded statistics for a documented subset of one residual vector."""
    return {
        "definition": label,
        "points_evaluated": int(len(distances)),
        "rms_native": float(np.sqrt(np.mean(distances ** 2))),
        "mean_native": float(np.mean(distances)),
        "median_native": float(np.quantile(distances, 0.50)),
        "p95_native": float(np.quantile(distances, 0.95)),
        "p99_native": float(np.quantile(distances, 0.99)),
        "maximum_directed_nearest_neighbor_distance_native": float(np.max(distances)),
        "standard_deviation_native": float(np.std(distances)),
    }
