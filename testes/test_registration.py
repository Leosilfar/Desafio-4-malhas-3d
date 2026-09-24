from __future__ import annotations

import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from src.registration import rigid_from_correspondences, apply
from src.metrics import bidirectional_metrics, directional_metrics, subset_metrics


class RegistrationTests(unittest.TestCase):
    def test_rigid_fit_recovers_known_transform(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        angle = np.deg2rad(25)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        expected = np.eye(4); expected[:3, :3] = rotation; expected[:3, 3] = [2., -1., .5]
        result = rigid_from_correspondences(source, apply(source, expected))
        self.assertTrue(np.allclose(result, expected, atol=1e-10))
        self.assertAlmostEqual(float(np.linalg.det(result[:3, :3])), 1.0, places=10)

    def test_identical_cloud_has_zero_directed_rms(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
        _, metric = directional_metrics(points, points, 0.1, "moving -> reference")
        self.assertAlmostEqual(metric["rms_native"], 0.0, places=12)

    def test_known_translation_is_removed_by_rigid_fit(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        expected = np.eye(4)
        expected[:3, 3] = [3., -2., 0.5]
        result = rigid_from_correspondences(source, apply(source, expected))
        distances, metric = directional_metrics(apply(source, result), apply(source, expected), 0.1, "moving -> reference")
        self.assertAlmostEqual(metric["rms_native"], 0.0, places=12)
        self.assertTrue(np.allclose(distances, 0.0))

    def test_bidirectional_uses_pooled_sum_of_squares(self):
        forward_distances = np.array([1., 3.])
        reverse_distances = np.array([2., 2., 2.])
        forward = {
            "points_evaluated": 2,
            "sum_squared_distances": float(np.sum(forward_distances ** 2)),
            "maximum_directed_nearest_neighbor_distance_native": 3.,
        }
        reverse = {
            "points_evaluated": 3,
            "sum_squared_distances": float(np.sum(reverse_distances ** 2)),
            "maximum_directed_nearest_neighbor_distance_native": 2.,
        }
        result = bidirectional_metrics(forward, reverse)
        self.assertAlmostEqual(result["rms_bidirectional_native"], np.sqrt(22 / 5), places=12)

    def test_maximum_is_unfiltered(self):
        source = np.array([[0., 0., 0.], [100., 0., 0.]])
        target = np.array([[0., 0., 0.]])
        _, metric = directional_metrics(source, target, 1., "moving -> reference")
        self.assertEqual(metric["maximum_directed_nearest_neighbor_distance_native"], 100.)
        self.assertEqual(metric["points_evaluated"], 2)

    def test_subset_metrics_reports_all_unfiltered_statistics(self):
        distances = np.array([1., 2., 10.])
        metric = subset_metrics(distances, "synthetic subset")
        self.assertEqual(metric["points_evaluated"], 3)
        self.assertAlmostEqual(metric["rms_native"], np.sqrt(35), places=12)
        self.assertEqual(metric["maximum_directed_nearest_neighbor_distance_native"], 10.)
        self.assertAlmostEqual(metric["standard_deviation_native"], np.std(distances), places=12)


if __name__ == "__main__":
    unittest.main()
