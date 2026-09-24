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


class RegistrationTests(unittest.TestCase):
    def test_rigid_fit_recovers_known_transform(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        angle = np.deg2rad(25)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        expected = np.eye(4); expected[:3, :3] = rotation; expected[:3, 3] = [2., -1., .5]
        result = rigid_from_correspondences(source, apply(source, expected))
        self.assertTrue(np.allclose(result, expected, atol=1e-10))
        self.assertAlmostEqual(float(np.linalg.det(result[:3, :3])), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
