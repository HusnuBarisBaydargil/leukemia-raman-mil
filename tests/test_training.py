"""Check target alignment for the uncertainty-guided training stage."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import make_soft_targets


class SoftTargetTests(unittest.TestCase):
    def test_targets_follow_inner_train_order_and_exclude_validation(self):
        outer_rows = [{"file_name": name} for name in ["aml", "validation", "all"]]
        inner_rows = [outer_rows[2], outer_rows[0]]
        targets = make_soft_targets(
            np.asarray([0.5, 0.99, 0.5], dtype=np.float32),
            np.asarray([1, 1, 0], dtype=np.int64),
            outer_rows, inner_rows, {"entropy_epsilon": 1e-8, "alpha_max": 0.2},
        )
        self.assertEqual(targets.dtype, np.float32)
        np.testing.assert_allclose(targets, [0.1, 0.9], rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
