import unittest

import numpy as np

from ease.metrics import auroc, calibrate_threshold_at_fpr, tpr_at_threshold


class MetricTests(unittest.TestCase):
    def test_one_percent_is_fraction_point_zero_one(self):
        human = np.arange(200, dtype=float)
        threshold, fpr = calibrate_threshold_at_fpr(human, 0.01)
        self.assertLessEqual(fpr, 0.01)
        self.assertNotEqual(0.01, 0.01 / 100)
        self.assertGreaterEqual(threshold, 198.0)

    def test_ties_never_exceed_target(self):
        human = np.concatenate([np.zeros(190), np.ones(10)])
        threshold, fpr = calibrate_threshold_at_fpr(human, 0.01)
        self.assertGreater(threshold, 1.0)
        self.assertEqual(fpr, 0.0)

    def test_fixed_threshold_tpr(self):
        self.assertEqual(tpr_at_threshold([0.0, 1.0, 2.0, 3.0], 2.0), 0.5)

    def test_auc_direction(self):
        self.assertEqual(auroc([0.0, 0.1], [0.9, 1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
