import unittest

from ease.config import CROSS_DETECTOR_CONFIG, TABLE1_CONFIG


class FrozenConfigTests(unittest.TestCase):
    def test_table1_profiles(self):
        p = TABLE1_CONFIG["profiles"]
        self.assertEqual(p["simple"]["temperature"], 0.6)
        self.assertEqual(p["recursive"]["temperature"], 0.6)
        self.assertEqual(p["adv_base"]["temperature"], 1.0)
        self.assertEqual(p["adv_large"]["ranking"], "language_model_probability_plus_proxy_score")
        self.assertEqual(p["ease_plugin"]["delta"], 2.0)
        self.assertEqual(p["ease_rewrite"]["delta"], 2.0)
        self.assertEqual(TABLE1_CONFIG["target_fpr"], 0.01)

    def test_cross_detector_profile(self):
        c = CROSS_DETECTOR_CONFIG
        self.assertEqual(c["n_evaluation"], 200)
        self.assertEqual(c["temperature"], 1.0)
        self.assertEqual(c["delta_ease"], 2.0)
        self.assertEqual(c["target_fpr"], 0.01)
        self.assertEqual(len(c["detectors"]), 8)


if __name__ == "__main__":
    unittest.main()
