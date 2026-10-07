"""Exercise saved-score merging and independent verification without models."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
    from experiments.table1 import score
except ModuleNotFoundError:
    score = None


@unittest.skipIf(score is None, "Install the experiment dependencies")
class ScorePipelineTests(unittest.TestCase):
    def test_subset_merges_chunks_and_verifies_metrics(self):
        args = SimpleNamespace(
            detector="roberta_base", corpora=["np", "ease_plugin"],
            n_calibration=10, n_eval=10, chunk_size=4, target_fpr=0.01, smoke=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for corpus in ("human_calibration", "human_evaluation", "np", "ease_plugin"):
                values = np.arange(10, dtype=float)
                if corpus == "np":
                    values += 10
                for start, end in score.chunk_ranges(10, 4):
                    score.atomic_npz(
                        score.score_chunk_path(out, args.detector, corpus, start, end),
                        values[start:end], np.arange(start, end),
                    )
            score.finalize(args, out)
            score.verify(args, out)
            summary = score.load_json(out / "summary.json")
            self.assertEqual(summary["methods"]["np"]["detectors"][args.detector]["auroc"], 1.0)
            self.assertEqual(summary["methods"]["ease_plugin"]["detectors"][args.detector]["auroc"], 0.5)
            self.assertEqual(score.load_json(out / "verification.json")["comparisons"], 2)
            summary["methods"]["np"]["detectors"][args.detector]["auroc"] = 0.25
            score.atomic_json(out / "summary.json", summary)
            with self.assertRaises(AssertionError):
                score.verify(args, out)


if __name__ == "__main__":
    unittest.main()
