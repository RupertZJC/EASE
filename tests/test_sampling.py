import unittest

try:
    import torch
except ModuleNotFoundError:  # permits lightweight config/metric checks
    torch = None

if torch is not None:
    from ease.sampling import topk_probabilities
else:
    topk_probabilities = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this lightweight test environment")
class SamplingTests(unittest.TestCase):
    def test_ease_preserves_uniform_distribution_and_changes_concentrated_one(self):
        previous = torch.tensor([[42]])
        for logits, should_change in ((torch.zeros(1, 20), False),
                                      (torch.linspace(-4, 4, 20).reshape(1, 20), True)):
            candidates, plain = topk_probabilities(logits, previous, 3, delta=0.0)
            shaped_candidates, shaped = topk_probabilities(logits, previous, 3, delta=2.0)
            self.assertTrue(torch.equal(candidates, shaped_candidates))
            self.assertTrue(torch.isfinite(shaped).all())
            self.assertTrue((shaped >= 0).all())
            self.assertTrue(torch.allclose(shaped.sum(-1), torch.ones(1)))
            self.assertEqual(not torch.allclose(plain, shaped), should_change)

    def test_delta_zero_is_plain_topk_temperature_sampling(self):
        logits = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        candidates, probabilities = topk_probabilities(
            logits, torch.tensor([[0]]), 0, top_k=3, temperature=1.0, delta=0.0
        )
        values = logits.gather(1, candidates)
        self.assertTrue(torch.allclose(probabilities, torch.softmax(values, dim=-1)))
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
