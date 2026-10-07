"""The shared EASE decoding implementation used by plugin and rewrite modes."""

import math

import torch
import torch.nn.functional as F


def ease_probabilities(logits, candidate_ids, previous_ids, step,
                       temperature=1.0, delta=2.0):
    """Return the EASE distribution over an already selected top-k set."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    top_k = logits.shape[-1]
    base_probs = F.softmax(logits.float() / temperature, dim=-1)
    entropy = -(base_probs * torch.log(base_probs + 1e-8)).sum(-1, keepdim=True)
    concentration = torch.clamp(1.0 - entropy / math.log(top_k), 0.0, 1.0)
    adaptive_temperature = temperature + 0.5 * concentration
    hash_input = previous_ids.float() * 137.0 + candidate_ids.float() * 19.0 + step
    shaped_logits = logits.float() + torch.sin(hash_input) * (delta * concentration)
    return F.softmax(shaped_logits / adaptive_temperature, dim=-1)


def topk_probabilities(logits, previous_ids, step, *, top_k=20,
                       temperature=1.0, delta=2.0):
    values, candidates = torch.topk(logits, top_k, dim=-1)
    if delta == 0:
        probabilities = F.softmax(values.float() / temperature, dim=-1)
    else:
        probabilities = ease_probabilities(
            values, candidates, previous_ids, step,
            temperature=temperature, delta=delta,
        )
    return candidates, probabilities
