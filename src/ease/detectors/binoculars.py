"""Binoculars observer/performer score used in the released evaluation."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _perplexity(encoding, logits):
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    shifted_logits = logits[..., :-1, :].contiguous()
    labels = encoding.input_ids[..., 1:].contiguous()
    mask = encoding.attention_mask[..., 1:].contiguous()
    return ((loss(shifted_logits.transpose(1, 2), labels) * mask).sum(1) / mask.sum(1)).cpu().float().numpy()


def _cross_entropy(observer_logits, performer_logits, encoding, pad_token_id):
    vocab = observer_logits.shape[-1]
    probabilities = torch.softmax(observer_logits, dim=-1).view(-1, vocab)
    performer = performer_logits.view(-1, vocab)
    loss = torch.nn.CrossEntropyLoss(reduction="none")(performer, probabilities.to(performer.device))
    loss = loss.view(-1, performer_logits.shape[-2])
    mask = (encoding.input_ids != pad_token_id).to(loss.dtype)
    return ((loss * mask).sum(1) / mask.sum(1)).cpu().float().numpy()


class Binoculars:
    def __init__(self, observer="tiiuae/falcon-7b", performer="tiiuae/falcon-7b-instruct",
                 max_token_observed=512):
        device2 = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        self.observer_device, self.performer_device = "cuda:0", device2
        self.observer = AutoModelForCausalLM.from_pretrained(
            observer, device_map={"": self.observer_device}, torch_dtype=torch.bfloat16
        ).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer, device_map={"": self.performer_device}, torch_dtype=torch.bfloat16
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    @torch.inference_mode()
    def compute_score(self, texts):
        is_single = isinstance(texts, str)
        batch = [texts] if is_single else texts
        encoded = self.tokenizer(
            batch, return_tensors="pt", padding=len(batch) > 1, truncation=True,
            max_length=self.max_token_observed, return_token_type_ids=False,
        )
        observer_logits = self.observer(**encoded.to(self.observer_device)).logits
        performer_logits = self.performer(**encoded.to(self.performer_device)).logits
        ppl = _perplexity(encoded, performer_logits)
        xent = _cross_entropy(observer_logits, performer_logits, encoded, self.tokenizer.pad_token_id)
        scores = ppl / xent
        return float(scores[0]) if is_single else np.asarray(scores)
