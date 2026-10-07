"""Black-box DNA-GPT scorer used by the cross-detector experiment."""

import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dna_overlap import official_overlap_score


class DNAGPT:
    def __init__(self, model_name, temperature=0.7, regenerations=10, max_new_tokens=200):
        self.temperature = float(temperature)
        self.regenerations = int(regenerations)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda").eval()

    @torch.inference_mode()
    def score_one(self, text, seed=0):
        words = text.split()
        split = max(1, len(words) // 2)
        prefix, suffix = " ".join(words[:split]), " ".join(words[split:])
        if not suffix.strip():
            return np.nan
        encoded = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        prompt_length = encoded.input_ids.shape[1]
        scores = []
        for regeneration in range(self.regenerations):
            local_seed = seed + regeneration
            random.seed(local_seed)
            np.random.seed(local_seed % (2**32))
            torch.manual_seed(local_seed)
            torch.cuda.manual_seed_all(local_seed)
            output = self.model.generate(
                **encoded, do_sample=True, temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            continuation = self.tokenizer.decode(
                output[0, prompt_length:], skip_special_tokens=True
            ).strip()
            scores.append(official_overlap_score(suffix, continuation))
        return float(np.mean(scores))

    def score(self, texts, seed=0):
        return np.asarray([
            self.score_one(text, seed + index * 1009)
            for index, text in enumerate(tqdm(texts, desc="DNA-GPT"))
        ], dtype=np.float64)
