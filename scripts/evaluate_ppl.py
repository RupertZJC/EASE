"""Compute the exact Table-1 PPL statistic for a saved text corpus."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ease.config import TABLE1_CONFIG

MODEL_ID = TABLE1_CONFIG["generator"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON list of generated/rephrased texts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_texts(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    texts = value["texts"] if isinstance(value, dict) and "texts" in value else value
    if not isinstance(texts, list) or not texts or any(not isinstance(x, str) or not x.strip() for x in texts):
        raise ValueError("input must be a non-empty JSON list of final output texts")
    return texts


@torch.inference_mode()
def per_text_perplexities(model, tokenizer, texts, batch_size, device):
    values = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size], padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(device)
        labels = encoded.input_ids[:, 1:]
        mask = encoded.attention_mask[:, 1:].float()
        if bool((mask.sum(1) == 0).any()):
            raise ValueError("every text must contain at least two scoring tokens")
        logits = model(**encoded).logits[:, :-1].float()
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).reshape(labels.shape)
        mean_loss = (losses * mask).sum(1) / mask.sum(1)
        values.extend(torch.exp(mean_loss).cpu().tolist())
        print(f"[ppl] {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return values


def main():
    args = parse_args()
    texts = load_texts(args.input)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    raw = per_text_perplexities(model, tokenizer, texts, args.batch_size, args.device)
    values = np.asarray(raw, dtype=np.float64)
    result = {
        "definition": "arithmetic mean of per-text perplexities; final output text only",
        "model": MODEL_ID,
        "max_length": 512,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "n": int(values.size),
        "per_text_ppl": raw,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)
    print(f"Table-1 PPL mean: {result['mean']:.10f}")


if __name__ == "__main__":
    main()
