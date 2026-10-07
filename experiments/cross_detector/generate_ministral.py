"""Generate the aligned Ministral cross-detector extension.

This intentionally preserves the decoding implementation and sample alignment of
``cross_detector_generation.py``.  Only the source model/loading path changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ease.config import CROSS_DETECTOR_CONFIG
from ease.sampling import topk_probabilities


MODEL_KEY = "ministral3_8b"
MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
METHODS = ("vanilla", "ease")
CORPORA = tuple(f"{MODEL_KEY}_{method}" for method in METHODS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("self-test", "prepare", "model-smoke", "generate", "merge", "validate"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-dir")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.n_calibration = 10 if args.smoke else CROSS_DETECTOR_CONFIG["n_calibration"]
    args.n_eval = 10 if args.smoke else CROSS_DETECTOR_CONFIG["n_evaluation"]
    args.gen_len = CROSS_DETECTOR_CONFIG["generated_tokens"]
    args.temperature = CROSS_DETECTOR_CONFIG["temperature"]
    args.top_k = CROSS_DETECTOR_CONFIG["top_k"]
    args.delta = CROSS_DETECTOR_CONFIG["delta_ease"]
    args.seed = CROSS_DETECTOR_CONFIG["seed"]
    return args


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def stable_seed(base, *parts):
    raw = ":".join(str(part) for part in (base,) + parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def digest_texts(texts):
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def ranges(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def spec(args):
    return {
        "schema_version": 1,
        "experiment": "cross_detector_evasion_ministral_200",
        "source_model": {MODEL_KEY: MODEL_ID},
        "n_calibration": args.n_calibration,
        "n_evaluation": args.n_eval,
        "continuation_tokens": args.gen_len,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "delta_vanilla": 0.0,
        "delta_ease": args.delta,
        "prompt_policy": "same exact prompt_text and indices as Qwen3/Llama3 cross-detector study",
        "detector_input_policy": "generated continuation only; prompt excluded",
        "tokenizer_fix_mistral_regex": True,
        "seed": args.seed,
        "corpora": list(CORPORA),
    }


def ensure_spec(args, out_dir):
    path = out_dir / "run_config.json"
    expected = spec(args)
    if path.exists() and load_json(path) != expected:
        raise RuntimeError("Ministral cross-detector configuration mismatch")
    if not path.exists():
        atomic_json(path, expected)


def prepare(args, out_dir):
    if args.base_dir is None:
        raise ValueError("--base-dir is required")
    ensure_spec(args, out_dir)
    base = Path(args.base_dir).expanduser().resolve()
    samples = load_json(base / "samples.json")
    prompts = samples["prompt_texts"][:args.n_eval]
    source_indices = samples["evaluation_source_indices"][:args.n_eval]
    if len(prompts) != args.n_eval or len(source_indices) != args.n_eval or any(not text.strip() for text in prompts):
        raise RuntimeError("invalid aligned prompts")
    atomic_json(out_dir / "samples.json", {
        "base_dir": str(base),
        "evaluation_source_indices": source_indices,
        "prompt_texts": prompts,
        "prompt_sha256": digest_texts(prompts),
    })
    manifest = {}
    for name, count in (("human_calibration", args.n_calibration), ("human_evaluation", args.n_eval)):
        texts = load_json(base / "corpora" / f"{name}.json")[:count]
        if len(texts) != count or any(not text.strip() for text in texts):
            raise RuntimeError(f"invalid reused corpus {name}")
        atomic_json(out_dir / "corpora" / f"{name}.json", texts)
        manifest[name] = {"source": str(base / "corpora" / f"{name}.json"), "count": count, "sha256": digest_texts(texts)}
    atomic_json(out_dir / "reuse_manifest.json", manifest)
    print("MINISTRAL PREPARE PASSED")


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, padding_side="left", fix_mistral_regex=True, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda").eval()
    return model, tokenizer


@torch.inference_mode()
def generate_one(model, input_ids, attention_mask, gen_len, temperature, top_k, delta):
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits, past = output.logits[:, -1, :], output.past_key_values
    previous = input_ids[:, -1:]
    generated = []
    for step in range(gen_len):
        candidates, probabilities = topk_probabilities(
            logits, previous, step, top_k=top_k,
            temperature=temperature, delta=delta,
        )
        token = candidates.gather(1, torch.multinomial(probabilities, 1))
        generated.append(token)
        previous = token
        if step + 1 < gen_len:
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=input_ids.device)], dim=1)
            output = model(input_ids=token, attention_mask=attention_mask, past_key_values=past, use_cache=True)
            logits, past = output.logits[:, -1, :], output.past_key_values
    ids = torch.cat(generated, dim=1)
    return ids


def chunk_path(out_dir, corpus, start, end):
    return out_dir / "corpora" / "chunks" / corpus / f"chunk_{start:05d}_{end:05d}.json"


def valid_chunk(path, start, end, gen_len):
    if not path.exists():
        return False
    value = load_json(path)
    return (
        value.get("indices") == list(range(start, end))
        and len(value.get("texts", [])) == end - start
        and all(isinstance(text, str) and text.strip() for text in value["texts"])
        and value.get("token_counts") == [gen_len] * (end - start)
        and value.get("prompt_included") is False
    )


def generate(args, out_dir, smoke=False):
    ensure_spec(args, out_dir)
    if args.method is None:
        raise ValueError("--method is required")
    corpus = f"{MODEL_KEY}_{args.method}"
    prompts = load_json(out_dir / "samples.json")["prompt_texts"]
    count = 1 if smoke else args.n_eval
    model, tokenizer = load_generator()
    delta = 0.0 if args.method == "vanilla" else args.delta
    all_ranges = ranges(count, 1 if smoke else args.chunk_size)
    assigned = [item for index, item in enumerate(all_ranges) if index % args.num_workers == args.worker_id]
    for start, end in assigned:
        path = chunk_path(out_dir, corpus, start, end)
        if valid_chunk(path, start, end, args.gen_len):
            continue
        texts = []
        for index in range(start, end):
            seed = stable_seed(args.seed, MODEL_KEY, args.method, index)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            encoded = tokenizer(prompts[index], return_tensors="pt", return_token_type_ids=False).to("cuda")
            ids = generate_one(model, encoded.input_ids, encoded.attention_mask, args.gen_len, args.temperature, args.top_k, delta)
            text = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            if not text:
                raise RuntimeError(f"empty continuation at {index}")
            texts.append(text)
        atomic_json(path, {
            "indices": list(range(start, end)), "texts": texts,
            "token_counts": [args.gen_len] * (end - start),
            "prompt_included": False,
        })
        print(f"[{corpus}] saved {start}:{end}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"MINISTRAL MODEL RUN PASSED peak_allocated_gib={peak:.3f}")


def merge(args, out_dir):
    ensure_spec(args, out_dir)
    if args.method is None:
        raise ValueError("--method is required")
    corpus = f"{MODEL_KEY}_{args.method}"
    texts = [None] * args.n_eval
    for start, end in ranges(args.n_eval, args.chunk_size):
        path = chunk_path(out_dir, corpus, start, end)
        if not valid_chunk(path, start, end, args.gen_len):
            raise RuntimeError(f"missing/invalid chunk {path}")
        value = load_json(path)
        for index, text in zip(value["indices"], value["texts"]):
            texts[index] = text
    atomic_json(out_dir / "corpora" / f"{corpus}.json", texts)
    atomic_json(out_dir / "corpora" / f"{corpus}_metadata.json", {
        "count": len(texts), "generated_token_count_min": args.gen_len, "generated_token_count_max": args.gen_len,
        "prompt_included": False, "sha256": digest_texts(texts),
    })
    print(f"merged {corpus}")


def validate(args, out_dir):
    ensure_spec(args, out_dir)
    if load_json(out_dir / "samples.json")["prompt_sha256"] != digest_texts(load_json(out_dir / "samples.json")["prompt_texts"]):
        raise AssertionError("prompt digest mismatch")
    for name, count in (("human_calibration", args.n_calibration), ("human_evaluation", args.n_eval), *[(corpus, args.n_eval) for corpus in CORPORA]):
        texts = load_json(out_dir / "corpora" / f"{name}.json")
        if len(texts) != count or any(not text.strip() for text in texts):
            raise RuntimeError(f"invalid corpus {name}")
    for corpus in CORPORA:
        meta = load_json(out_dir / "corpora" / f"{corpus}_metadata.json")
        if meta["prompt_included"] is not False or meta["generated_token_count_min"] != args.gen_len:
            raise AssertionError(f"metadata validation failed for {corpus}")
    print("MINISTRAL CORPUS VALIDATION PASSED")


def self_test():
    assert 0.01 == 1 / 100 and 0.01 != 0.01 / 100
    assert ranges(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert stable_seed(1, "vanilla", 0) != stable_seed(1, "ease", 0)
    print("MINISTRAL GENERATION SELF-TEST PASSED")


def main():
    args = parse_args(); out_dir = Path(args.out_dir).expanduser().resolve()
    if args.stage == "self-test": self_test()
    elif args.stage == "prepare": prepare(args, out_dir)
    elif args.stage == "model-smoke": generate(args, out_dir, smoke=True)
    elif args.stage == "generate": generate(args, out_dir)
    elif args.stage == "merge": merge(args, out_dir)
    else: validate(args, out_dir)


if __name__ == "__main__":
    main()
