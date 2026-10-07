"""Prepare and generate the aligned 200-sample cross-detector corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ease.config import CROSS_DETECTOR_CONFIG
from ease.sampling import topk_probabilities


MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "llama3_8b": "NousResearch/Meta-Llama-3-8B-Instruct",
}
METHODS = ("vanilla", "ease")
CORPORA = tuple(f"{model}_{method}" for model in MODELS for method in METHODS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("self-test", "prepare", "generate", "merge", "validate"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
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
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def spec(args):
    return {
        "schema_version": 1,
        "experiment": "cross_detector_evasion_200",
        "source_models": MODELS,
        "n_calibration": args.n_calibration,
        "n_evaluation": args.n_eval,
        "continuation_tokens": args.gen_len,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "delta_vanilla": 0.0,
        "delta_ease": args.delta,
        "prompt_policy": "same exact prompt_text for both source models",
        "detector_input_policy": "generated continuation only; prompt excluded",
        "seed": args.seed,
        "corpora": list(CORPORA),
    }


def ensure_spec(args, out_dir):
    path = out_dir / "run_config.json"
    expected = spec(args)
    if path.exists() and load_json(path) != expected:
        raise RuntimeError("cross-detector run configuration mismatch")
    if not path.exists():
        atomic_json(path, expected)


def prepare(args, out_dir):
    if args.source_dir is None:
        raise ValueError("--source-dir is required")
    ensure_spec(args, out_dir)
    source = Path(args.source_dir)
    samples = load_json(source / "samples.json")
    evaluation = samples["evaluation"][:args.n_eval]
    if len(evaluation) != args.n_eval:
        raise RuntimeError("not enough aligned source samples")
    prompts = [row["prompt_text"] for row in evaluation]
    if any(not text.strip() for text in prompts):
        raise RuntimeError("empty prompt")
    atomic_json(out_dir / "samples.json", {
        "source_dir": str(source),
        "evaluation_source_indices": [row["source_index"] for row in evaluation],
        "prompt_texts": prompts,
        "prompt_sha256": digest_texts(prompts),
    })
    mappings = {
        "human_calibration": "human_calibration",
        "human_evaluation": "human_evaluation",
        "qwen3_8b_vanilla": "np",
        "qwen3_8b_ease": "ease_plugin",
    }
    manifest = {}
    for destination, origin in mappings.items():
        count = args.n_calibration if destination == "human_calibration" else args.n_eval
        texts = load_json(source / "corpora" / f"{origin}.json")[:count]
        if len(texts) != count or any(not text.strip() for text in texts):
            raise RuntimeError(f"invalid reused corpus {origin}")
        atomic_json(out_dir / "corpora" / f"{destination}.json", texts)
        manifest[destination] = {
            "source": str(source / "corpora" / f"{origin}.json"),
            "count": count,
            "sha256": digest_texts(texts),
        }
    atomic_json(out_dir / "reuse_manifest.json", manifest)
    print("CROSS PREPARE PASSED")


@torch.inference_mode()
def generate_batch(model, input_ids, attention_mask, gen_len, temperature, top_k, delta):
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
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=input_ids.device),
            ], dim=1)
            output = model(
                input_ids=token, attention_mask=attention_mask,
                past_key_values=past, use_cache=True,
            )
            logits, past = output.logits[:, -1, :], output.past_key_values
    return torch.cat(generated, dim=1)


def ranges(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def chunk_path(out_dir, corpus, start, end):
    return out_dir / "corpora" / "chunks" / corpus / f"chunk_{start:05d}_{end:05d}.json"


def valid_chunk(path, start, end):
    if not path.exists():
        return False
    value = load_json(path)
    return (
        value.get("indices") == list(range(start, end))
        and len(value.get("texts", [])) == end - start
        and all(isinstance(text, str) and text.strip() for text in value["texts"])
        and value.get("prompt_included") is False
    )


def generate(args, out_dir):
    ensure_spec(args, out_dir)
    if args.model is None or args.method is None:
        raise ValueError("--model and --method are required")
    corpus = f"{args.model}_{args.method}"
    if (out_dir / "corpora" / f"{corpus}.json").exists():
        print(f"reusing complete {corpus}")
        return
    prompts = load_json(out_dir / "samples.json")["prompt_texts"]
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    delta = 0.0 if args.method == "vanilla" else args.delta
    assigned = [item for index, item in enumerate(ranges(args.n_eval, args.chunk_size))
                if index % args.num_workers == args.worker_id]
    for start, end in assigned:
        path = chunk_path(out_dir, corpus, start, end)
        if valid_chunk(path, start, end):
            continue
        seed = stable_seed(args.seed, args.model, args.method, start)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        texts, token_counts = [], []
        for offset in range(start, end, args.batch_size):
            stop = min(offset + args.batch_size, end)
            encoded = tokenizer(
                prompts[offset:stop], padding=True, return_tensors="pt",
                return_token_type_ids=False,
            ).to("cuda")
            generated = generate_batch(
                model, encoded.input_ids, encoded.attention_mask,
                args.gen_len, args.temperature, args.top_k, delta,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            if any(not text.strip() for text in decoded):
                raise RuntimeError(f"empty generated continuation in {corpus} {offset}:{stop}")
            texts.extend(text.strip() for text in decoded)
            token_counts.extend([generated.shape[1]] * generated.shape[0])
        atomic_json(path, {
            "indices": list(range(start, end)),
            "texts": texts,
            "token_counts": token_counts,
            "prompt_included": False,
        })
        print(f"[{corpus}] saved {start}:{end}", flush=True)


def merge(args, out_dir):
    ensure_spec(args, out_dir)
    if args.model is None or args.method is None:
        raise ValueError("--model and --method are required")
    corpus = f"{args.model}_{args.method}"
    if (out_dir / "corpora" / f"{corpus}.json").exists():
        print(f"reusing merged {corpus}")
        return
    texts = [None] * args.n_eval
    counts = [None] * args.n_eval
    for start, end in ranges(args.n_eval, args.chunk_size):
        path = chunk_path(out_dir, corpus, start, end)
        if not valid_chunk(path, start, end):
            raise RuntimeError(f"missing chunk {path}")
        value = load_json(path)
        for index, text, count in zip(value["indices"], value["texts"], value["token_counts"]):
            texts[index], counts[index] = text, count
    if any(text is None for text in texts):
        raise RuntimeError("incomplete cross corpus")
    atomic_json(out_dir / "corpora" / f"{corpus}.json", texts)
    atomic_json(out_dir / "corpora" / f"{corpus}_metadata.json", {
        "count": len(texts), "generated_token_count_min": min(counts),
        "generated_token_count_max": max(counts),
        "prompt_included": False, "sha256": digest_texts(texts),
    })
    print(f"merged {corpus}")


def validate(args, out_dir):
    ensure_spec(args, out_dir)
    expected = {"human_calibration": args.n_calibration, "human_evaluation": args.n_eval}
    expected.update({corpus: args.n_eval for corpus in CORPORA})
    for corpus, count in expected.items():
        texts = load_json(out_dir / "corpora" / f"{corpus}.json")
        if len(texts) != count or any(not text.strip() for text in texts):
            raise RuntimeError(f"invalid corpus {corpus}")
    for corpus in CORPORA:
        if corpus.startswith("llama"):
            metadata = load_json(out_dir / "corpora" / f"{corpus}_metadata.json")
            if metadata["prompt_included"] is not False:
                raise AssertionError("prompt leakage flag")
    print("CROSS CORPUS VALIDATION PASSED")


def self_test():
    assert 0.01 == 1 / 100
    assert len(CORPORA) == 4
    assert stable_seed(1, "a") != stable_seed(1, "b")
    assert ranges(5, 2) == [(0, 2), (2, 4), (4, 5)]
    print("CROSS GENERATION SELF-TEST PASSED")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "prepare":
        prepare(args, out_dir)
    elif args.stage == "generate":
        generate(args, out_dir)
    elif args.stage == "merge":
        merge(args, out_dir)
    else:
        validate(args, out_dir)


if __name__ == "__main__":
    main()
