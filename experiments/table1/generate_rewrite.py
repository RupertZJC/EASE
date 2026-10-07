"""Generate one-pass EASE paraphrases from the aligned Table-1 NP corpus.

The input task is deliberately identical to the AP/Simple baseline: the full
NP text is inserted into the same chat paraphrase prompt.  Only newly generated
tokens are saved and later exposed to quality metrics or detectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ease.config import TABLE1_CONFIG
from ease.sampling import topk_probabilities


MODEL_ID = TABLE1_CONFIG["generator"]
METHODS = ("ease_rewrite_d2",)
DELTAS = {"ease_rewrite_d2": TABLE1_CONFIG["profiles"]["ease_rewrite"]["delta"]}
AP_SYSTEM_PROMPT = (
    "You are a rephraser. Given any input text, you are supposed to "
    "rephrase the text without changing its meaning and content, while "
    "maintaining the text quality. Also, it is important for you to output "
    "a rephrased text that has a different style from the input text. You "
    "can not just make a few changes to the input text. The input text is "
    "given below. Print your rephrased output text between tags <TAG> and "
    "</TAG>."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("self-test", "prepare", "generate", "merge", "validate"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.method = "ease_rewrite_d2"
    profile = TABLE1_CONFIG["profiles"]["ease_rewrite"]
    args.n_calibration = 10 if args.smoke else TABLE1_CONFIG["n_calibration"]
    args.n_eval = 10 if args.smoke else TABLE1_CONFIG["n_evaluation"]
    args.max_new_tokens = TABLE1_CONFIG["generated_tokens"]
    args.temperature = profile["temperature"]
    args.top_k = profile["top_k"]
    args.seed = TABLE1_CONFIG["seed_rewrite"]
    return args


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, *parts) -> int:
    payload = ":".join(map(str, (base,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def ranges(count: int, size: int):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def chunk_path(out: Path, method: str, start: int, end: int):
    return out / "corpora" / "chunks" / method / f"chunk_{start:05d}_{end:05d}.json"


def valid_chunk(path: Path, start: int, end: int):
    if not path.exists():
        return False
    value = load_json(path)
    return (
        value.get("indices") == list(range(start, end))
        and len(value.get("texts", [])) == end - start
        and len(value.get("input_token_counts", [])) == end - start
        and len(value.get("output_token_counts", [])) == end - start
        and all(isinstance(text, str) and text.strip() for text in value.get("texts", []))
    )


def run_config(args):
    return {
        "schema_version": 1,
        "experiment": "ease_rewrite_aligned_table1_qwen3_8b",
        "model": MODEL_ID,
        "task": "one-pass paraphrase of the complete aligned NP text",
        "prompt": AP_SYSTEM_PROMPT,
        "prompt_sha256": hashlib.sha256(AP_SYSTEM_PROMPT.encode()).hexdigest(),
        "input_truncation": False,
        "saved_and_scored_text": "newly generated tokens only; input NP and chat prompt excluded",
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "thinking": False,
        "n_calibration": args.n_calibration,
        "n_evaluation": args.n_eval,
        "seed": args.seed,
        "methods": {method: {"delta": DELTAS[method], "passes": 1} for method in METHODS},
    }


def ensure_config(args, out: Path):
    path = out / "ease_rewrite_run_config.json"
    expected = run_config(args)
    if path.exists() and load_json(path) != expected:
        raise RuntimeError(f"configuration mismatch: {path}")
    if not path.exists():
        atomic_json(path, expected)


def prepare(args, out: Path):
    if not args.source_dir:
        raise ValueError("--source-dir is required")
    source = Path(args.source_dir).expanduser().resolve()
    ensure_config(args, out)
    copied = {}
    for name, count in (("human_calibration", args.n_calibration), ("human_evaluation", args.n_eval), ("np", args.n_eval)):
        source_path = source / "corpora" / f"{name}.json"
        source_values = load_json(source_path)
        if len(source_values) < count:
            raise RuntimeError(f"source corpus is too short: {source_path}")
        values = source_values[:count]
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise RuntimeError(f"invalid source corpus: {source_path}")
        target = out / "corpora" / f"{name}.json"
        if len(source_values) == count and source_path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        else:
            atomic_json(target, values)
        copied[name] = {
            "count": count,
            "slice": [0, count],
            "source_count": len(source_values),
            "source_sha256": sha256(source_path),
            "copied_sha256": sha256(target),
        }
    atomic_json(out / "ease_rewrite_reuse_manifest.json", {"source_dir": str(source), "corpora": copied})
    print("EASE-REWRITE PREPARE PASSED")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    return model, tokenizer


def build_prompts(tokenizer, texts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": AP_SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) + "<TAG> "
        for text in texts
    ]


@torch.inference_mode()
def ease_generate(model, input_ids, attention_mask, max_new_tokens, temperature, top_k, delta, eos_token_id):
    if temperature != 1.0:
        raise ValueError("aligned experiment is frozen to T=1.0")
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits = output.logits[:, -1, :]
    past = output.past_key_values
    previous = input_ids[:, -1:]
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    rows = [[] for _ in range(input_ids.size(0))]
    for step in range(max_new_tokens):
        candidates, probabilities = topk_probabilities(
            logits, previous, step, top_k=top_k,
            temperature=temperature, delta=delta,
        )
        sampled = candidates.gather(1, torch.multinomial(probabilities, 1))
        sampled[finished] = eos_token_id
        for index, token in enumerate(sampled.squeeze(1).tolist()):
            if not finished[index]:
                rows[index].append(token)
                if token == eos_token_id:
                    finished[index] = True
        if bool(finished.all()):
            break
        previous = sampled
        attention_mask = torch.cat(
            [attention_mask, torch.ones((input_ids.size(0), 1), dtype=attention_mask.dtype, device=input_ids.device)],
            dim=1,
        )
        output = model(
            input_ids=sampled,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        logits = output.logits[:, -1, :]
        past = output.past_key_values
    return rows


def clean_response(text: str) -> str:
    text = text.replace("<TAG>", "").replace("</TAG>", "").strip()
    for marker in ("Note: I rephrased", "Note: I've rephrased", "Note: I have rephrased", "(Note:"):
        if marker in text:
            text = text.split(marker)[0].strip()
            break
    return text


@torch.inference_mode()
def generate(args, out: Path):
    ensure_config(args, out)
    if args.method is None:
        raise ValueError("--method is required")
    source = load_json(out / "corpora" / "np.json")
    if len(source) != args.n_eval:
        raise RuntimeError("NP corpus count mismatch")
    pending = [(s, e) for s, e in ranges(args.n_eval, args.chunk_size) if not valid_chunk(chunk_path(out, args.method, s, e), s, e)]
    if not pending:
        print(f"[{args.method}] nothing pending")
        return
    model, tokenizer = load_model()
    delta = DELTAS[args.method]
    for start, end in pending:
        # Deliberately omit delta/method: paired chunks use identical RNG seeds.
        seed = stable_seed(args.seed, "ease_rewrite", start)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        texts, input_counts, output_counts, fallbacks = [], [], [], []
        for offset in range(start, end, args.batch_size):
            stop = min(offset + args.batch_size, end)
            prompts = build_prompts(tokenizer, source[offset:stop])
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            input_counts.extend(encoded.attention_mask.sum(1).tolist())
            encoded = encoded.to("cuda")
            rows = ease_generate(
                model, encoded.input_ids, encoded.attention_mask,
                args.max_new_tokens, args.temperature, args.top_k, delta,
                tokenizer.eos_token_id,
            )
            for local_index, ids in enumerate(rows):
                text = clean_response(tokenizer.decode(ids, skip_special_tokens=True))
                normalized_ids = tokenizer.encode(text, add_special_tokens=False)[: args.max_new_tokens]
                text = tokenizer.decode(normalized_ids, skip_special_tokens=True).strip()
                if not text:
                    text = source[offset + local_index]
                    normalized_ids = tokenizer.encode(text, add_special_tokens=False)[: args.max_new_tokens]
                    text = tokenizer.decode(normalized_ids, skip_special_tokens=True).strip()
                    fallbacks.append(offset + local_index)
                texts.append(text)
                output_counts.append(len(normalized_ids))
        atomic_json(chunk_path(out, args.method, start, end), {
            "indices": list(range(start, end)),
            "texts": texts,
            "input_token_counts": input_counts,
            "output_token_counts": output_counts,
            "identity_fallback_indices": fallbacks,
            "seed": seed,
            "delta": delta,
        })
        print(f"[{args.method}] saved {start}:{end}", flush=True)


def merge(args, out: Path):
    ensure_config(args, out)
    if args.method is None:
        raise ValueError("--method is required")
    texts, input_counts, output_counts, fallbacks = [], [], [], []
    seeds = []
    for start, end in ranges(args.n_eval, args.chunk_size):
        path = chunk_path(out, args.method, start, end)
        if not valid_chunk(path, start, end):
            raise RuntimeError(f"missing chunk: {path}")
        value = load_json(path)
        texts.extend(value["texts"])
        input_counts.extend(value["input_token_counts"])
        output_counts.extend(value["output_token_counts"])
        fallbacks.extend(value.get("identity_fallback_indices", []))
        seeds.append(value["seed"])
    atomic_json(out / "corpora" / f"{args.method}.json", texts)
    atomic_json(out / "corpora" / f"{args.method}_metadata.json", {
        "count": len(texts),
        "delta": DELTAS[args.method],
        "input_token_count": {"min": min(input_counts), "max": max(input_counts), "mean": float(np.mean(input_counts))},
        "output_token_count": {"min": min(output_counts), "max": max(output_counts), "mean": float(np.mean(output_counts))},
        "identity_fallback_indices": sorted(fallbacks),
        "chunk_seeds": seeds,
    })
    print(f"[{args.method}] MERGE PASSED")


def validate(args, out: Path):
    config = load_json(out / "ease_rewrite_run_config.json")
    manifest = load_json(out / "ease_rewrite_reuse_manifest.json")
    if config["temperature"] != 1.0 or config["input_truncation"] is not False:
        raise AssertionError("temperature/input truncation invariant failed")
    if config["prompt_sha256"] != hashlib.sha256(AP_SYSTEM_PROMPT.encode()).hexdigest():
        raise AssertionError("prompt fingerprint mismatch")
    for name, count in (("human_calibration", args.n_calibration), ("human_evaluation", args.n_eval), ("np", args.n_eval)):
        path = out / "corpora" / f"{name}.json"
        if sha256(path) != manifest["corpora"][name]["copied_sha256"] or len(load_json(path)) != count:
            raise AssertionError(f"reused corpus mismatch: {name}")
    metadata = {}
    for method in METHODS:
        values = load_json(out / "corpora" / f"{method}.json")
        meta = load_json(out / "corpora" / f"{method}_metadata.json")
        if len(values) != args.n_eval or any(not text.strip() or "<TAG>" in text or "</TAG>" in text for text in values):
            raise AssertionError(f"invalid generated corpus: {method}")
        if meta["output_token_count"]["max"] > args.max_new_tokens:
            raise AssertionError("output length exceeded")
        metadata[method] = meta
    atomic_json(out / "corpus_verification.json", {
        "status": "passed",
        "n_evaluation": args.n_eval,
        "temperature": 1.0,
        "deltas": [2.0],
        "prompt_excluded_from_saved_output": True,
        "np_input_truncation": False,
    })
    print("EASE-REWRITE CORPUS VALIDATION PASSED")


def self_test():
    assert METHODS == ("ease_rewrite_d2",)
    assert DELTAS[METHODS[0]] == 2.0
    assert ranges(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert clean_response("<TAG> hello </TAG>") == "hello"
    assert stable_seed(1, "ease_rewrite", 0) == stable_seed(1, "ease_rewrite", 0)
    print("EASE-REWRITE SELF-TEST PASSED")


def main():
    args = parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "prepare":
        prepare(args, out)
    elif args.stage == "generate":
        generate(args, out)
    elif args.stage == "merge":
        merge(args, out)
    else:
        validate(args, out)


if __name__ == "__main__":
    main()
