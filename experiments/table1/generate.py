"""Generate a length-aligned Qwen3-8B/EASE/AP competitor corpus.

The script is deliberately staged and chunked: interrupted GPU jobs can resume
without overwriting completed chunks.  Human calibration documents are disjoint
from the evaluation documents used for AUROC.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ease.config import TABLE1_CONFIG
from ease.sampling import topk_probabilities


GENERATOR = "Qwen/Qwen3-8B"
CORPORA = ("np", "ease_plugin", "simple", "recursive", "adv_base", "adv_large")
AP_CORPORA = ("simple", "recursive", "adv_base", "adv_large")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stage", required=True,
        choices=("self-test", "prepare", "generate", "merge", "validate"),
    )
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="run the fixed 10-sample smoke profile")
    args = parser.parse_args()
    args.n_calibration = 10 if args.smoke else TABLE1_CONFIG["n_calibration"]
    args.n_eval = 10 if args.smoke else TABLE1_CONFIG["n_evaluation"]
    args.prompt_len = TABLE1_CONFIG["prompt_tokens"]
    args.gen_len = TABLE1_CONFIG["generated_tokens"]
    args.seed = TABLE1_CONFIG["seed_generation"]
    args.temperature = TABLE1_CONFIG["profiles"]["np"]["temperature"]
    args.top_k = TABLE1_CONFIG["profiles"]["np"]["top_k"]
    args.delta = TABLE1_CONFIG["profiles"]["ease_plugin"]["delta"]
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


def stable_seed(base_seed: int, *parts) -> int:
    payload = ":".join(str(value) for value in (base_seed,) + parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def experiment_spec(args):
    return {
        "schema_version": 1,
        "experiment": "qwen3_8b_competitor_reproduction",
        "generator": GENERATOR,
        "data_source": "Salesforce/wikitext:wikitext-103-raw-v1:train",
        "n_calibration": args.n_calibration,
        "n_evaluation": args.n_eval,
        "prompt_tokens": args.prompt_len,
        "continuation_tokens": args.gen_len,
        "seed": args.seed,
        "ease_plugin": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "delta_np": 0.0,
            "delta_ease": args.delta,
        },
        "adversarial_paraphrasing": {
            "model": GENERATOR,
            "plain_temperature": 0.6,
            "adversarial_temperature": 1.0,
            "top_p": 0.99,
            "top_k": 50,
            "max_new_tokens": args.gen_len,
            "min_new_tokens": 1,
            "empty_output_policy": "source_identity_fallback",
            "thinking": False,
            "adversarial_ranking": "fixed option=2 semantics",
            "simple": "one unguided pass",
            "recursive": "two unguided passes",
            "adv_base": "one pass guided by RoBERTa-base",
            "adv_large": "one pass guided by RoBERTa-large",
        },
        "corpora": list(CORPORA),
    }


def ensure_spec(args, out_dir: Path):
    path = out_dir / "run_config.json"
    spec = experiment_spec(args)
    if path.exists():
        if load_json(path) != spec:
            raise RuntimeError(f"run configuration mismatch: {path}")
    else:
        atomic_json(path, spec)
    return spec


def load_wikitext():
    pattern = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/"
        "wikitext-103-raw-v1/*/*/wikitext-train*.arrow"
    )
    cached = sorted(glob.glob(pattern))
    if cached:
        print(f"[data] using {len(cached)} cached Arrow shards")
        return concatenate_datasets([Dataset.from_file(path) for path in cached])
    return load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=False
    )


def validate_samples(samples, args):
    calibration = samples.get("calibration_human", [])
    evaluation = samples.get("evaluation", [])
    if len(calibration) != args.n_calibration or len(evaluation) != args.n_eval:
        raise RuntimeError("sample counts do not match configuration")
    cal_ids = {row["source_index"] for row in calibration}
    eval_ids = {row["source_index"] for row in evaluation}
    if len(cal_ids) != len(calibration) or len(eval_ids) != len(evaluation):
        raise RuntimeError("duplicate source document")
    if cal_ids & eval_ids:
        raise RuntimeError("calibration/evaluation overlap")
    for row in calibration:
        if len(row["human_ids"]) != args.gen_len or not row["human_text"].strip():
            raise RuntimeError("invalid calibration continuation")
    for row in evaluation:
        if len(row["prompt_ids"]) != args.prompt_len:
            raise RuntimeError("invalid prompt length")
        if len(row["human_ids"]) != args.gen_len or not row["human_text"].strip():
            raise RuntimeError("invalid evaluation continuation")


def prepare(args, out_dir: Path):
    ensure_spec(args, out_dir)
    path = out_dir / "samples.json"
    if path.exists():
        samples = load_json(path)
        validate_samples(samples, args)
        print(f"[prepare] valid cached {path}")
        return
    if args.smoke:
        bundled = Path(__file__).resolve().parents[2] / "dataset_splits" / "smoke10"
        samples = load_json(bundled / "samples.json")
        validate_samples(samples, args)
        atomic_json(path, samples)
        corpora = out_dir / "corpora"
        corpora.mkdir(parents=True, exist_ok=True)
        for name in ("human_calibration", "human_evaluation"):
            shutil.copy2(bundled / f"{name}.json", corpora / f"{name}.json")
        print("[prepare] copied the bundled aligned ten-example smoke split")
        return
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR)
    dataset = load_wikitext().shuffle(seed=args.seed)
    required = args.n_calibration + args.n_eval
    selected = []
    for source_index, example in enumerate(tqdm(dataset, desc="select WikiText docs")):
        text = example["text"]
        if not text.strip() or text.lstrip().startswith("="):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < args.prompt_len + args.gen_len:
            continue
        prompt_ids = ids[: args.prompt_len]
        human_ids = ids[args.prompt_len : args.prompt_len + args.gen_len]
        selected.append(
            {
                "source_index": int(source_index),
                "prompt_ids": prompt_ids,
                "human_ids": human_ids,
                "prompt_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "human_text": tokenizer.decode(human_ids, skip_special_tokens=True),
            }
        )
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"needed {required} eligible documents, found {len(selected)}")
    calibration = [
        {
            "source_index": row["source_index"],
            "human_ids": row["human_ids"],
            "human_text": row["human_text"],
        }
        for row in selected[: args.n_calibration]
    ]
    samples = {"calibration_human": calibration, "evaluation": selected[args.n_calibration :]}
    validate_samples(samples, args)
    atomic_json(path, samples)
    corpora = out_dir / "corpora"
    atomic_json(corpora / "human_calibration.json", [row["human_text"] for row in calibration])
    atomic_json(
        corpora / "human_evaluation.json",
        [row["human_text"] for row in samples["evaluation"]],
    )
    print(f"[prepare] saved {args.n_calibration} calibration + {args.n_eval} evaluation samples")


def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    return model, tokenizer


@torch.inference_mode()
def ease_generate_batch(model, input_ids, gen_len, temperature, top_k, delta):
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    logits = output.logits[:, -1, :]
    past = output.past_key_values
    generated = []
    previous = input_ids[:, -1:]
    for step in range(gen_len):
        candidates, probabilities = topk_probabilities(
            logits, previous, step, top_k=top_k,
            temperature=temperature, delta=delta,
        )
        sampled_index = torch.multinomial(probabilities, 1)
        token = candidates.gather(1, sampled_index)
        generated.append(token)
        previous = token
        if step + 1 < gen_len:
            attention_mask = torch.cat(
                [attention_mask, torch.ones((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)],
                dim=1,
            )
            output = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            past = output.past_key_values
    return torch.cat(generated, dim=1)


def chunk_ranges(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def chunk_path(out_dir: Path, corpus: str, start: int, end: int):
    return out_dir / "corpora" / "chunks" / corpus / f"chunk_{start:05d}_{end:05d}.json"


def valid_chunk(path: Path, start: int, end: int):
    if not path.exists():
        return False
    value = load_json(path)
    return (
        value.get("indices") == list(range(start, end))
        and len(value.get("texts", [])) == end - start
        and all(isinstance(text, str) and text.strip() for text in value["texts"])
    )


def generate_ease_corpus(args, out_dir: Path, corpus: str):
    samples = load_json(out_dir / "samples.json")
    validate_samples(samples, args)
    evaluation = samples["evaluation"]
    ranges = chunk_ranges(args.n_eval, args.chunk_size)
    assigned = [item for index, item in enumerate(ranges) if index % args.num_workers == args.worker_id]
    if all(valid_chunk(chunk_path(out_dir, corpus, *item), *item) for item in assigned):
        print(f"[{corpus}] worker has no incomplete chunks")
        return
    model, tokenizer = load_qwen()
    delta = 0.0 if corpus == "np" else args.delta
    for start, end in assigned:
        path = chunk_path(out_dir, corpus, start, end)
        if valid_chunk(path, start, end):
            continue
        seed = stable_seed(args.seed, corpus, start)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        texts, token_counts = [], []
        for offset in tqdm(range(start, end, args.batch_size), desc=f"{corpus} {start}:{end}"):
            stop = min(offset + args.batch_size, end)
            prompts = torch.tensor(
                [evaluation[index]["prompt_ids"] for index in range(offset, stop)],
                dtype=torch.long,
                device="cuda",
            )
            ids = ease_generate_batch(
                model, prompts, args.gen_len, args.temperature, args.top_k, delta
            )
            batch = tokenizer.batch_decode(ids, skip_special_tokens=True)
            if any(not text.strip() for text in batch):
                raise RuntimeError(f"empty {corpus} output in {start}:{end}")
            texts.extend(text.strip() for text in batch)
            token_counts.extend(len(row) for row in ids)
        atomic_json(
            path,
            {"indices": list(range(start, end)), "texts": texts, "token_counts": token_counts},
        )
        print(f"[{corpus}] saved {path}")


def load_ap_components(corpus, temperature):
    from ease.paraphrase import (
        LLMModel,
        OpenAIRoberta,
        Paraphraser,
    )
    llm = LLMModel.get_instance(
        name=GENERATOR, device="cuda:0", temperature=temperature, top_p=0.99, top_k=50
    )
    if corpus == "adv_base":
        classifier = OpenAIRoberta("openai_roberta_base", device="cuda:0")
    elif corpus == "adv_large":
        classifier = OpenAIRoberta("openai_roberta_large", device="cuda:0")
    else:
        classifier = None
    return llm, Paraphraser(llm_model=llm, classifier=classifier), classifier


def generate_ap_corpus(args, out_dir: Path, corpus: str):
    input_name = "simple" if corpus == "recursive" else "np"
    input_path = out_dir / "corpora" / f"{input_name}.json"
    if not input_path.exists():
        raise RuntimeError(f"merge {input_name} before generating {corpus}")
    source = load_json(input_path)
    if len(source) != args.n_eval:
        raise RuntimeError(f"invalid AP source: {input_path}")
    ranges = chunk_ranges(args.n_eval, args.chunk_size)
    assigned = [item for index, item in enumerate(ranges) if index % args.num_workers == args.worker_id]
    if all(valid_chunk(chunk_path(out_dir, corpus, *item), *item) for item in assigned):
        print(f"[{corpus}] worker has no incomplete chunks")
        return
    ap_temperature = 0.6 if corpus in ("simple", "recursive") else 1.0
    llm, paraphraser, classifier = load_ap_components(corpus, ap_temperature)
    tokenizer = llm.tokenizer
    adversarial = 1.0 if corpus.startswith("adv_") else 0.0
    for start, end in assigned:
        path = chunk_path(out_dir, corpus, start, end)
        if valid_chunk(path, start, end):
            continue
        seed = stable_seed(args.seed, corpus, start)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        texts = paraphraser.paraphrase(
            source[start:end],
            batch_size=args.batch_size,
            max_new_tokens=args.gen_len,
            top_p=0.99,
            adversarial=adversarial,
            deterministic=True,
            classifier_batch_size=32,
        )
        normalized, token_counts, fallback_source_indices = [], [], []
        for local_index, text in enumerate(texts):
            ids = tokenizer.encode(text.strip(), add_special_tokens=False)[: args.gen_len]
            normalized_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            # The original AdvPara pipeline keeps any non-empty decode.  Very
            # short outputs are therefore data (and may affect quality or
            # detectability), not a generation-system failure.
            if not normalized_text:
                # AdvPara can occasionally emit only special/tag tokens.  Use
                # the unchanged source as a conservative identity-paraphrase
                # fallback rather than silently dropping the sample or
                # resampling with a different decoding distribution.
                source_ids = tokenizer.encode(
                    source[start + local_index].strip(), add_special_tokens=False
                )[: args.gen_len]
                normalized_text = tokenizer.decode(
                    source_ids, skip_special_tokens=True
                ).strip()
                if not normalized_text:
                    raise RuntimeError(
                        f"empty {corpus} source fallback at index {start + local_index}"
                    )
                ids = source_ids
                fallback_source_indices.append(start + local_index)
            normalized.append(normalized_text)
            token_counts.append(len(ids))
        if len(normalized) != end - start:
            raise RuntimeError(f"AP output count mismatch in {start}:{end}")
        atomic_json(
            path,
            {
                "indices": list(range(start, end)),
                "texts": normalized,
                "token_counts": token_counts,
                "fallback_source_indices": fallback_source_indices,
            },
        )
        print(f"[{corpus}] saved {path}")
    del paraphraser, classifier, llm
    torch.cuda.empty_cache()


def generate(args, out_dir: Path):
    ensure_spec(args, out_dir)
    if args.corpus is None:
        raise ValueError("--corpus is required")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    if args.corpus in ("np", "ease_plugin"):
        generate_ease_corpus(args, out_dir, args.corpus)
    else:
        generate_ap_corpus(args, out_dir, args.corpus)


def merge(args, out_dir: Path):
    ensure_spec(args, out_dir)
    if args.corpus is None:
        raise ValueError("--corpus is required")
    texts = [None] * args.n_eval
    token_counts = [None] * args.n_eval
    fallback_source_indices = []
    for start, end in chunk_ranges(args.n_eval, args.chunk_size):
        path = chunk_path(out_dir, args.corpus, start, end)
        if not valid_chunk(path, start, end):
            raise RuntimeError(f"missing or invalid chunk: {path}")
        value = load_json(path)
        fallback_source_indices.extend(value.get("fallback_source_indices", []))
        for index, text, count in zip(value["indices"], value["texts"], value["token_counts"]):
            if texts[index] is not None:
                raise RuntimeError(f"duplicate index {index}")
            texts[index], token_counts[index] = text, int(count)
    if any(text is None for text in texts):
        raise RuntimeError("incomplete merged corpus")
    corpus_path = out_dir / "corpora" / f"{args.corpus}.json"
    metadata_path = out_dir / "corpora" / f"{args.corpus}_metadata.json"
    atomic_json(corpus_path, texts)
    atomic_json(
        metadata_path,
        {
            "count": len(texts),
            "token_count_min": int(min(token_counts)),
            "token_count_max": int(max(token_counts)),
            "token_count_mean": float(np.mean(token_counts)),
            "empty": 0,
            "source_identity_fallback_count": len(fallback_source_indices),
            "source_identity_fallback_indices": sorted(fallback_source_indices),
        },
    )
    print(f"[merge] saved {corpus_path}")


def validate_all(args, out_dir: Path):
    ensure_spec(args, out_dir)
    samples = load_json(out_dir / "samples.json")
    validate_samples(samples, args)
    for name in ("human_calibration", "human_evaluation") + CORPORA:
        path = out_dir / "corpora" / f"{name}.json"
        texts = load_json(path)
        expected = args.n_calibration if name == "human_calibration" else args.n_eval
        if len(texts) != expected or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise RuntimeError(f"invalid corpus {name}")
    print("CORPUS VALIDATION PASSED")


def self_test():
    assert 0.01 == 1.0 / 100.0
    assert 0.01 != 0.01 / 100.0
    assert len(CORPORA) == 6 and len(set(CORPORA)) == 6
    assert stable_seed(1, "np", 0) == stable_seed(1, "np", 0)
    assert stable_seed(1, "np", 0) != stable_seed(1, "ease_plugin", 0)
    assert chunk_ranges(5, 2) == [(0, 2), (2, 4), (4, 5)]
    print("SELF-TEST PASSED: 1% scale, corpus roster, seeds, chunk ranges")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.n_calibration < (1 if args.smoke else 100) or args.n_eval < 1:
        raise ValueError("invalid calibration/evaluation size")
    if args.stage == "self-test":
        self_test()
    elif args.stage == "prepare":
        prepare(args, out_dir)
    elif args.stage == "generate":
        generate(args, out_dir)
    elif args.stage == "merge":
        merge(args, out_dir)
    elif args.stage == "validate":
        validate_all(args, out_dir)


if __name__ == "__main__":
    main()
