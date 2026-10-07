"""Detector scoring and verified metrics for the Ministral cross-detector extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ease.config import CROSS_DETECTOR_CONFIG
from ease.metrics import auroc as fixed_auroc
from experiments.cross_detector.score import (
    BinocularsScorer, DistributionScorer, PerturbationScorer,
    atomic_json, atomic_npz, independent_auc, threshold_at_fpr,
)


MODEL_KEY = "ministral3_8b"
MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
CORPORA = ("human_calibration", "human_evaluation", f"{MODEL_KEY}_vanilla", f"{MODEL_KEY}_ease")
METHODS = ("likelihood", "entropy", "logrank", "lrr", "npr", "dna_gpt", "detectgpt", "binoculars")
FAMILIES = ("distribution", "perturb", "dna_ministral", "binoculars")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("self-test", "score", "finalize", "verify"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--corpus", choices=CORPORA)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.n_calibration = 10 if args.smoke else CROSS_DETECTOR_CONFIG["n_calibration"]
    args.n_eval = 10 if args.smoke else CROSS_DETECTOR_CONFIG["n_evaluation"]
    args.target_fpr = CROSS_DETECTOR_CONFIG["target_fpr"]
    args.n_perturbations = CROSS_DETECTOR_CONFIG["n_perturbations"]
    args.n_regenerations = CROSS_DETECTOR_CONFIG["n_regenerations"]
    args.seed = CROSS_DETECTOR_CONFIG["seed"]
    return args


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ranges(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def score_names(family):
    return {
        "distribution": ("likelihood", "entropy", "logrank", "lrr"),
        "perturb": ("detectgpt", "npr"),
        "dna_ministral": ("dna_gpt_ministral3_8b",),
        "binoculars": ("binoculars",),
    }[family]


def chunk_path(out, name, corpus, start, end):
    return out / "scores" / "chunks" / name / corpus / f"chunk_{start:05d}_{end:05d}.npz"


def valid_chunk(path, start, end):
    if not path.exists(): return False
    with np.load(path) as archive:
        scores, indices = np.asarray(archive["scores"], dtype=np.float64), np.asarray(archive["indices"], dtype=np.int64)
    return scores.shape == (end - start,) and np.isfinite(scores).all() and np.array_equal(indices, np.arange(start, end))


def family_scorer(args):
    if args.family == "distribution": return DistributionScorer()
    if args.family == "perturb": return PerturbationScorer(args.n_perturbations)
    if args.family == "binoculars": return BinocularsScorer()
    if args.family == "dna_ministral":
        return MinistralDNAGPT(regenerations=args.n_regenerations)
    raise ValueError("unknown family")


class MinistralDNAGPT:
    """Official DNA-GPT statistic with a multimodal-model-compatible loader."""

    def __init__(self, temperature=0.7, regenerations=10, max_new_tokens=200):
        self.temperature = float(temperature)
        self.regenerations = int(regenerations)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, padding_side="left", fix_mistral_regex=True, trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
        ).to("cuda").eval()

    @torch.inference_mode()
    def score_one(self, text, seed=0):
        from ease.detectors.dna_overlap import official_overlap_score
        words = text.split()
        split = max(1, len(words) // 2)
        prefix, suffix = " ".join(words[:split]), " ".join(words[split:])
        if not suffix.strip(): return np.nan
        encoded = self.tokenizer(prefix, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        prompt_length = encoded.input_ids.shape[1]
        scores = []
        for regeneration in range(self.regenerations):
            local_seed = seed + regeneration
            random.seed(local_seed); np.random.seed(local_seed % (2**32)); torch.manual_seed(local_seed); torch.cuda.manual_seed_all(local_seed)
            output = self.model.generate(
                **encoded, do_sample=True, temperature=self.temperature,
                max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            continuation = self.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()
            scores.append(official_overlap_score(suffix, continuation))
        return float(np.mean(scores))

    def score(self, texts, seed=0):
        return np.asarray([self.score_one(text, seed + index * 1009) for index, text in enumerate(tqdm(texts, desc="DNA-GPT"))], dtype=np.float64)


def score(args, out):
    if args.family is None or args.corpus is None: raise ValueError("--family and --corpus are required")
    count = args.n_calibration if args.corpus == "human_calibration" else args.n_eval
    assigned = [item for i, item in enumerate(ranges(count, args.chunk_size)) if i % args.num_workers == args.worker_id]
    names = score_names(args.family)
    pending = [item for item in assigned if not all(valid_chunk(chunk_path(out, name, args.corpus, *item), *item) for name in names)]
    if not pending:
        print("worker has no pending score chunks"); return
    texts = load_json(out / "corpora" / f"{args.corpus}.json")
    if len(texts) != count: raise RuntimeError("score corpus count mismatch")
    scorer = family_scorer(args)
    for start, end in pending:
        subset = texts[start:end]
        if args.family == "perturb":
            results = scorer.score(subset, [args.seed + i * 1009 for i in range(start, end)])
        elif args.family == "dna_ministral":
            results = {names[0]: scorer.score(subset, seed=args.seed + start * 1009)}
        else:
            results = scorer.score(subset)
        for name, values in results.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (end - start,) or not np.isfinite(values).all(): raise RuntimeError(f"invalid scores {name}/{args.corpus}/{start}:{end}")
            atomic_npz(chunk_path(out, name, args.corpus, start, end), scores=values, indices=np.arange(start, end))
        print(f"[{args.family}/{args.corpus}] {start}:{end}", flush=True)


def merge(out, name, corpus, count, chunk_size):
    path = out / "scores" / f"{name}_{corpus}.npy"
    if path.exists():
        values = np.load(path)
        if values.shape == (count,) and np.isfinite(values).all(): return values
    values = np.full(count, np.nan)
    for start, end in ranges(count, chunk_size):
        path_chunk = chunk_path(out, name, corpus, start, end)
        if not valid_chunk(path_chunk, start, end): raise RuntimeError(f"missing score chunk {path_chunk}")
        with np.load(path_chunk) as archive: values[start:end] = archive["scores"]
    if not np.isfinite(values).all(): raise RuntimeError("non-finite merged scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle: np.save(handle, values)
    return values


def key(method):
    return "dna_gpt_ministral3_8b" if method == "dna_gpt" else method


def normalize_score_direction(method, values):
    """Return scores with larger values consistently meaning more machine-like.

    ``DistributionScorer`` exposes Shannon entropy as a positive quantity, but
    low entropy is the machine-like direction used by the cross-detector table.
    Negating here preserves the raw detector arrays while calibrating and
    reporting entropy in the same direction as every other detector.
    """
    values = np.asarray(values, dtype=np.float64)
    return -values if method == "entropy" else values


def finalize(args, out):
    summary = {
        "metric": "AUROC / TPR@1%FPR (lower is better for evasion)", "target_fpr": args.target_fpr,
        "score_direction": "higher_is_more_machine_like", "n_calibration": args.n_calibration, "n_evaluation": args.n_eval,
        "source_model": MODEL_ID,
        "detector_settings": {"distribution_scoring_model": "EleutherAI/gpt-neo-2.7B", "detectgpt_npr_mask_model": "t5-small", "n_perturbations": args.n_perturbations, "dna_source_model": MODEL_ID, "dna_regenerations": args.n_regenerations, "binoculars_pair": ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"]},
        "results": {},
    }
    rows = []
    for method in METHODS:
        name = key(method)
        human_cal = normalize_score_direction(method, merge(out, name, "human_calibration", args.n_calibration, args.chunk_size))
        human_eval = normalize_score_direction(method, merge(out, name, "human_evaluation", args.n_eval, args.chunk_size))
        threshold, calibration_fpr = threshold_at_fpr(human_cal, args.target_fpr)
        summary["results"][method] = {}
        row = {"detector": method}
        for variant in ("vanilla", "ease"):
            machine = normalize_score_direction(method, merge(out, name, f"{MODEL_KEY}_{variant}", args.n_eval, args.chunk_size))
            result = {"auroc": fixed_auroc(human_eval, machine), "tpr_at_1pct_fpr": float(np.mean(machine >= threshold)), "threshold": threshold, "calibration_fpr": calibration_fpr, "heldout_human_fpr": float(np.mean(human_eval >= threshold))}
            summary["results"][method][variant] = result
            row[f"{variant}_auroc"] = result["auroc"]; row[f"{variant}_tpr_at_1pct_fpr"] = result["tpr_at_1pct_fpr"]
        rows.append(row)
    atomic_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["| Detector | Ministral-3-8B Vanilla | Ministral-3-8B EASE |", "|---|---:|---:|"]
    for method in METHODS:
        cells = [f'{summary["results"][method][variant]["auroc"]:.4f} / {summary["results"][method][variant]["tpr_at_1pct_fpr"]:.4f}' for variant in ("vanilla", "ease")]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (out / "table_cross_detector_ministral.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("MINISTRAL DETECTOR SUMMARY SAVED")


def verify(args, out):
    summary = load_json(out / "summary.json")
    if summary["target_fpr"] != 0.01: raise AssertionError("wrong FPR scale")
    max_auc_error = max_tpr_error = 0.0
    for method in METHODS:
        name = key(method)
        human_cal = normalize_score_direction(method, np.load(out / "scores" / f"{name}_human_calibration.npy"))
        human_eval = normalize_score_direction(method, np.load(out / "scores" / f"{name}_human_evaluation.npy"))
        threshold, _ = threshold_at_fpr(human_cal, args.target_fpr)
        for variant in ("vanilla", "ease"):
            machine = normalize_score_direction(method, np.load(out / "scores" / f"{name}_{MODEL_KEY}_{variant}.npy")); reported = summary["results"][method][variant]
            max_auc_error = max(max_auc_error, abs(independent_auc(human_eval, machine) - reported["auroc"]))
            max_tpr_error = max(max_tpr_error, abs(float(np.mean(machine >= threshold)) - reported["tpr_at_1pct_fpr"]))
    if max(max_auc_error, max_tpr_error) > 1e-12: raise AssertionError("metric verification failed")
    atomic_json(out / "verification.json", {"status": "passed", "target_fpr": 0.01, "max_auroc_abs_error": max_auc_error, "max_tpr_abs_error": max_tpr_error})
    print("MINISTRAL DETECTOR VERIFICATION PASSED")


def self_test():
    assert 0.01 == 1 / 100 and 0.01 != 0.01 / 100
    threshold, fpr = threshold_at_fpr(np.arange(200, dtype=float), 0.01)
    assert fpr == 0.01 and threshold > 197
    assert len(METHODS) == 8
    assert np.array_equal(normalize_score_direction("entropy", np.array([1.0, 2.0])), np.array([-1.0, -2.0]))
    print("MINISTRAL SCORING SELF-TEST PASSED")


def main():
    args = parse_args(); out = Path(args.out_dir).expanduser().resolve()
    if args.stage == "self-test": self_test()
    elif args.stage == "score": score(args, out)
    elif args.stage == "finalize": finalize(args, out)
    else: verify(args, out)


if __name__ == "__main__":
    main()
