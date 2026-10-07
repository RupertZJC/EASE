"""Official-method scoring and metrics for the 200-sample cross-detector study."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from ease.config import CROSS_DETECTOR_CONFIG
from ease.metrics import auroc as fixed_auroc
from ease.metrics import calibrate_threshold_at_fpr


CORPORA = (
    "human_calibration", "human_evaluation",
    "qwen3_8b_vanilla", "qwen3_8b_ease",
    "llama3_8b_vanilla", "llama3_8b_ease",
)
METHODS = ("likelihood", "entropy", "logrank", "lrr", "npr", "dna_gpt", "detectgpt", "binoculars")
GPT_NEO = "EleutherAI/gpt-neo-2.7B"
DNA_MODELS = {
    "dna_qwen": "Qwen/Qwen3-8B",
    "dna_llama": "NousResearch/Meta-Llama-3-8B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("self-test", "score", "finalize", "verify"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--family", choices=("distribution", "perturb", "dna_qwen", "dna_llama", "binoculars"))
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


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def ranges(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def score_names(family):
    return {
        "distribution": ("likelihood", "entropy", "logrank", "lrr"),
        "perturb": ("detectgpt", "npr"),
        "dna_qwen": ("dna_gpt_qwen3_8b",),
        "dna_llama": ("dna_gpt_llama3_8b",),
        "binoculars": ("binoculars",),
    }[family]


def chunk_path(out_dir, score_name, corpus, start, end):
    return out_dir / "scores" / "chunks" / score_name / corpus / f"chunk_{start:05d}_{end:05d}.npz"


def valid_chunk(path, start, end):
    if not path.exists():
        return False
    with np.load(path) as archive:
        scores = np.asarray(archive["scores"], dtype=np.float64)
        indices = np.asarray(archive["indices"], dtype=np.int64)
    return (
        scores.shape == (end - start,)
        and np.isfinite(scores).all()
        and np.array_equal(indices, np.arange(start, end))
    )


class DistributionScorer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(GPT_NEO, padding_side="right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            GPT_NEO, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda").eval()

    @torch.inference_mode()
    def features_one(self, text):
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            return_token_type_ids=False,
        ).to("cuda")
        if encoded.input_ids.shape[1] < 2:
            prefixed = self.tokenizer.eos_token + text
            encoded = self.tokenizer(
                prefixed, return_tensors="pt", truncation=True, max_length=512,
                return_token_type_ids=False,
            ).to("cuda")
        logits = self.model(**encoded).logits[:, :-1].float()
        labels = encoded.input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        observed = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        likelihood = observed.mean()
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(-1).mean()
        observed_logits = logits.gather(-1, labels.unsqueeze(-1))
        ranks = (logits > observed_logits).sum(-1).float() + 1.0
        mean_logrank = torch.log(ranks).mean()
        nll = -likelihood
        lrr = nll / mean_logrank.clamp(min=1e-8)
        values = np.asarray([
            likelihood.item(), entropy.item(), -mean_logrank.item(), lrr.item()
        ], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite distribution score")
        return values

    def score(self, texts):
        values = np.stack([self.features_one(text) for text in texts])
        return {name: values[:, index] for index, name in enumerate(score_names("distribution"))}


class PerturbationScorer(DistributionScorer):
    def __init__(self, n_perturbations):
        super().__init__()
        self.n_perturbations = n_perturbations
        self.mask_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to("cuda").eval()

    @torch.inference_mode()
    def ll_logrank(self, text):
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            return_token_type_ids=False,
        ).to("cuda")
        if encoded.input_ids.shape[1] < 2:
            encoded = self.tokenizer(
                self.tokenizer.eos_token + text, return_tensors="pt",
                truncation=True, max_length=512, return_token_type_ids=False,
            ).to("cuda")
        logits = self.model(**encoded).logits[:, :-1].float()
        labels = encoded.input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        ll = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
        observed_logits = logits.gather(-1, labels.unsqueeze(-1))
        ranks = (logits > observed_logits).sum(-1).float() + 1.0
        return float(ll.item()), float(torch.log(ranks).mean().item())

    def score(self, texts, seeds):
        from ease.detectors.perturb import perturb_texts
        detectgpt, npr = [], []
        for text, seed in zip(texts, seeds):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            original_ll, original_logrank = self.ll_logrank(text)
            perturbed = perturb_texts(
                [text], self.mask_tokenizer, self.mask_model, "cuda",
                span_length=2, pct_words_masked=0.3, buffer_size=1,
                mask_top_p=1.0, n_perturbations=self.n_perturbations,
            )
            if len(perturbed) != self.n_perturbations or any(not item.strip() for item in perturbed):
                raise RuntimeError("perturbation count/content mismatch")
            features = [self.ll_logrank(item) for item in perturbed]
            pert_ll = np.asarray([item[0] for item in features], dtype=np.float64)
            pert_logrank = np.asarray([item[1] for item in features], dtype=np.float64)
            std = float(pert_ll.std())
            if std == 0.0:
                std = 1.0
            detectgpt.append((original_ll - float(pert_ll.mean())) / std)
            npr.append(float(pert_logrank.mean()) / max(original_logrank, 1e-8))
        return {
            "detectgpt": np.asarray(detectgpt, dtype=np.float64),
            "npr": np.asarray(npr, dtype=np.float64),
        }


class BinocularsScorer:
    def __init__(self):
        from ease.detectors.binoculars import Binoculars
        self.detector = Binoculars(max_token_observed=512)

    def score(self, texts):
        # Official Binoculars is lower for machine text; normalize direction.
        values = -np.asarray(self.detector.compute_score(texts), dtype=np.float64)
        return {"binoculars": values}


def family_scorer(args):
    if args.family == "distribution":
        return DistributionScorer()
    if args.family == "perturb":
        return PerturbationScorer(args.n_perturbations)
    if args.family in DNA_MODELS:
        from ease.detectors.dna_gpt import DNAGPT
        return DNAGPT(DNA_MODELS[args.family], regenerations=args.n_regenerations)
    if args.family == "binoculars":
        return BinocularsScorer()
    raise ValueError("unknown family")


def score(args, out_dir):
    if args.family is None or args.corpus is None:
        raise ValueError("--family and --corpus are required")
    if args.family == "dna_qwen" and args.corpus.startswith("llama"):
        raise ValueError("Qwen DNA-GPT must not score Llama corpora")
    if args.family == "dna_llama" and args.corpus.startswith("qwen"):
        raise ValueError("Llama DNA-GPT must not score Qwen corpora")
    count = args.n_calibration if args.corpus == "human_calibration" else args.n_eval
    assigned = [item for index, item in enumerate(ranges(count, args.chunk_size))
                if index % args.num_workers == args.worker_id]
    names = score_names(args.family)
    pending = [item for item in assigned if not all(
        valid_chunk(chunk_path(out_dir, name, args.corpus, *item), *item) for name in names
    )]
    if not pending:
        print("worker has no pending score chunks")
        return
    texts = load_json(out_dir / "corpora" / f"{args.corpus}.json")
    if len(texts) != count:
        raise RuntimeError("score corpus count mismatch")
    scorer = family_scorer(args)
    for start, end in pending:
        subset = texts[start:end]
        if args.family == "perturb":
            seeds = [args.seed + index * 1009 for index in range(start, end)]
            results = scorer.score(subset, seeds)
        elif args.family in DNA_MODELS:
            values = scorer.score(subset, seed=args.seed + start * 1009)
            results = {names[0]: values}
        else:
            results = scorer.score(subset)
        for name, values in results.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (end - start,) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid scores {name}/{args.corpus}/{start}:{end}")
            atomic_npz(
                chunk_path(out_dir, name, args.corpus, start, end),
                scores=values, indices=np.arange(start, end),
            )
        print(f"[{args.family}/{args.corpus}] {start}:{end}", flush=True)


def merge(out_dir, name, corpus, count, chunk_size):
    path = out_dir / "scores" / f"{name}_{corpus}.npy"
    if path.exists():
        values = np.load(path)
        if values.shape == (count,) and np.isfinite(values).all():
            return values
    values = np.full(count, np.nan)
    for start, end in ranges(count, chunk_size):
        chunk = chunk_path(out_dir, name, corpus, start, end)
        if not valid_chunk(chunk, start, end):
            raise RuntimeError(f"missing score chunk {chunk}")
        with np.load(chunk) as archive:
            values[start:end] = archive["scores"]
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite merged scores")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, values)
    return values


threshold_at_fpr = calibrate_threshold_at_fpr


def detector_key(method, model):
    return f"dna_gpt_{model}" if method == "dna_gpt" else method


def normalize_score_direction(method, values):
    """Normalize every detector so larger values mean more machine-like."""
    values = np.asarray(values, dtype=np.float64)
    return -values if method == "entropy" else values


def finalize(args, out_dir):
    summary = {
        "metric": "AUROC / TPR@1%FPR (lower is better for evasion)",
        "target_fpr": args.target_fpr,
        "score_direction": "higher_is_more_machine_like",
        "n_calibration": args.n_calibration,
        "n_evaluation": args.n_eval,
        "detector_settings": {
            "distribution_scoring_model": GPT_NEO,
            "detectgpt_npr_mask_model": "t5-small",
            "n_perturbations": args.n_perturbations,
            "dna_regenerations": args.n_regenerations,
            "binoculars_pair": ["tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"],
        },
        "results": {},
    }
    rows = []
    for method in METHODS:
        row = {"detector": method}
        summary["results"][method] = {}
        for model in ("qwen3_8b", "llama3_8b"):
            key = detector_key(method, model)
            human_cal = normalize_score_direction(
                method, merge(out_dir, key, "human_calibration", args.n_calibration, args.chunk_size)
            )
            human_eval = normalize_score_direction(
                method, merge(out_dir, key, "human_evaluation", args.n_eval, args.chunk_size)
            )
            threshold, calibration_fpr = threshold_at_fpr(human_cal, args.target_fpr)
            summary["results"][method][model] = {}
            for variant in ("vanilla", "ease"):
                machine = normalize_score_direction(
                    method, merge(out_dir, key, f"{model}_{variant}", args.n_eval, args.chunk_size)
                )
                labels = np.concatenate([np.zeros(args.n_eval), np.ones(args.n_eval)])
                scores = np.concatenate([human_eval, machine])
                result = {
                    "auroc": fixed_auroc(human_eval, machine),
                    "tpr_at_1pct_fpr": float(np.mean(machine >= threshold)),
                    "threshold": threshold,
                    "calibration_fpr": calibration_fpr,
                    "heldout_human_fpr": float(np.mean(human_eval >= threshold)),
                }
                summary["results"][method][model][variant] = result
                row[f"{model}_{variant}_auroc"] = result["auroc"]
                row[f"{model}_{variant}_tpr_at_1pct_fpr"] = result["tpr_at_1pct_fpr"]
        rows.append(row)
    atomic_json(out_dir / "summary.json", summary)
    fields = list(rows[0])
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| Detector | Qwen3-8B Vanilla | Qwen3-8B EASE | Llama3-8B Vanilla | Llama3-8B EASE |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = []
        for model in ("qwen3_8b", "llama3_8b"):
            for variant in ("vanilla", "ease"):
                item = summary["results"][method][model][variant]
                cells.append(f'{item["auroc"]:.4f} / {item["tpr_at_1pct_fpr"]:.4f}')
        table.append(f"| {method} | " + " | ".join(cells) + " |")
    (out_dir / "table_cross_detector.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("CROSS DETECTOR SUMMARY SAVED")


def independent_auc(human, machine):
    comparisons = (machine[:, None] > human[None, :]).mean()
    ties = (machine[:, None] == human[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def verify(args, out_dir):
    summary = load_json(out_dir / "summary.json")
    if summary["target_fpr"] != 0.01:
        raise AssertionError("wrong FPR scale")
    max_auc_error = max_tpr_error = 0.0
    for method in METHODS:
        for model in ("qwen3_8b", "llama3_8b"):
            key = detector_key(method, model)
            human_cal = normalize_score_direction(
                method, np.load(out_dir / "scores" / f"{key}_human_calibration.npy")
            )
            human_eval = normalize_score_direction(
                method, np.load(out_dir / "scores" / f"{key}_human_evaluation.npy")
            )
            threshold, _ = threshold_at_fpr(human_cal, args.target_fpr)
            for variant in ("vanilla", "ease"):
                machine = normalize_score_direction(
                    method, np.load(out_dir / "scores" / f"{key}_{model}_{variant}.npy")
                )
                reported = summary["results"][method][model][variant]
                max_auc_error = max(max_auc_error, abs(independent_auc(human_eval, machine) - reported["auroc"]))
                max_tpr_error = max(max_tpr_error, abs(float(np.mean(machine >= threshold)) - reported["tpr_at_1pct_fpr"]))
    if max(max_auc_error, max_tpr_error) > 1e-12:
        raise AssertionError("cross-detector metric verification failed")
    atomic_json(out_dir / "verification.json", {
        "status": "passed", "target_fpr": 0.01,
        "max_auroc_abs_error": max_auc_error,
        "max_tpr_abs_error": max_tpr_error,
    })
    print("CROSS DETECTOR VERIFICATION PASSED")


def self_test():
    assert 0.01 == 1 / 100 and 0.01 != 0.01 / 100
    threshold, fpr = threshold_at_fpr(np.arange(200, dtype=float), 0.01)
    assert fpr == 0.01 and threshold > 197
    assert independent_auc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    assert np.array_equal(
        normalize_score_direction("entropy", np.array([1.0, 2.0])),
        np.array([-1.0, -2.0]),
    )
    assert len(METHODS) == 8
    print("CROSS DETECTOR SELF-TEST PASSED")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "score":
        score(args, out_dir)
    elif args.stage == "finalize":
        finalize(args, out_dir)
    else:
        verify(args, out_dir)


if __name__ == "__main__":
    main()
