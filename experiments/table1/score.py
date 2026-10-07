"""Score the Qwen3-8B/EASE/AP corpora with five detector baselines.

Each detector emits chunked raw scores.  Final TPR@1%FPR uses one detector-
specific threshold calibrated on the disjoint human calibration corpus and
holds that threshold fixed across every method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
)

from ease.config import TABLE1_CONFIG
from ease.metrics import auroc as fixed_auroc
from ease.metrics import calibrate_threshold_at_fpr


DETECTORS = ("roberta_base", "roberta_large", "mage", "radar", "fast_detectgpt")
MACHINE_CORPORA = (
    "np", "simple", "recursive", "adv_base", "adv_large", "ease_plugin",
    "ease_rewrite_d2",
)
ALL_CORPORA = ("human_calibration", "human_evaluation") + MACHINE_CORPORA
DISPLAY = {
    "np": "Qwen3-8B",
    "simple": "Simple Paraphrase",
    "recursive": "Rec. Para. 2",
    "adv_base": "AdvPara (RoBERTa-B)",
    "adv_large": "AdvPara (RoBERTa-L)",
    "ease_plugin": "EASE-plugin (delta=2)",
    "ease_rewrite_d2": "EASE-Rewrite (delta=2)",
}
MODEL_IDS = {
    "roberta_base": "openai-community/roberta-base-openai-detector",
    "roberta_large": "openai-community/roberta-large-openai-detector",
    "mage": "yaful/MAGE",
    "radar": "TrustSafeAI/RADAR-Vicuna-7B",
    "fast_sampling": "EleutherAI/gpt-j-6B",
    "fast_scoring": "EleutherAI/gpt-neo-2.7B",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", required=True, choices=("self-test", "score", "finalize", "verify"))
    parser.add_argument("--detector", choices=DETECTORS)
    parser.add_argument("--corpora", nargs="+", choices=MACHINE_CORPORA,
                        help="evaluate a subset; human sets are always included")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.target_fpr = TABLE1_CONFIG["target_fpr"]
    args.n_calibration = 10 if args.smoke else TABLE1_CONFIG["n_calibration"]
    args.n_eval = 10 if args.smoke else TABLE1_CONFIG["n_evaluation"]
    return args


def selected_methods(args):
    return tuple(dict.fromkeys(args.corpora)) if args.corpora else MACHINE_CORPORA


def selected_detectors(args):
    return (args.detector,) if args.detector else DETECTORS


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_npy(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float64))
    os.replace(temporary, path)


def atomic_npz(path: Path, scores, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            scores=np.asarray(scores, dtype=np.float64),
            indices=np.asarray(indices, dtype=np.int64),
        )
    os.replace(temporary, path)


def corpus_path(out_dir: Path, corpus: str):
    return out_dir / "corpora" / f"{corpus}.json"


def score_chunk_path(out_dir: Path, detector: str, corpus: str, start: int, end: int):
    return out_dir / "scores" / "chunks" / detector / corpus / f"chunk_{start:05d}_{end:05d}.npz"


def chunk_ranges(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def valid_score_chunk(path: Path, start: int, end: int):
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


def merged_score_path(out_dir: Path, detector: str, corpus: str):
    return out_dir / "scores" / f"{detector}_{corpus}.npy"


def valid_merged_scores(path: Path, count: int):
    if not path.exists():
        return False
    values = np.load(path)
    return values.shape == (count,) and np.isfinite(values).all()


class ClassifierDetector:
    def __init__(self, detector: str, max_length: int):
        self.detector = detector
        self.max_length = max_length
        self.preprocess = None
        model_id = MODEL_IDS[detector]
        mage_revision = "refs/pr/2"
        config = None
        if detector == "mage":
            from ease.detectors.mage_preprocess import preprocess
            self.preprocess = preprocess
            dtype = torch.float32
            # The published MAGE config stores numeric id2label values.  Newer
            # Transformers releases validate these as strings before either the
            # tokenizer or model can load, so normalize only this malformed
            # metadata while leaving the checkpoint architecture untouched.
            config_dict, _ = PretrainedConfig.get_config_dict(model_id, revision=mage_revision)
            model_type = config_dict.pop("model_type")
            if "id2label" in config_dict:
                config_dict["id2label"] = {
                    int(key): str(value) for key, value in config_dict["id2label"].items()
                }
            if "label2id" in config_dict:
                config_dict["label2id"] = {
                    str(key): int(value) for key, value in config_dict["label2id"].items()
                }
            config = AutoConfig.for_model(model_type, **config_dict)
        elif detector == "radar":
            dtype = torch.float16
        else:
            dtype = None
        tokenizer_kwargs = {"config": config} if config is not None else {}
        if detector == "mage":
            tokenizer_kwargs["revision"] = mage_revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        kwargs = {"low_cpu_mem_usage": True}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if config is not None:
            kwargs["config"] = config
        if detector == "mage":
            # The upstream main branch contains only pytorch_model.bin.  Use
            # Hugging Face's weight-equivalent safetensors conversion PR so
            # torch<2.6 never needs to deserialize the pickle checkpoint.
            kwargs["revision"] = mage_revision
            kwargs["use_safetensors"] = True
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs).to("cuda").eval()
        if self.model.config.num_labels != 2:
            raise RuntimeError(f"{detector} must expose exactly two output classes")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def score(self, texts, batch_size):
        if self.preprocess is not None:
            texts = [self.preprocess(text) for text in texts]
        values = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to("cuda")
            logits = self.model(**encoded).logits.float()
            values.extend(F.softmax(logits, dim=-1)[:, 0].cpu().tolist())
        return np.asarray(values, dtype=np.float64)


class FastDetectGPT:
    """Official black-box ordering: GPT-J sampling, GPT-Neo scoring."""

    def __init__(self, max_length):
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Fast-DetectGPT requires at least one CUDA GPU")
        self.max_length = max_length
        self.sampling_device = "cuda:0"
        if device_count >= 2:
            self.scoring_device = "cuda:1"
        else:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory < 40 * 1024**3:
                raise RuntimeError(
                    "Fast-DetectGPT needs two GPUs, or one GPU with at least 40 GiB"
                )
            # GPT-J-6B and GPT-Neo-2.7B fit together on one 48 GiB A6000.
            self.scoring_device = "cuda:0"
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_IDS["fast_scoring"], padding_side="right"
        )
        self.sampling_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_IDS["fast_sampling"], padding_side="right"
        )
        for tokenizer in (self.scoring_tokenizer, self.sampling_tokenizer):
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        self.sampling_model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS["fast_sampling"], torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(self.sampling_device).eval()
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            MODEL_IDS["fast_scoring"], torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(self.scoring_device).eval()

    @torch.inference_mode()
    def score(self, texts, batch_size):
        values = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            score_inputs = self.scoring_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                                return_token_type_ids=False,
            )
            # AdvPara can legitimately emit a one-token non-empty string.  The
            # discrepancy statistic predicts token t from tokens <t, so such a
            # string otherwise has zero scored positions and produces 0/0.
            # Give only those edge cases the standard GPT end-of-text context.
            short = score_inputs.attention_mask.sum(dim=1) < 2
            if bool(short.any()):
                batch = [
                    (self.scoring_tokenizer.eos_token + text) if bool(is_short) else text
                    for text, is_short in zip(batch, short.tolist())
                ]
                score_inputs = self.scoring_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            sampling_inputs = self.sampling_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            if not torch.equal(score_inputs.input_ids, sampling_inputs.input_ids):
                raise RuntimeError("GPT-J and GPT-Neo tokenizers produced different token ids")
            labels = score_inputs.input_ids[:, 1:].to(self.scoring_device)
            mask = score_inputs.attention_mask[:, 1:].to(self.scoring_device).float()
            score_gpu = score_inputs.to(self.scoring_device)
            sampling_gpu = sampling_inputs.to(self.sampling_device)
            logits_score = self.scoring_model(**score_gpu).logits[:, :-1].float()
            logits_sampling = self.sampling_model(**sampling_gpu).logits[:, :-1].to(
                self.scoring_device, dtype=torch.float32
            )
            vocab = min(logits_score.size(-1), logits_sampling.size(-1))
            if int(labels.max()) >= vocab:
                raise RuntimeError("observed token is outside the shared Fast-DetectGPT vocabulary")
            logits_score = logits_score[..., :vocab]
            logits_sampling = logits_sampling[..., :vocab]
            log_probs_score = F.log_softmax(logits_score, dim=-1)
            probabilities_sampling = F.softmax(logits_sampling, dim=-1)
            observed = log_probs_score.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mean = (probabilities_sampling * log_probs_score).sum(-1)
            variance = (
                (probabilities_sampling * log_probs_score.square()).sum(-1) - mean.square()
            ).clamp(min=1e-12)
            numerator = ((observed - mean) * mask).sum(-1)
            denominator = (variance * mask).sum(-1).sqrt()
            batch_scores = numerator / denominator
            if not torch.isfinite(batch_scores).all():
                raise RuntimeError("non-finite Fast-DetectGPT score")
            values.extend(batch_scores.cpu().tolist())
            del logits_score, logits_sampling, log_probs_score, probabilities_sampling
        return np.asarray(values, dtype=np.float64)


def load_detector(name, max_length):
    if name == "fast_detectgpt":
        return FastDetectGPT(max_length)
    return ClassifierDetector(name, max_length)


def score(args, out_dir: Path):
    if args.detector is None:
        raise ValueError("--detector is required")
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0,num-workers)")
    corpora = ("human_calibration", "human_evaluation") + selected_methods(args)
    counts = {
        "human_calibration": args.n_calibration,
        **{name: args.n_eval for name in corpora if name != "human_calibration"},
    }
    pending = []
    for corpus in corpora:
        if valid_merged_scores(
            merged_score_path(out_dir, args.detector, corpus), counts[corpus]
        ):
            print(f"[{args.detector}] reusing merged scores for {corpus}")
            continue
        ranges = chunk_ranges(counts[corpus], args.chunk_size)
        for chunk_index, (start, end) in enumerate(ranges):
            if chunk_index % args.num_workers != args.worker_id:
                continue
            path = score_chunk_path(out_dir, args.detector, corpus, start, end)
            if not valid_score_chunk(path, start, end):
                pending.append((corpus, start, end, path))
    if not pending:
        print(f"[{args.detector}] worker has no incomplete chunks")
        return
    detector = load_detector(args.detector, args.max_length)
    cache = {}
    for corpus, start, end, path in pending:
        if corpus not in cache:
            values = load_json(corpus_path(out_dir, corpus))
            if len(values) != counts[corpus] or any(not text.strip() for text in values):
                raise RuntimeError(f"invalid corpus: {corpus}")
            cache[corpus] = values
        batch_size = args.batch_size
        if args.detector == "fast_detectgpt":
            batch_size = min(batch_size, 2)
        scores = detector.score(cache[corpus][start:end], batch_size)
        if scores.shape != (end - start,) or not np.isfinite(scores).all():
            raise RuntimeError(f"invalid {args.detector} scores for {corpus} {start}:{end}")
        atomic_npz(path, scores, np.arange(start, end))
        print(f"[{args.detector}] saved {corpus} {start}:{end}", flush=True)


def merge_scores(out_dir: Path, detector: str, corpus: str, count: int, chunk_size: int):
    merged_path = merged_score_path(out_dir, detector, corpus)
    if valid_merged_scores(merged_path, count):
        return np.asarray(np.load(merged_path), dtype=np.float64)
    scores = np.full(count, np.nan, dtype=np.float64)
    seen = np.zeros(count, dtype=bool)
    for start, end in chunk_ranges(count, chunk_size):
        path = score_chunk_path(out_dir, detector, corpus, start, end)
        if not valid_score_chunk(path, start, end):
            raise RuntimeError(f"missing score chunk: {path}")
        with np.load(path) as archive:
            indices = np.asarray(archive["indices"], dtype=np.int64)
            values = np.asarray(archive["scores"], dtype=np.float64)
        if seen[indices].any():
            raise RuntimeError(f"duplicate score indices: {detector}/{corpus}")
        scores[indices] = values
        seen[indices] = True
    if not seen.all() or not np.isfinite(scores).all():
        raise RuntimeError(f"incomplete scores: {detector}/{corpus}")
    atomic_npy(merged_path, scores)
    return scores


def finalize(args, out_dir: Path):
    methods = selected_methods(args)
    detectors = selected_detectors(args)
    summary = {
        "metric": "AUROC and fixed-threshold TPR@1%FPR",
        "target_fpr": args.target_fpr,
        "score_direction": "higher_is_more_machine_like",
        "smoke": args.smoke,
        "detectors": list(detectors),
        "models": MODEL_IDS,
        "thresholds": {},
        "methods": {},
    }
    rows = {method: {"method": method, "display_name": DISPLAY[method]} for method in methods}
    for detector in detectors:
        merged = {}
        for corpus in ("human_calibration", "human_evaluation") + methods:
            count = args.n_calibration if corpus == "human_calibration" else args.n_eval
            merged[corpus] = merge_scores(out_dir, detector, corpus, count, args.chunk_size)
        threshold, calibration_fpr = calibrate_threshold_at_fpr(
            merged["human_calibration"], args.target_fpr
        )
        heldout_fpr = float(np.mean(merged["human_evaluation"] >= threshold))
        summary["thresholds"][detector] = {
            "threshold": threshold,
            "target_fpr": args.target_fpr,
            "calibration_fpr": calibration_fpr,
            "heldout_human_fpr": heldout_fpr,
        }
        for method in methods:
            human = merged["human_evaluation"]
            machine = merged[method]
            result = {
                "auroc": fixed_auroc(human, machine),
                "tpr_at_1pct_fpr": float(np.mean(machine >= threshold)),
                "n_human": int(human.size),
                "n_machine": int(machine.size),
            }
            summary["methods"].setdefault(method, {"display_name": DISPLAY[method], "detectors": {}})
            summary["methods"][method]["detectors"][detector] = result
            rows[method][f"{detector}_auroc"] = result["auroc"]
            rows[method][f"{detector}_tpr_at_1pct_fpr"] = result["tpr_at_1pct_fpr"]
    atomic_json(out_dir / "summary.json", summary)
    fieldnames = list(rows[methods[0]])
    path = out_dir / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[method] for method in methods)
    os.replace(temporary, path)
    print(f"[finalize] saved {out_dir / 'summary.json'} and {path}")


def independent_threshold(human, target_fpr):
    ordered = np.sort(np.asarray(human, dtype=np.float64))
    index = math.ceil((1.0 - target_fpr) * (ordered.size - 1))
    threshold = float(ordered[index])
    fpr = float(np.mean(ordered >= threshold))
    if fpr > target_fpr:
        threshold = float(np.nextafter(threshold, np.inf))
        fpr = float(np.mean(ordered >= threshold))
    return threshold, fpr


def independent_auroc(human, machine):
    human = np.asarray(human, dtype=np.float64)
    machine = np.asarray(machine, dtype=np.float64)
    values = np.concatenate([human, machine])
    labels = np.concatenate([np.zeros(human.size, dtype=np.int8), np.ones(machine.size, dtype=np.int8)])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - machine.size * (machine.size + 1) / 2.0) / (machine.size * human.size)
    )


def verify(args, out_dir: Path):
    summary = load_json(out_dir / "summary.json")
    methods = selected_methods(args)
    detectors = selected_detectors(args)
    if set(summary["methods"]) != set(methods) or set(summary["thresholds"]) != set(detectors):
        raise AssertionError("summary scope does not match requested methods/detectors")
    if summary["target_fpr"] != 0.01:
        raise AssertionError("summary is not configured for 1% FPR")
    max_auc_error = max_tpr_error = max_threshold_error = 0.0
    for detector in detectors:
        calibration = np.load(out_dir / "scores" / f"{detector}_human_calibration.npy")
        human = np.load(out_dir / "scores" / f"{detector}_human_evaluation.npy")
        for values, count in ((calibration, args.n_calibration), (human, args.n_eval)):
            if values.shape != (count,) or not np.isfinite(values).all():
                raise AssertionError(f"invalid human scores for {detector}")
        threshold, fpr = independent_threshold(calibration, args.target_fpr)
        reported_threshold = summary["thresholds"][detector]
        if not np.isfinite(reported_threshold["threshold"]) or fpr > args.target_fpr:
            raise AssertionError(f"invalid calibrated threshold for {detector}")
        max_threshold_error = max(max_threshold_error, abs(threshold - reported_threshold["threshold"]))
        if abs(fpr - reported_threshold["calibration_fpr"]) > 1e-15:
            raise AssertionError(f"calibration FPR mismatch for {detector}")
        if abs(float(np.mean(human >= threshold)) - reported_threshold["heldout_human_fpr"]) > 1e-15:
            raise AssertionError(f"heldout FPR mismatch for {detector}")
        for method in methods:
            machine = np.load(out_dir / "scores" / f"{detector}_{method}.npy")
            if machine.shape != (args.n_eval,) or not np.isfinite(machine).all():
                raise AssertionError(f"invalid machine scores for {detector}/{method}")
            expected_auc = independent_auroc(human, machine)
            expected_tpr = float(np.mean(machine >= threshold))
            reported = summary["methods"][method]["detectors"][detector]
            if not all(np.isfinite(reported[key]) for key in ("auroc", "tpr_at_1pct_fpr")):
                raise AssertionError(f"non-finite summary metric for {detector}/{method}")
            max_auc_error = max(max_auc_error, abs(expected_auc - reported["auroc"]))
            max_tpr_error = max(max_tpr_error, abs(expected_tpr - reported["tpr_at_1pct_fpr"]))
    if max(max_auc_error, max_tpr_error, max_threshold_error) > 1e-12:
        raise AssertionError("independent metric verification failed")
    result = {
        "status": "passed",
        "detectors": list(detectors),
        "methods": list(methods),
        "comparisons": len(detectors) * len(methods),
        "target_fpr": args.target_fpr,
        "max_auroc_abs_error": max_auc_error,
        "max_tpr_abs_error": max_tpr_error,
        "max_threshold_abs_error": max_threshold_error,
    }
    atomic_json(out_dir / "verification.json", result)
    print(json.dumps(result, indent=2))


def self_test():
    assert 0.01 == 1.0 / 100.0 and 0.01 != 0.01 / 100.0
    human = np.arange(200, dtype=np.float64)
    threshold, fpr = calibrate_threshold_at_fpr(human, 0.01)
    independent, independent_fpr = independent_threshold(human, 0.01)
    assert threshold == independent and fpr == independent_fpr == 0.01
    assert independent_auroc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert independent_auroc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert len(DETECTORS) == 5 and len(MACHINE_CORPORA) == 7
    print("SELF-TEST PASSED: 1% FPR, threshold, AUROC, detector/method roster")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.n_calibration < (1 if args.smoke else 100) or args.n_eval < 1:
        raise ValueError("invalid calibration/evaluation size")
    if args.stage == "self-test":
        self_test()
    elif args.stage == "score":
        score(args, out_dir)
    elif args.stage == "finalize":
        finalize(args, out_dir)
    elif args.stage == "verify":
        verify(args, out_dir)


if __name__ == "__main__":
    main()
