"""Frozen scientific configurations used by the released experiments.

Only operational settings such as output paths, devices, batch sizes and
worker counts are intentionally left to the command-line runners.
"""

from copy import deepcopy


TABLE1_CONFIG = {
    "schema_version": 1,
    "generator": "Qwen/Qwen3-8B",
    "dataset": "Salesforce/wikitext:wikitext-103-raw-v1:train",
    "n_calibration": 2000,
    "n_evaluation": 2000,
    "prompt_tokens": 30,
    "generated_tokens": 200,
    "target_fpr": 0.01,
    "seed_generation": 20260909,
    "seed_rewrite": 20260912,
    "profiles": {
        "np": {"kind": "generation", "temperature": 1.0, "top_k": 20, "delta": 0.0},
        "simple": {
            "kind": "plain_paraphrase", "passes": 1, "temperature": 0.6,
            "top_p": 0.99, "top_k": 50,
        },
        "recursive": {
            "kind": "plain_paraphrase", "passes": 2, "temperature": 0.6,
            "top_p": 0.99, "top_k": 50,
        },
        "adv_base": {
            "kind": "adversarial_paraphrase", "proxy": "roberta_base",
            "temperature": 1.0, "top_p": 0.99, "top_k": 50,
            "ranking": "language_model_probability_plus_proxy_score",
            "deterministic": True,
        },
        "adv_large": {
            "kind": "adversarial_paraphrase", "proxy": "roberta_large",
            "temperature": 1.0, "top_p": 0.99, "top_k": 50,
            "ranking": "language_model_probability_plus_proxy_score",
            "deterministic": True,
        },
        "ease_plugin": {"kind": "generation", "temperature": 1.0, "top_k": 20, "delta": 2.0},
        "ease_rewrite": {
            "kind": "ease_paraphrase", "passes": 1,
            "temperature": 1.0, "top_k": 20, "delta": 2.0,
        },
    },
    "detectors": {
        "roberta_base": "openai-community/roberta-base-openai-detector",
        "roberta_large": "openai-community/roberta-large-openai-detector",
        "mage": "yaful/MAGE",
        "radar": "TrustSafeAI/RADAR-Vicuna-7B",
        "fast_detectgpt_sampling": "EleutherAI/gpt-j-6B",
        "fast_detectgpt_scoring": "EleutherAI/gpt-neo-2.7B",
    },
}


CROSS_DETECTOR_CONFIG = {
    "schema_version": 1,
    "source_models": {
        "qwen3_8b": "Qwen/Qwen3-8B",
        "llama3_8b": "NousResearch/Meta-Llama-3-8B-Instruct",
        "ministral3_8b": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    },
    "n_calibration": 200,
    "n_evaluation": 200,
    "prompt_tokens": 30,
    "generated_tokens": 200,
    "temperature": 1.0,
    "top_k": 20,
    "delta_vanilla": 0.0,
    "delta_ease": 2.0,
    "target_fpr": 0.01,
    "n_perturbations": 10,
    "n_regenerations": 10,
    "seed": 20260910,
    "detectors": [
        "likelihood", "entropy", "logrank", "lrr", "npr",
        "dna_gpt", "detectgpt", "binoculars",
    ],
}


def table1_config():
    return deepcopy(TABLE1_CONFIG)


def cross_detector_config():
    return deepcopy(CROSS_DETECTOR_CONFIG)
