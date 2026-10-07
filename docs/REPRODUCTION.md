# Reproduction guide

Run commands from the repository root. Scientific parameters are read from
`src/ease/config.py` and cannot be overridden from the command line. This
repository is intentionally not packaged; expose the source directory first:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Ten-example GPU check

```bash
python scripts/run_smoke.py --out-dir runs/smoke
```

This runs NP, EASE-plugin, EASE-Rewrite, and RoBERTa-base using the bundled
ten-example split. The batch size defaults to 1; set `--batch-size` only to
change execution throughput. Keep batch/chunk sizes unchanged when resuming a
run, since they affect the random sampling stream. Use a fresh output directory
for a different execution layout.

For a subset of a larger run, the Table-1 scorer accepts `--corpora` and
`--detector` in its `score`, `finalize`, and `verify` stages. Pass the same
selection to all three. Human calibration and evaluation are always included.
Omitting the selection retains the complete paper comparison.

## Table 1 corpora

```bash
OUT=runs/table1

python -m experiments.table1.generate --stage prepare --out-dir "$OUT"

for METHOD in np ease_plugin simple recursive adv_base adv_large; do
  python -m experiments.table1.generate --stage generate --out-dir "$OUT" --corpus "$METHOD"
  python -m experiments.table1.generate --stage merge --out-dir "$OUT" --corpus "$METHOD"
done

python -m experiments.table1.generate_rewrite --stage prepare --out-dir "$OUT" --source-dir "$OUT"
python -m experiments.table1.generate_rewrite --stage generate --out-dir "$OUT"
python -m experiments.table1.generate_rewrite --stage merge --out-dir "$OUT"
python -m experiments.table1.generate_rewrite --stage validate --out-dir "$OUT"
python -m experiments.table1.generate --stage validate --out-dir "$OUT"
```

The order matters: NP precedes all paraphrase methods, and Simple Paraphrase
precedes Recursive Paraphrase 2. Jobs may be sharded with `--worker-id` and
`--num-workers`; merge only after every shard has completed.

## Table 1 detectors

```bash
for DETECTOR in roberta_base roberta_large mage radar fast_detectgpt; do
  python -m experiments.table1.score --stage score --out-dir "$OUT" --detector "$DETECTOR"
done
python -m experiments.table1.score --stage finalize --out-dir "$OUT"
python -m experiments.table1.score --stage verify --out-dir "$OUT"
```

Fast-DetectGPT uses GPT-J-6B for sampling and GPT-Neo-2.7B for scoring. Its
runner requires two ordinary GPUs or one GPU with at least 40 GiB.

## Cross-detector generation

The Qwen corpus is reused from Table 1. Llama uses exactly the same prompt text
and source ordering.

```bash
CROSS=runs/cross-detector
python -m experiments.cross_detector.generate --stage prepare --out-dir "$CROSS" --source-dir "$OUT"

for MODEL in llama3_8b; do
  for METHOD in vanilla ease; do
    python -m experiments.cross_detector.generate --stage generate --out-dir "$CROSS" --model "$MODEL" --method "$METHOD"
    python -m experiments.cross_detector.generate --stage merge --out-dir "$CROSS" --model "$MODEL" --method "$METHOD"
  done
done
python -m experiments.cross_detector.generate --stage validate --out-dir "$CROSS"
```

Score each corpus with the distribution, perturbation and Binoculars families.
DNA-GPT uses the matching source LLM (`dna_qwen` or `dna_llama`). The scorer
supports chunking and worker sharding; finalize only after all corpus/family
jobs have completed.

Ministral is generated and scored by `generate_ministral.py` and
`score_ministral.py`; it reuses the same human sets and prompt ordering from
the two-model cross-detector directory.

```bash
MINISTRAL=runs/ministral
python -m experiments.cross_detector.generate_ministral --stage prepare --out-dir "$MINISTRAL" --base-dir "$CROSS"
for METHOD in vanilla ease; do
  python -m experiments.cross_detector.generate_ministral --stage generate --out-dir "$MINISTRAL" --method "$METHOD"
  python -m experiments.cross_detector.generate_ministral --stage merge --out-dir "$MINISTRAL" --method "$METHOD"
done
python -m experiments.cross_detector.generate_ministral --stage validate --out-dir "$MINISTRAL"
for FAMILY in distribution perturb dna_ministral binoculars; do
  for CORPUS in human_calibration human_evaluation ministral3_8b_vanilla ministral3_8b_ease; do
    python -m experiments.cross_detector.score_ministral --stage score --out-dir "$MINISTRAL" --family "$FAMILY" --corpus "$CORPUS"
  done
done
python -m experiments.cross_detector.score_ministral --stage finalize --out-dir "$MINISTRAL"
python -m experiments.cross_detector.score_ministral --stage verify --out-dir "$MINISTRAL"
```

## Perplexity

```bash
python scripts/evaluate_ppl.py --input "$OUT/corpora/ease_rewrite_d2.json" --output "$OUT/ppl/ease_rewrite_d2.json"
```

PPL is the arithmetic mean of per-text perplexities under Qwen3-8B, computed
on final output text only.

## Integrity checks

Each experiment writes its complete frozen configuration, corpus metadata and
raw score arrays. `finalize` creates the table CSV/JSON; `verify` independently
recomputes thresholds, AUROC and TPR from those arrays. A run is complete only
after verification passes.
