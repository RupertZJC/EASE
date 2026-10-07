#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

TABLE1_DIR="${1:-runs/table1}"
OUT_DIR="${2:-runs/cross-detector}"

python -m experiments.cross_detector.generate --stage prepare --out-dir "$OUT_DIR" --source-dir "$TABLE1_DIR"
for method in vanilla ease; do
  python -m experiments.cross_detector.generate --stage generate --out-dir "$OUT_DIR" --model llama3_8b --method "$method"
  python -m experiments.cross_detector.generate --stage merge --out-dir "$OUT_DIR" --model llama3_8b --method "$method"
done
python -m experiments.cross_detector.generate --stage validate --out-dir "$OUT_DIR"

all_corpora=(human_calibration human_evaluation qwen3_8b_vanilla qwen3_8b_ease llama3_8b_vanilla llama3_8b_ease)
for family in distribution perturb binoculars; do
  for corpus in "${all_corpora[@]}"; do
    python -m experiments.cross_detector.score --stage score --out-dir "$OUT_DIR" --family "$family" --corpus "$corpus"
  done
done

for family in dna_qwen dna_llama; do
  model_prefix="qwen3_8b"
  [[ "$family" == "dna_llama" ]] && model_prefix="llama3_8b"
  for corpus in human_calibration human_evaluation "${model_prefix}_vanilla" "${model_prefix}_ease"; do
    python -m experiments.cross_detector.score --stage score --out-dir "$OUT_DIR" --family "$family" --corpus "$corpus"
  done
done

python -m experiments.cross_detector.score --stage finalize --out-dir "$OUT_DIR"
python -m experiments.cross_detector.score --stage verify --out-dir "$OUT_DIR"
