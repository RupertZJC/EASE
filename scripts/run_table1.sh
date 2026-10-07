#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

OUT_DIR="${1:-runs/table1}"

python -m experiments.table1.generate --stage prepare --out-dir "$OUT_DIR"
for method in np ease_plugin simple recursive adv_base adv_large; do
  python -m experiments.table1.generate --stage generate --out-dir "$OUT_DIR" --corpus "$method"
  python -m experiments.table1.generate --stage merge --out-dir "$OUT_DIR" --corpus "$method"
done

python -m experiments.table1.generate_rewrite --stage prepare --out-dir "$OUT_DIR" --source-dir "$OUT_DIR"
python -m experiments.table1.generate_rewrite --stage generate --out-dir "$OUT_DIR"
python -m experiments.table1.generate_rewrite --stage merge --out-dir "$OUT_DIR"
python -m experiments.table1.generate_rewrite --stage validate --out-dir "$OUT_DIR"
python -m experiments.table1.generate --stage validate --out-dir "$OUT_DIR"

for detector in roberta_base roberta_large mage radar fast_detectgpt; do
  python -m experiments.table1.score --stage score --out-dir "$OUT_DIR" --detector "$detector"
done
python -m experiments.table1.score --stage finalize --out-dir "$OUT_DIR"
python -m experiments.table1.score --stage verify --out-dir "$OUT_DIR"

for corpus in np simple recursive adv_base adv_large ease_plugin ease_rewrite_d2; do
  python scripts/evaluate_ppl.py \
    --input "$OUT_DIR/corpora/$corpus.json" \
    --output "$OUT_DIR/ppl/$corpus.json"
done
