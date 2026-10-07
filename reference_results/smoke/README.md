# Ten-example smoke record (2026-10-07)

These are pipeline-validation outputs, not the paper's reported results.
Generation uses Qwen3-8B; the detector is RoBERTa-base. Each machine corpus
contains ten final outputs. Human calibration and evaluation use ten disjoint
records each.

- `environment.json`: library/checkpoint versions, source and artifact hashes.
- `corpora/`: generated text and metadata, plus the two human sets.
- `scores/`: the five raw ten-element score arrays.
- `summary.csv` / `summary.json`: calibrated thresholds and comparison metrics.
- `verification.json` / `corpus_verification.json`: independent checks.
- `run_config.json`: the generation profile.

From the repository root, with the dependencies installed and `PYTHONPATH` set
as described in the main README:

```bash
python -m experiments.table1.score --stage verify --out-dir reference_results/smoke --detector roberta_base --corpora np ease_plugin ease_rewrite_d2 --smoke
```

To repeat generation and scoring, use `python scripts/run_smoke.py` with a fresh
output directory. The fixed seed, batch size of 1, and recorded model revisions
identify the tested run. Different devices or library versions can change
sampled outputs. With only ten calibration texts, the conservative threshold
allows no calibration false positives; this is too small to characterize a 1%
FPR operating point statistically.
