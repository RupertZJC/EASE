# Validation

Validated on 2026-10-07 with Python 3.10.20, PyTorch 2.6.0 + CUDA 12.4,
Transformers 5.8.1, and one RTX 3090. Core dependency versions are recorded in
`constraints-smoke.txt`.

## Checks performed

- Source/configuration audit, including bundled split counts, alignment, and
  calibration/evaluation disjointness: passed.
- Nine unit tests, including EASE sampling properties, score-chunk merging,
  and rejection of an inconsistent summary: passed, with no skips.
- All seven experiment entry-point self-tests: passed.
- Real Qwen3-8B generation: ten NP and ten EASE-plugin continuations, each
  generated for 200 tokens, plus ten EASE-Rewrite outputs of at most 200 tokens.
- RoBERTa-base scoring: 50 finite scores covering the two ten-text human sets
  and three ten-text machine corpora.
- Independent verification of all three comparisons: AUROC, TPR, and threshold
  absolute errors were zero.
- Re-running the final release code reused completed chunks; SHA-256 hashes
  of all corpus files and raw score arrays remained unchanged.

The [smoke record](../reference_results/smoke/) includes environment and checkpoint
revisions, source hashes, generated outputs, raw scores, and the verification
report. The runner uses batch size 1; keep batch/chunk sizes fixed when resuming.

## Coverage limits

The ten-example smoke run validates execution and data flow. Its metrics are
not estimates of paper-scale performance; the 2,000-text comparison and complete
cross-detector experiment were not rerun for this release.

The earlier 2026-09-22 validation record covered representative Llama,
Ministral, AP, and other detector paths. Those GPU paths were not repeated in
this release check. That record also noted slow GPT-J missing-parameter
initialization under Transformers 5.8.1 for Fast-DetectGPT. This path remains
unverified end to end; the smoke constraints do not certify its compatibility.
