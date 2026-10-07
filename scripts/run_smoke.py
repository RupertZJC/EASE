"""Run real ten-example generation, detector scoring and metric verification."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("np", "ease_plugin", "ease_rewrite_d2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/smoke")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    out = str(Path(args.out_dir).resolve())
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    def run(module, stage, *extra):
        subprocess.run(
            [sys.executable, "-m", module, "--stage", stage,
             "--out-dir", out, "--smoke", *extra],
            cwd=ROOT, env=env, check=True,
        )

    generate = "experiments.table1.generate"
    rewrite = "experiments.table1.generate_rewrite"
    score = "experiments.table1.score"
    run(generate, "prepare")
    for method in METHODS[:2]:
        run(generate, "generate", "--corpus", method,
            "--batch-size", str(args.batch_size))
        run(generate, "merge", "--corpus", method)
    run(rewrite, "prepare", "--source-dir", out)
    run(rewrite, "generate", "--batch-size", str(args.batch_size))
    run(rewrite, "merge")
    run(rewrite, "validate")
    for method in METHODS:
        values = json.loads((Path(out) / "corpora" / f"{method}.json").read_text(encoding="utf-8"))
        if len(values) != 10 or any(not isinstance(text, str) or not text.strip() for text in values):
            raise AssertionError(f"invalid smoke corpus: {method}")
    for method in METHODS[:2]:
        metadata = json.loads((Path(out) / "corpora" / f"{method}_metadata.json").read_text(encoding="utf-8"))
        if metadata["token_count_min"] != 200 or metadata["token_count_max"] != 200:
            raise AssertionError(f"invalid continuation length: {method}")
    for stage in ("score", "finalize", "verify"):
        run(score, stage, "--detector", "roberta_base", "--corpora", *METHODS)
    print(f"GPU SMOKE PASSED: {out}/verification.json", flush=True)


if __name__ == "__main__":
    main()
