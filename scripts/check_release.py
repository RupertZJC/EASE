"""Lightweight release audit that does not load models or require a GPU."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = (
    "evaluation_effective",
    "evaluation_competitor",
    "np.interp(0.01 / 100",
    "ease_rewrite_d4",
)
FORBIDDEN_SCIENTIFIC_FLAGS = (
    "--temperature", "--delta", "--top-k", "--option",
    "--n-eval", "--n-calibration", "--gen-len", "--prompt-len",
)


def main():
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        if any(part in {"__pycache__", ".venv"} for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        for fragment in FORBIDDEN_TEXT:
            if fragment in source:
                raise AssertionError(f"forbidden legacy reference {fragment!r}: {path}")
        for flag in FORBIDDEN_SCIENTIFIC_FLAGS:
            if f'add_argument("{flag}"' in source or f"add_argument('{flag}'" in source:
                raise AssertionError(f"scientific CLI override {flag!r}: {path}")

    from ease.config import CROSS_DETECTOR_CONFIG, TABLE1_CONFIG

    table = TABLE1_CONFIG["profiles"]
    assert table["np"]["temperature"] == 1.0
    assert table["simple"]["temperature"] == table["recursive"]["temperature"] == 0.6
    assert table["adv_base"]["temperature"] == table["adv_large"]["temperature"] == 1.0
    assert table["ease_plugin"]["delta"] == table["ease_rewrite"]["delta"] == 2.0
    assert TABLE1_CONFIG["target_fpr"] == CROSS_DETECTOR_CONFIG["target_fpr"] == 0.01
    assert not (ROOT / "configs").exists(), "src/ease/config.py must be the only configuration source"
    assert not (ROOT / "pyproject.toml").exists(), "requirements.txt is the dependency source"
    smoke = ROOT / "dataset_splits" / "smoke10"
    samples = json.loads((smoke / "samples.json").read_text(encoding="utf-8"))
    calibration, evaluation = samples["calibration_human"], samples["evaluation"]
    assert len(calibration) == len(evaluation) == 10
    cal_ids = {row["source_index"] for row in calibration}
    eval_ids = {row["source_index"] for row in evaluation}
    assert len(cal_ids) == len(eval_ids) == 10 and not cal_ids & eval_ids
    for name, rows in (("human_calibration", calibration), ("human_evaluation", evaluation)):
        texts = json.loads((smoke / f"{name}.json").read_text(encoding="utf-8"))
        assert texts == [row["human_text"] for row in rows]
        assert all(isinstance(text, str) and text.strip() for text in texts)
    assert all(len(row["prompt_ids"]) == TABLE1_CONFIG["prompt_tokens"] for row in evaluation)
    assert all(len(row["human_ids"]) == TABLE1_CONFIG["generated_tokens"]
               for row in calibration + evaluation)
    print(f"RELEASE AUDIT PASSED ({len(python_files)} Python files)")


if __name__ == "__main__":
    main()
