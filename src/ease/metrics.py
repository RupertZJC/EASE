"""Single, auditable definition of the released detection metrics."""

import numpy as np


def _finite_vector(values, name):
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D array")
    return scores


def calibrate_threshold_at_fpr(human_calibration_scores, target_fpr=0.01):
    """Choose a human-only threshold whose empirical FPR does not exceed target.

    ``target_fpr`` is a fraction: 1% FPR is exactly ``0.01``. Ties at the
    quantile are handled conservatively by moving the threshold upward.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be in (0, 1); use 0.01 for 1% FPR")
    scores = _finite_vector(human_calibration_scores, "human_calibration_scores")
    threshold = float(np.quantile(scores, 1.0 - target_fpr, method="higher"))
    empirical_fpr = float(np.mean(scores >= threshold))
    if empirical_fpr > target_fpr:
        threshold = float(np.nextafter(threshold, np.inf))
        empirical_fpr = float(np.mean(scores >= threshold))
    if empirical_fpr > target_fpr + np.finfo(np.float64).eps:
        raise AssertionError("calibrated FPR exceeds target_fpr")
    return threshold, empirical_fpr


def tpr_at_threshold(machine_scores, threshold):
    scores = _finite_vector(machine_scores, "machine_scores")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return float(np.mean(scores >= threshold))


def auroc(human_evaluation_scores, machine_scores):
    human = _finite_vector(human_evaluation_scores, "human_evaluation_scores")
    machine = _finite_vector(machine_scores, "machine_scores")
    # Mann-Whitney interpretation of AUROC, including half credit for ties.
    comparisons = (machine[:, None] > human[None, :]).mean()
    ties = (machine[:, None] == human[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def evaluate_detector(human_calibration_scores, human_evaluation_scores,
                      machine_scores, target_fpr=0.01):
    threshold, calibration_fpr = calibrate_threshold_at_fpr(
        human_calibration_scores, target_fpr
    )
    return {
        "auroc": auroc(human_evaluation_scores, machine_scores),
        "tpr_at_1pct_fpr": tpr_at_threshold(machine_scores, threshold),
        "threshold": threshold,
        "calibration_fpr": calibration_fpr,
        "target_fpr": float(target_fpr),
    }
