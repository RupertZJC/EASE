"""EASE: entropy-adaptive distribution shaping for LLM decoding."""

from .config import CROSS_DETECTOR_CONFIG, TABLE1_CONFIG
from .metrics import auroc, calibrate_threshold_at_fpr, tpr_at_threshold

__all__ = [
    "TABLE1_CONFIG",
    "CROSS_DETECTOR_CONFIG",
    "auroc",
    "calibrate_threshold_at_fpr",
    "tpr_at_threshold",
]
