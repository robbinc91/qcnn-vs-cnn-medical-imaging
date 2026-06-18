"""Evaluation metrics and tracking for segmentation and classification."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def compute_dice_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_bg: bool = True,
) -> Dict[str, float]:
    """Compute per-class Dice coefficients from flat label arrays.

    Args:
        y_true: Ground-truth integer labels (N,) flattened.
        y_pred: Predicted integer labels (N,) flattened.
        num_classes: Total number of classes including background.
        ignore_bg: If True, class 0 is excluded from the results dict.

    Returns:
        Dict mapping ``'dice_class_{c}'`` to scalar Dice coefficient.
    """
    results: Dict[str, float] = {}
    start = 1 if ignore_bg else 0
    for c in range(start, num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        dice  = (2 * tp / denom) if denom > 0 else 0.0
        results[f"dice_class_{c}"] = float(dice)
    return results


def compute_mean_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    ignore_bg: bool = True,
) -> float:
    """Mean Dice coefficient across foreground classes.

    NOTE: this is a MICRO-averaged (pixel-pooled) Dice — TP/FP/FN are counted
    over the whole flat ``y_true``/``y_pred`` arrays and a single Dice per class
    is taken. When called on a full test set, all subjects' pixels are pooled
    together, so this is NOT the per-subject mean Dice ± s.d. that medical-
    imaging benchmarks report. For per-subject statistics, score each subject
    separately and average (see ``run/pipeline/eval_per_subject2.py``).
    """
    per_class = compute_dice_per_class(y_true, y_pred, num_classes, ignore_bg)
    if not per_class:
        return 0.0
    return float(np.mean(list(per_class.values())))


def pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified pixels."""
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Classification metrics (kept for backward-compat / non-medical datasets)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: List[str],
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute metrics from flat integer label arrays.

    Supports both classification and segmentation metrics in a single call.
    Segmentation-specific names: ``'mean_dice'``, ``'pixel_accuracy'``,
    ``'dice_class_1'``, ``'dice_class_2'``, ``'dice_class_3'``.
    Classification names: ``'accuracy'``, ``'f1_macro'``, ``'precision'``,
    ``'recall'``.
    """
    results: Dict[str, float] = {}
    n_cls = num_classes or int(y_true.max()) + 1

    for name in metric_names:
        if name == "mean_dice":
            results[name] = compute_mean_dice(y_true, y_pred, n_cls)

        elif name == "pixel_accuracy":
            results[name] = pixel_accuracy(y_true, y_pred)

        elif name.startswith("dice_class_"):
            c = int(name.split("_")[-1])
            per = compute_dice_per_class(y_true, y_pred, n_cls, ignore_bg=False)
            results[name] = per.get(f"dice_class_{c}", 0.0)

        elif name == "accuracy":
            results[name] = pixel_accuracy(y_true, y_pred)

        elif name in ("f1_macro", "precision", "recall"):
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score
                average = "binary" if n_cls == 2 else "macro"
                fn_map = {
                    "f1_macro":  lambda: f1_score(y_true, y_pred, average=average, zero_division=0),
                    "precision": lambda: precision_score(y_true, y_pred, average=average, zero_division=0),
                    "recall":    lambda: recall_score(y_true, y_pred, average=average, zero_division=0),
                }
                results[name] = float(fn_map[name]())
            except Exception as exc:
                logger.warning(f"Metric '{name}' failed: {exc}")
                results[name] = 0.0
        else:
            logger.warning(f"Unknown metric: {name}")

    return results


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

@dataclass
class MetricsTracker:
    """Track training and validation metrics across epochs."""

    history: Dict[str, List[float]] = field(default_factory=dict)
    best_metric: Optional[float] = None
    best_epoch: int = 0
    monitor: str = "val_mean_dice"
    mode: str = "max"

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update history. Returns True if this is a new best."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        monitored = metrics.get(self.monitor)
        if monitored is None:
            return False

        is_best = (
            self.best_metric is None
            or (self.mode == "max" and monitored > self.best_metric)
            or (self.mode == "min" and monitored < self.best_metric)
        )

        if is_best:
            self.best_metric = monitored
            self.best_epoch  = epoch
            logger.info(f"New best {self.monitor}: {monitored:.4f} at epoch {epoch}")

        return is_best

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "best_epoch": self.best_epoch,
            f"best_{self.monitor}": self.best_metric,
        }
        for key, values in self.history.items():
            summary[f"final_{key}"] = values[-1] if values else None
        return summary
