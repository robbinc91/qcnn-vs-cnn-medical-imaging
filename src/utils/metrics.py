"""Evaluation metrics and tracking for experiment analysis."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: List[str],
) -> Dict[str, float]:
    """Compute specified metrics from predictions."""
    results: Dict[str, float] = {}
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"

    metric_fn = {
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "f1_macro": lambda: f1_score(y_true, y_pred, average=average, zero_division=0),
        "precision": lambda: precision_score(
            y_true, y_pred, average=average, zero_division=0
        ),
        "recall": lambda: recall_score(
            y_true, y_pred, average=average, zero_division=0
        ),
    }

    for name in metric_names:
        if name in metric_fn:
            results[name] = float(metric_fn[name]())
        else:
            logger.warning(f"Unknown metric: {name}")

    return results


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


@dataclass
class MetricsTracker:
    """Track training and validation metrics across epochs."""

    history: Dict[str, List[float]] = field(default_factory=dict)
    best_metric: Optional[float] = None
    best_epoch: int = 0
    monitor: str = "val_accuracy"
    mode: str = "max"  # max or min

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update metrics history. Returns True if new best."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        monitored = metrics.get(self.monitor)
        if monitored is None:
            return False

        is_best = False
        if self.best_metric is None:
            is_best = True
        elif self.mode == "max" and monitored > self.best_metric:
            is_best = True
        elif self.mode == "min" and monitored < self.best_metric:
            is_best = True

        if is_best:
            self.best_metric = monitored
            self.best_epoch = epoch
            logger.info(
                f"New best {self.monitor}: {monitored:.4f} at epoch {epoch}"
            )

        return is_best

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of tracked metrics."""
        summary: Dict[str, Any] = {
            "best_epoch": self.best_epoch,
            f"best_{self.monitor}": self.best_metric,
        }
        for key, values in self.history.items():
            summary[f"final_{key}"] = values[-1] if values else None
        return summary
