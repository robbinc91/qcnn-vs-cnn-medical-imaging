"""Training loop with early stopping, checkpointing, and metric tracking."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import MetricsTracker, compute_metrics
from src.utils.reproducibility import count_parameters

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training, validation, and testing with metric tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: Any,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.tracker = MetricsTracker(monitor="val_accuracy", mode="max")
        self.num_params = count_parameters(model)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        t_cfg = self.cfg.training
        if t_cfg.optimizer == "adam":
            return Adam(
                self.model.parameters(),
                lr=t_cfg.learning_rate,
                weight_decay=t_cfg.weight_decay,
            )
        elif t_cfg.optimizer == "sgd":
            return SGD(
                self.model.parameters(),
                lr=t_cfg.learning_rate,
                momentum=0.9,
                weight_decay=t_cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {t_cfg.optimizer}")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        t_cfg = self.cfg.training
        if t_cfg.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=t_cfg.epochs)
        elif t_cfg.scheduler == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)
        return None

    def train(self) -> Dict[str, Any]:
        """Run full training loop with validation and early stopping."""
        t_cfg = self.cfg.training
        best_model_state = None
        patience_counter = 0

        logger.info(f"Starting training: {t_cfg.epochs} epochs, {self.num_params} params")
        start_time = time.time()

        for epoch in range(1, t_cfg.epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc, val_metrics = self._validate(self.val_loader)

            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            is_best = self.tracker.update(epoch, metrics)
            if is_best:
                best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if self.scheduler:
                self.scheduler.step()

            logger.info(
                f"Epoch {epoch}/{t_cfg.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if patience_counter >= t_cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time

        # Restore best model for testing
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Test evaluation
        test_loss, test_acc, test_metrics = self._validate(self.test_loader)

        results = {
            "num_parameters": self.num_params,
            "training_time_seconds": total_time,
            "best_epoch": self.tracker.best_epoch,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "history": self.tracker.history,
        }

        logger.info(f"Test Accuracy: {test_acc:.4f} | Params: {self.num_params:,}")
        return results

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(
            self.train_loader, desc=f"Epoch {epoch}", leave=False
        )):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _validate(
        self, loader: DataLoader
    ) -> Tuple[float, float, Dict[str, float]]:
        """Evaluate on a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            total_loss += self.criterion(output, target).item() * data.size(0)
            all_preds.extend(output.argmax(dim=1).cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

        avg_loss = total_loss / len(all_targets)
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        metrics = compute_metrics(
            y_true, y_pred, self.cfg.evaluation.metrics
        )
        acc = metrics.get("accuracy", 0.0)

        return avg_loss, acc, metrics

    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """Save experiment results to JSON."""
        path = Path(output_dir) / "results.json"
        # Convert numpy types for JSON serialization
        clean = _make_serializable(results)
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info(f"Results saved to {path}")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
