"""Training loop with early stopping, checkpointing, gradient clipping, and LR warmup."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.losses import build_criterion
from src.utils.metrics import MetricsTracker, compute_metrics
from src.utils.reproducibility import count_parameters

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training, validation, and testing with metric tracking.

    Supports segmentation (pixel-level) and classification tasks uniformly.
    For segmentation the model outputs (B, C, H, W) logits and targets are
    (B, H, W) integer masks.

    Key features:
    - Early stopping (``training.early_stopping_patience``)
    - Gradient clipping (``training.gradient_clip_val``)
    - Linear LR warmup + cosine decay (``training.warmup_epochs``)
    - Configurable loss: ``'ce'``, ``'dice'``, ``'combined'``
      (``training.loss``, ``training.dice_weight``)
    - Inverse-frequency class weights (``training.use_class_weights``)
    - Configurable monitor metric (``training.monitor``)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: Any,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.cfg          = cfg
        self.device       = device

        t_cfg = cfg.training

        # Number of classes (needed for Dice)
        self.num_classes: int = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )

        # Loss
        use_weights = t_cfg.get("use_class_weights", False)
        w = class_weights.to(device) if (use_weights and class_weights is not None) else None
        if w is not None:
            logger.info(f"Class weights: {w.cpu().tolist()}")

        loss_type  = t_cfg.get("loss", "ce")
        dice_w     = float(t_cfg.get("dice_weight", 0.5))
        self.criterion = build_criterion(loss_type, self.num_classes, dice_w, w)
        logger.info(f"Loss: {loss_type}" + (f" (dice_weight={dice_w})" if loss_type == "combined" else ""))

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        monitor = t_cfg.get("monitor", "val_mean_dice")
        self.tracker = MetricsTracker(monitor=monitor, mode="max")
        self.num_params = count_parameters(model)

        self.grad_clip: Optional[float] = t_cfg.get("gradient_clip_val", None)
        if self.grad_clip is not None:
            logger.info(f"Gradient clipping: max_norm={self.grad_clip}")

        # Mixed precision — disabled automatically for CPU or when use_amp=false.
        # Quantum models keep float32 inside the circuit; the classical
        # encoder/decoder still benefit from fp16 activations.
        self.use_amp: bool = (
            t_cfg.get("use_amp", False) and device.type == "cuda"
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision (AMP) enabled")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        t = self.cfg.training
        if t.optimizer == "adam":
            return Adam(self.model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
        if t.optimizer == "sgd":
            return SGD(self.model.parameters(), lr=t.learning_rate,
                       momentum=0.9, weight_decay=t.weight_decay)
        raise ValueError(f"Unknown optimizer: {t.optimizer}")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        t = self.cfg.training
        warmup = int(t.get("warmup_epochs", 0))
        total  = int(t.epochs)

        if t.scheduler == "cosine":
            cosine_ep = max(1, total - warmup)
            cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_ep, eta_min=1e-6)
            if warmup > 0:
                warmup_sched = LambdaLR(self.optimizer,
                                        lr_lambda=lambda ep: float(ep + 1) / warmup)
                sched = SequentialLR(self.optimizer,
                                     schedulers=[warmup_sched, cosine],
                                     milestones=[warmup])
                logger.info(f"LR: warmup {warmup}ep → cosine {cosine_ep}ep")
                return sched
            logger.info(f"LR: cosine {total}ep")
            return cosine

        if t.scheduler == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.5)

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run full training loop with validation and early stopping."""
        t_cfg = self.cfg.training
        best_state = None
        patience   = 0

        logger.info(f"Training: {t_cfg.epochs} epochs | {self.num_params:,} params")
        t0 = time.time()

        for epoch in range(1, t_cfg.epochs + 1):
            train_loss, train_dice = self._train_epoch(epoch)
            val_loss, val_metrics  = self._evaluate(self.val_loader)

            metrics = {
                "train_loss":      train_loss,
                "train_mean_dice": train_dice,
                "val_loss":        val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            is_best = self.tracker.update(epoch, metrics)
            if is_best:
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience   = 0
            else:
                patience += 1

            if self.scheduler is not None:
                self.scheduler.step()

            logger.info(
                f"Epoch {epoch}/{t_cfg.epochs} | "
                f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
                f"val loss={val_loss:.4f} dice={val_metrics.get('mean_dice', 0):.4f} | "
                f"patience {patience}/{t_cfg.early_stopping_patience}"
            )

            min_ep = int(t_cfg.get("early_stopping_min_epoch", 0))
            if epoch >= min_ep and patience >= t_cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - t0

        # Capture final state before restoring best
        self._final_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self._best_state = best_state

        _, test_metrics = self._evaluate(self.test_loader)

        results = {
            "num_parameters":        self.num_params,
            "training_time_seconds": total_time,
            "best_epoch":            self.tracker.best_epoch,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "history": self.tracker.history,
        }

        logger.info(
            f"Done | test_mean_dice={test_metrics.get('mean_dice', 0):.4f} | "
            f"test_pixel_acc={test_metrics.get('pixel_accuracy', 0):.4f} | "
            f"params={self.num_params:,} | time={total_time:.1f}s"
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_pred:   list = []
        all_target: list = []

        for data, target in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.use_amp):
                output = self.model(data)
                loss   = self.criterion(output, target)

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1).cpu().numpy().ravel()
            tgt  = target.cpu().numpy().ravel()
            all_pred.append(pred)
            all_target.append(tgt)

        n = sum(len(p) for p in all_pred)
        y_pred   = np.concatenate(all_pred)
        y_target = np.concatenate(all_target)

        from src.utils.metrics import compute_mean_dice
        mean_dice = compute_mean_dice(y_target, y_pred, self.num_classes)

        # Approximate sample count for normalising loss
        n_samples = len(self.train_loader.dataset)  # type: ignore[arg-type]
        return total_loss / n_samples, mean_dice

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss  = 0.0
        all_pred:   list = []
        all_target: list = []

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            total_loss += self.criterion(output, target).item() * data.size(0)

            all_pred.append(output.argmax(dim=1).cpu().numpy().ravel())
            all_target.append(target.cpu().numpy().ravel())

        # NOTE: metrics here are MICRO-averaged — predictions/targets from the
        # whole loader are pooled into one flat array, so the resulting Dice is
        # a single pixel-pooled score over all subjects, NOT a per-subject mean.
        # Report per-subject mean +/- s.d. (eval_per_subject2.py) in papers.
        n_samples = sum(len(p) for p in all_pred)
        y_pred    = np.concatenate(all_pred)
        y_true    = np.concatenate(all_target)

        metric_names = list(self.cfg.evaluation.metrics)
        metrics = compute_metrics(y_true, y_pred, metric_names, self.num_classes)

        avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
        return avg_loss, metrics

    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        out = Path(output_dir)
        path = out / "results.json"
        with open(path, "w") as f:
            json.dump(_make_serializable(results), f, indent=2)
        logger.info(f"Results saved to {path}")

        if hasattr(self, "_best_state"):
            ckpt_path = out / "best_model.pth"
            torch.save(self._best_state, ckpt_path)
            logger.info(f"Best model weights saved to {ckpt_path}")

        if hasattr(self, "_final_state"):
            ckpt_path = out / "final_model.pth"
            torch.save(self._final_state, ckpt_path)
            logger.info(f"Final model weights saved to {ckpt_path}")


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
