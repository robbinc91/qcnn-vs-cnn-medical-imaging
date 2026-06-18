"""Segmentation loss functions: Dice, combined CE+Dice."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss.

    Computes the mean Dice loss over all foreground classes (background is
    excluded from the average by default to avoid trivial inflation from the
    dominant background class in medical images).

    Args:
        num_classes: Total number of classes (including background).
        ignore_bg: If True, exclude class 0 (background) from Dice average.
        smooth: Laplace smoothing constant to avoid division by zero.
        weight: Per-class weights tensor of shape (num_classes,). Optional.
            When given, the per-class Dice values are combined as a *weighted
            average normalised by the sum of weights* (so the loss stays in
            [0, 1]); they are NOT multiplied raw, which would mis-scale the loss.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_bg: bool = True,
        smooth: float = 1.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_bg   = ignore_bg
        self.smooth      = smooth
        self.register_buffer("weight", weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, *spatial) — raw unnormalised scores.
            targets: (B, *spatial)    — integer labels in [0, C).

        Returns:
            Scalar Dice loss (1 − mean Dice).
        """
        probs = F.softmax(logits, dim=1)  # (B, C, *spatial)

        # One-hot encode targets: (B, C, *spatial)
        spatial = targets.shape[1:]
        t_oh = F.one_hot(targets, self.num_classes)   # (B, *spatial, C)
        # Move class axis to dim 1
        perm = [0, len(t_oh.shape) - 1] + list(range(1, len(t_oh.shape) - 1))
        t_oh = t_oh.permute(*perm).float()            # (B, C, *spatial)

        start_cls = 1 if self.ignore_bg else 0
        dice_scores = []
        for c in range(start_cls, self.num_classes):
            p = probs[:, c].reshape(probs.shape[0], -1)      # (B, N)
            t = t_oh[:, c].reshape(t_oh.shape[0], -1)        # (B, N)
            intersection = (p * t).sum(dim=1)                 # (B,)
            cardinality  = p.sum(dim=1) + t.sum(dim=1)        # (B,)
            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice_c.mean())

        dice_stack = torch.stack(dice_scores)                 # (n_fg_classes,)
        if self.weight is not None:
            # Weighted *average* of the per-class Dice, normalised by the sum of
            # the weights so the result stays in [0, 1].  A plain multiply
            # (dice_c * weight[c]) without this normalisation mis-scales the
            # loss and can drive it negative when class weights are >> 1.
            w = self.weight[start_cls:self.num_classes].to(dice_stack.dtype)
            mean_dice = (dice_stack * w).sum() / (w.sum() + 1e-8)
        else:
            mean_dice = dice_stack.mean()

        return 1.0 - mean_dice


class CombinedLoss(nn.Module):
    """Weighted sum of CrossEntropyLoss and DiceLoss.

    ``loss = (1 - dice_weight) * CE  +  dice_weight * Dice``

    Args:
        num_classes: Total number of classes.
        dice_weight: Blending coefficient in [0, 1]. Default: 0.5.
        ignore_bg:   Exclude background from Dice. Default: True.
        weight:      Per-class CE/Dice weight tensor. Optional.
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ignore_bg: bool = True,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce   = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes, ignore_bg=ignore_bg, weight=weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (1.0 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


def build_criterion(
    loss_type: str,
    num_classes: int,
    dice_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Factory that builds a loss function from a string name.

    Args:
        loss_type: One of ``'ce'``, ``'dice'``, ``'combined'``.
        num_classes: Number of output classes.
        dice_weight: Blend ratio for ``'combined'`` loss.
        class_weights: Optional per-class weight tensor.

    Returns:
        Configured loss module.
    """
    w = class_weights
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if loss_type == "dice":
        return DiceLoss(num_classes, weight=w)
    if loss_type == "combined":
        return CombinedLoss(num_classes, dice_weight=dice_weight, weight=w)
    raise ValueError(f"Unknown loss type '{loss_type}'. Choose: ce, dice, combined.")
