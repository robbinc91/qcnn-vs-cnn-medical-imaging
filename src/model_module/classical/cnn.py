"""Classical CNN baselines for comparison with quantum models."""

import logging
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> [BatchNorm] -> ReLU -> MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(pool_size)])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@register_model("classical_cnn")
class ClassicalCNN(nn.Module):
    """Standard CNN with configurable architecture."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        arch = cfg.model.architecture
        channels = [cfg.dataset.channels] + list(arch.conv_channels)
        conv_layers: List[nn.Module] = []

        for i in range(len(arch.conv_channels)):
            conv_layers.append(
                ConvBlock(
                    channels[i],
                    channels[i + 1],
                    arch.kernel_size,
                    arch.pool_size,
                    arch.batch_norm,
                )
            )

        self.features = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Compute flattened size
        feat_size = cfg.dataset.image_size
        for _ in arch.conv_channels:
            feat_size = feat_size // arch.pool_size
        flat_dim = arch.conv_channels[-1] * feat_size * feat_size

        fc_layers: List[nn.Module] = []
        prev_dim = flat_dim
        for dim in arch.fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(arch.dropout),
            ])
            prev_dim = dim

        num_classes = len(cfg.dataset.binary_classes) if cfg.dataset.binary_classes else cfg.dataset.num_classes
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


@register_model("classical_cnn_small")
class ClassicalCNNSmall(ClassicalCNN):
    """Small CNN with matched parameter count to QCNN."""

    pass  # Same architecture, different config drives smaller size
