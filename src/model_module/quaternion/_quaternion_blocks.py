"""Shared higher-level quaternion building blocks.

These sit on top of :mod:`_quaternion_ops` and are reused by the quaternion
ports of the classical architectures (U-Net, ACAPULCO, MipaimUNet, MARIN).

Design note — attention gating (CBAM) operates on the *real* representation of
the quaternion tensor (the ``4 * Q`` channels treated as ordinary channels).
This "split" attention keeps the lightweight gating cheap and leaves the heavy
convolutions — where quaternion weight sharing actually matters — quaternion.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from .._ops import SpatialOps
from ._quaternion_ops import QuaternionConv

logger = logging.getLogger(__name__)


class QuaternionUp(nn.Module):
    """Resize-convolution upsampler: interpolate to target size → QuaternionConv.

    Interpolation acts component-wise so it preserves the quaternion layout, and
    avoids the checkerboard artifacts of transposed convolutions.
    """

    def __init__(self, in_q: int, out_q: int, ops: SpatialOps) -> None:
        super().__init__()
        self.conv = QuaternionConv(in_q, out_q, ops, 3, padding=1, bias=True)
        self._ops = ops

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = self._ops.interpolate(x, size)
        return self.conv(x)


class QuaternionCBAM(nn.Module):
    """Convolutional Block Attention Module over the real (4*Q) channels.

    Channel attention (shared MLP on global avg+max descriptors) followed by
    spatial attention (7-kernel conv on per-channel avg+max), per Woo et al.
    (2018). Dimension-agnostic via :class:`SpatialOps`.
    """

    def __init__(self, q_channels: int, ops: SpatialOps, ratio: int = 8) -> None:
        super().__init__()
        self._ops = ops
        channels = 4 * q_channels
        mid = max(1, channels // ratio)
        self.ch_fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sp_conv = ops.Conv(2, 1, kernel_size=7, padding=3, bias=False)
        self._extra = tuple([1] * ops.dims)  # (1,1) or (1,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        # Channel attention
        avg_c = self._ops.adaptive_avg_pool(x, 1).view(B, C)
        max_c = self._ops.adaptive_max_pool(x, 1).view(B, C)
        ch_att = torch.sigmoid(self.ch_fc(avg_c) + self.ch_fc(max_c))
        x = x * ch_att.view(B, C, *self._extra)
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sp_att = torch.sigmoid(self.sp_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_att
