"""Quaternion ACAPULCO-style parcellating U-Net for 2-D or 3-D segmentation.

Quaternion port of :class:`ACAPULCONet` (Han et al., NeuroImage 2020): the
pre-activation residual structure and instance normalization are preserved, but
every 3x3 / 1x1 convolution is a :class:`QuaternionConv`.  InstanceNorm and ReLU
are applied "split" over the ``4 * Q`` real channels, which keeps the
component-major quaternion layout intact.

Real channel widths (config ``enc_channels``) must be divisible by 4
(real = 4 x quaternion).  Defaults are sized to the ~17 M field; because a
quaternion conv has 1/4 the params of a real conv at the same real width, the
widths are ~2x the classical ACAPULCO defaults.

``dataset.spatial_dims`` selects 2-D / 3-D via the shared ``get_ops``.
"""

import logging
from typing import Any, List

import torch
import torch.nn as nn

from .._ops import SpatialOps, get_ops
from ..registry import register_model
from ._quaternion_ops import QuaternionConv, quaternion_cat
from ._quaternion_blocks import QuaternionUp

logger = logging.getLogger(__name__)


class QPreActResBlock(nn.Module):
    """Quaternion pre-activation residual block: (IN → ReLU → QConv) × 2 + shortcut."""

    def __init__(self, in_q: int, out_q: int, ops: SpatialOps) -> None:
        super().__init__()
        self.norm1 = ops.InstanceNorm(4 * in_q, affine=True)
        self.conv1 = QuaternionConv(in_q, out_q, ops, 3, padding=1, bias=False)
        self.norm2 = ops.InstanceNorm(4 * out_q, affine=True)
        self.conv2 = QuaternionConv(out_q, out_q, ops, 3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.proj = (
            QuaternionConv(in_q, out_q, ops, 1, padding=0, bias=False)
            if in_q != out_q else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x)
        out = self.conv1(self.act(self.norm1(x)))
        out = self.conv2(self.act(self.norm2(out)))
        return out + shortcut


class QEncoderStage(nn.Module):
    """Optional MaxPool downsample → QPreActResBlock × num_blocks."""

    def __init__(
        self, in_q: int, out_q: int, ops: SpatialOps, num_blocks: int, downsample: bool
    ) -> None:
        super().__init__()
        self.pool = ops.MaxPool(2) if downsample else nn.Identity()
        blocks: List[nn.Module] = [QPreActResBlock(in_q, out_q, ops)]
        for _ in range(num_blocks - 1):
            blocks.append(QPreActResBlock(out_q, out_q, ops))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pool(x))


class QDecoderStage(nn.Module):
    """Quaternion upsample → quaternion-aware concat skip → QPreActResBlock × n."""

    def __init__(
        self, in_q: int, skip_q: int, out_q: int, ops: SpatialOps, num_blocks: int
    ) -> None:
        super().__init__()
        self.up = QuaternionUp(in_q, in_q, ops)
        self._in_q = in_q
        self._skip_q = skip_q
        blocks: List[nn.Module] = [QPreActResBlock(in_q + skip_q, out_q, ops)]
        for _ in range(num_blocks - 1):
            blocks.append(QPreActResBlock(out_q, out_q, ops))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x, skip.shape[2:])
        x = quaternion_cat([x, skip], [self._in_q, self._skip_q])
        return self.blocks(x)


@register_model("quaternion_acapulco")
class QuaternionACAPULCONet(nn.Module):
    """Quaternion ACAPULCO parcellating U-Net (pre-activation + instance norm).

    Config keys (under ``model``):
        enc_channels (list[int]): REAL widths per encoder stage (÷4 = quaternion).
            Auto-selected per spatial_dims if omitted:
              2D: [84, 168, 336, 448]   → ~16.7 M
              3D: [48, 96, 192, 264]    → ~16.9 M
        blocks_per_stage (int): ResBlocks per encoder/decoder stage. Default: 2.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        ops = get_ops(cfg.dataset.spatial_dims)
        in_ch = cfg.dataset.channels
        num_cls = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        default_chs = [84, 168, 336, 448] if ops.dims == 2 else [48, 96, 192, 264]
        chs: List[int] = list(getattr(cfg.model, "enc_channels", default_chs))
        n_blk: int = int(getattr(cfg.model, "blocks_per_stage", 2))

        assert all(c % 4 == 0 for c in chs), \
            f"enc_channels must be divisible by 4, got {chs}"
        q_chs: List[int] = [c // 4 for c in chs]

        # Real stem: lift input into quaternion space.
        self.stem = ops.Conv(in_ch, chs[0], 3, padding=1, bias=False)

        # Encoder (first stage no downsample — stem is at full res)
        self.enc_stages: nn.ModuleList = nn.ModuleList()
        q_in = q_chs[0]
        for i, q_out in enumerate(q_chs):
            self.enc_stages.append(
                QEncoderStage(q_in, q_out, ops, n_blk, downsample=(i > 0))
            )
            q_in = q_out

        # Bottleneck
        q_bot = q_chs[-1] * 2
        self.bottleneck = nn.Sequential(
            ops.MaxPool(2),
            QPreActResBlock(q_chs[-1], q_bot, ops),
            QPreActResBlock(q_bot, q_bot, ops),
        )

        # Decoder
        self.dec_stages: nn.ModuleList = nn.ModuleList()
        q_dec_in = q_bot
        for q in reversed(q_chs):
            self.dec_stages.append(QDecoderStage(q_dec_in, q, q, ops, n_blk))
            q_dec_in = q

        self.head = ops.Conv(chs[0], num_cls, 1)

        logger.info(
            f"QuaternionACAPULCONet {ops.dims}D | real_channels={chs} | "
            f"quaternion_channels={q_chs} | blocks/stage={n_blk} | "
            f"in={in_ch}ch → {num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: List[torch.Tensor] = []
        for enc in self.enc_stages:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.dec_stages, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)
