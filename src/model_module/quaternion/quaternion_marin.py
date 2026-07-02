"""Quaternion MARIN-Net for 2-D or 3-D brainstem segmentation.

Quaternion port of :class:`MARINNet`.  The distinctive elements are preserved,
with every convolution replaced by a :class:`QuaternionConv`:

  - pre-activation (split IN → PReLU → QConv) ordering
  - dual multi-scale branches (dilation 1 ‖ dilation 2) fused per block
  - competitive maxout shortcut at the bottleneck
  - CBAM attention gates on every skip connection (split over 4*Q channels)

InstanceNorm / PReLU act "split" over the ``4 * Q`` real channels; the parallel
branches are merged with :func:`quaternion_cat` to keep the component-major
layout.  ``enc_channels`` are REAL widths (÷4 = quaternion), divisible by 4.
"""

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from .._ops import SpatialOps, get_ops
from ..registry import register_model
from ._quaternion_ops import QuaternionConv, quaternion_cat
from ._quaternion_blocks import QuaternionUp, QuaternionCBAM

logger = logging.getLogger(__name__)


class QMultiScalePreActBlock(nn.Module):
    """Quaternion dual-dilated pre-activation block with additive residual."""

    def __init__(self, in_q: int, out_q: int, ops: SpatialOps, competitive: bool = False) -> None:
        super().__init__()
        self.competitive = competitive
        self.half_q = out_q // 2
        self.wide_q = out_q - self.half_q

        self.norm1 = ops.InstanceNorm(4 * in_q, affine=True)
        self.act1 = nn.PReLU(4 * in_q)
        self.branch_local = QuaternionConv(in_q, self.half_q, ops, 3, padding=1, bias=False)
        self.branch_wide = QuaternionConv(in_q, self.wide_q, ops, 3, padding=2, dilation=2, bias=False)

        self.norm2 = ops.InstanceNorm(4 * out_q, affine=True)
        self.act2 = nn.PReLU(4 * out_q)
        self.integrate = QuaternionConv(out_q, out_q, ops, 3, padding=1, bias=False)

        self.proj = (
            QuaternionConv(in_q, out_q, ops, 1, padding=0, bias=False)
            if in_q != out_q else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_act = self.act1(self.norm1(x))
        x_ms = quaternion_cat(
            [self.branch_local(x_act), self.branch_wide(x_act)],
            [self.half_q, self.wide_q],
        )
        out = self.integrate(self.act2(self.norm2(x_ms)))
        shortcut = self.proj(x)
        return torch.max(out, shortcut) if self.competitive else out + shortcut


class QMARINEncoderStage(nn.Module):
    """Optional MaxPool → QMultiScalePreActBlock × n → CBAM-attended skip."""

    def __init__(
        self, in_q: int, out_q: int, ops: SpatialOps, n_blocks: int, downsample: bool
    ) -> None:
        super().__init__()
        self.pool = ops.MaxPool(2) if downsample else nn.Identity()
        blocks: List[nn.Module] = [QMultiScalePreActBlock(in_q, out_q, ops)]
        for _ in range(n_blocks - 1):
            blocks.append(QMultiScalePreActBlock(out_q, out_q, ops))
        self.blocks = nn.Sequential(*blocks)
        self.cbam = QuaternionCBAM(out_q, ops)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(self.pool(x))
        return x, self.cbam(x)


class QMARINDecoderStage(nn.Module):
    """Quaternion upsample → quaternion-aware concat attended skip → blocks."""

    def __init__(
        self, in_q: int, skip_q: int, out_q: int, ops: SpatialOps, n_blocks: int
    ) -> None:
        super().__init__()
        self.up = QuaternionUp(in_q, in_q, ops)
        self._in_q = in_q
        self._skip_q = skip_q
        blocks: List[nn.Module] = [QMultiScalePreActBlock(in_q + skip_q, out_q, ops)]
        for _ in range(n_blocks - 1):
            blocks.append(QMultiScalePreActBlock(out_q, out_q, ops))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x, skip.shape[2:])
        x = quaternion_cat([x, skip], [self._in_q, self._skip_q])
        return self.blocks(x)


@register_model("quaternion_marin")
class QuaternionMARINNet(nn.Module):
    """Quaternion MARIN-Net: multi-scale, attended, competitive quaternion U-Net.

    Config keys (under ``model``):
        enc_channels (list[int]): REAL widths per encoder stage (÷4 = quaternion).
            Auto-selected per spatial_dims if omitted:
              2D: [84, 168, 336, 448]   → ~16.9 M
              3D: [48, 96, 192, 264]    → ~17.0 M
        blocks_per_stage (int): blocks per encoder/decoder stage. Default: 2.
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

        self.stem = QuaternionConv(  # input lifted by real stem first
            q_chs[0], q_chs[0], ops, 3, padding=1, bias=False
        )
        self.input_stem = ops.Conv(in_ch, chs[0], 3, padding=1, bias=False)

        # Encoder
        self.enc_stages: nn.ModuleList = nn.ModuleList()
        q_in = q_chs[0]
        for i, q_out in enumerate(q_chs):
            self.enc_stages.append(
                QMARINEncoderStage(q_in, q_out, ops, n_blk, downsample=(i > 0))
            )
            q_in = q_out

        # Bottleneck (competitive maxout)
        q_bot = q_chs[-1] * 2
        self.bottleneck = nn.Sequential(
            ops.MaxPool(2),
            QMultiScalePreActBlock(q_chs[-1], q_bot, ops, competitive=True),
            QMultiScalePreActBlock(q_bot, q_bot, ops, competitive=True),
        )

        # Decoder
        self.dec_stages: nn.ModuleList = nn.ModuleList()
        q_dec_in = q_bot
        for q in reversed(q_chs):
            self.dec_stages.append(QMARINDecoderStage(q_dec_in, q, q, ops, n_blk))
            q_dec_in = q

        self.head = ops.Conv(chs[0], num_cls, 1)

        logger.info(
            f"QuaternionMARINNet {ops.dims}D | real_channels={chs} | "
            f"quaternion_channels={q_chs} | bottleneck={q_bot * 4}real | "
            f"blocks/stage={n_blk} | in={in_ch}ch → {num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.input_stem(x))

        skips: List[torch.Tensor] = []
        for enc in self.enc_stages:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.dec_stages, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)
