"""Quaternion U-Net for 2-D or 3-D brainstem segmentation.

Structurally identical to :class:`ClassicalCNN` (encoder / bottleneck / decoder
U-Net with skip connections) but every convolution is a :class:`QuaternionConv`,
so features and weights live in quaternion space.  Input (1 real channel) is
lifted to quaternion space by a real stem; the final segmentation head is a real
1x1 conv.

Parameter matching
------------------
A quaternion conv has 1/4 the parameters of a real conv between the same number
of *real* channels.  To match the ~17 M-parameter classical / quantum field, the
real channel widths are therefore set to **2x** the ClassicalCNN defaults
(params scale ~width^2, and (2w)^2 / 4 = w^2):

    2D: [120, 240, 480, 672]   → ~17.1 M  (≈ classical [64, 128, 256, 352])
    3D: [72, 144, 288, 384]    → ~17.1 M  (≈ classical [38, 76, 156, 204])

All widths must be divisible by 4 (real channels = 4 x quaternion channels).
``dataset.spatial_dims`` selects 2-D or 3-D mode via the shared ``get_ops``.
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


class QuaternionDoubleConv(nn.Module):
    """(QuaternionConv → [split BN] → ReLU) × 2, with optional Dropout.

    BatchNorm and ReLU are applied "split" — independently over the 4*Q real
    channels — which is the standard simplification of quaternion normalization
    and activation and keeps the component-major layout intact.
    """

    def __init__(
        self,
        in_q: int,
        out_q: int,
        ops: SpatialOps,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = not batch_norm
        layers: List[nn.Module] = [
            QuaternionConv(in_q, out_q, ops, 3, padding=1, bias=bias),
            ops.BatchNorm(4 * out_q) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            QuaternionConv(out_q, out_q, ops, 3, padding=1, bias=bias),
            ops.BatchNorm(4 * out_q) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(ops.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("quaternion_unet")
@register_model("quaternion_cnn")
class QuaternionUNet(nn.Module):
    """Quaternion-valued U-Net for 2-D or 3-D semantic segmentation.

    Config keys (under ``model``):
        enc_channels (list[int]): Encoder widths in REAL channels (each must be
            divisible by 4). Auto-selected per spatial_dims if omitted:
              2D: [120, 240, 480, 672]  → ~17.1 M params
              3D: [72, 144, 288, 384]   → ~17.1 M params
        batch_norm (bool): Use split BatchNorm. Default: True.
        decoder_dropout (float): Decoder dropout. Default: 0.2.
    ``dataset.spatial_dims`` selects 2-D or 3-D mode.
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

        _default_chs = (
            [120, 240, 480, 672] if ops.dims == 2 else [72, 144, 288, 384]
        )
        chs: List[int] = list(getattr(cfg.model, "enc_channels", _default_chs))
        bn: bool = bool(getattr(cfg.model, "batch_norm", True))
        dp: float = float(getattr(cfg.model, "decoder_dropout", 0.2))

        assert all(c % 4 == 0 for c in chs), (
            "enc_channels must all be divisible by 4 "
            f"(real channels = 4 x quaternion channels), got {chs}"
        )
        q_chs: List[int] = [c // 4 for c in chs]

        # Real stem: lift 1-channel input into quaternion space (4*q0 real ch).
        self.stem = nn.Sequential(
            ops.Conv(in_ch, chs[0], 3, padding=1, bias=not bn),
            ops.BatchNorm(chs[0]) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc_blocks = nn.ModuleList()
        q_in = q_chs[0]
        for q in q_chs:
            self.enc_blocks.append(QuaternionDoubleConv(q_in, q, ops, bn))
            q_in = q
        self.pool = ops.MaxPool(2)

        # Bottleneck
        self.bottleneck = QuaternionDoubleConv(q_chs[-1], q_chs[-1] * 2, ops, bn)

        # Decoder (quaternion upsample + quaternion-aware skip concat)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        q_dec_in = q_chs[-1] * 2
        for q in reversed(q_chs):
            self.ups.append(QuaternionUp(q_dec_in, q, ops))
            self.dec_blocks.append(QuaternionDoubleConv(q * 2, q, ops, bn, dp))
            q_dec_in = q

        # Real segmentation head (quaternion → class logits)
        self.head = ops.Conv(chs[0], num_cls, 1)
        self._ops = ops
        self._q_chs = q_chs

        logger.info(
            f"QuaternionUNet {ops.dims}D | real_channels={chs} | "
            f"quaternion_channels={q_chs} | in={in_ch}ch → {num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: List[torch.Tensor] = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, (skip, q) in zip(
            self.ups, self.dec_blocks, zip(reversed(skips), reversed(self._q_chs))
        ):
            x = up(x, skip.shape[2:])
            x = quaternion_cat([skip, x], [q, q])
            x = dec(x)

        return self.head(x)
