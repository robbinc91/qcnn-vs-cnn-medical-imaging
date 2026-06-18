"""Classical CNN U-Net for 2-D or 3-D segmentation.

Controlled by ``dataset.spatial_dims`` (2 or 3).  All conv/pool/norm ops
are selected via the shared ``_ops.get_ops()`` helper.
"""

import logging
from typing import Any, List

import torch
import torch.nn as nn

from .._ops import get_ops
from ..registry import register_model

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """(Conv → [BN] → ReLU) × 2, with optional Dropout, dimension-agnostic."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ops,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = not batch_norm
        layers: List[nn.Module] = [
            ops.Conv(in_ch, out_ch, 3, padding=1, bias=bias),
            ops.BatchNorm(out_ch) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            ops.Conv(out_ch, out_ch, 3, padding=1, bias=bias),
            ops.BatchNorm(out_ch) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(ops.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("classical_cnn")
@register_model("classical_cnn_small")
class ClassicalCNN(nn.Module):
    """U-Net for 2-D or 3-D semantic segmentation.

    Config keys under ``model.architecture``:
        conv_channels (list[int]): Encoder channel widths. Default: [16, 32, 64].
        batch_norm (bool): Use BatchNorm. Default: True.
        dropout (float): Decoder dropout. Default: 0.2.
    ``dataset.spatial_dims`` selects 2-D or 3-D mode.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        arch     = cfg.model.architecture
        ops      = get_ops(cfg.dataset.spatial_dims)
        # Auto-select ~17 M defaults with 4 downsamples per spatial_dims:
        #   2D: [64, 128, 256, 352]   → ~17.0 M  (4 enc pools + bottleneck at /16)
        #   3D: [38, 76, 156, 204]    → ~16.9 M  (4 enc pools + bottleneck at /16)
        _default_chs = [64, 128, 256, 352] if ops.dims == 2 else [38, 76, 156, 204]
        chs: List[int] = list(getattr(arch, "conv_channels", _default_chs))
        in_ch    = cfg.dataset.channels
        num_cls  = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        bn: bool  = arch.batch_norm
        dp: float = arch.dropout

        # Encoder
        self.enc_blocks = nn.ModuleList()
        ch_in = in_ch
        for ch in chs:
            self.enc_blocks.append(DoubleConv(ch_in, ch, ops, bn))
            ch_in = ch
        self.pool = ops.MaxPool(2)

        # Bottleneck
        self.bottleneck = DoubleConv(chs[-1], chs[-1] * 2, ops, bn)

        # Decoder
        self.up_convs  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch_in = chs[-1] * 2
        for ch in reversed(chs):
            self.up_convs.append(ops.ConvTranspose(ch_in, ch, 2, stride=2))
            self.dec_blocks.append(DoubleConv(ch * 2, ch, ops, bn, dp))
            ch_in = ch

        self.head = ops.Conv(chs[0], num_cls, 1)
        self._ops = ops

        logger.info(
            f"ClassicalCNN U-Net {ops.dims}D | channels={chs} | "
            f"in={in_ch}ch → out={num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = self._ops.interpolate(x, skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)
