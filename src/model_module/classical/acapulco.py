"""ACAPULCO-style parcellating U-Net for 2D or 3D segmentation.

Based on: Han et al., "Automatic Cerebellum Anatomical Parcellation using
U-Net with Locally Constrained Optimization," NeuroImage 2020.
https://pubmed.ncbi.nlm.nih.gov/32438049/

Architecture — parcellating network (Han 2020, Figs 2–3):
- Pre-activation residual blocks: (IN → ReLU → Conv) × 2 + shortcut
- Instance normalization throughout (invariant to intensity rescaling)
- MaxPool downsampling; ConvTranspose upsampling
- Skip connections (concat) at each decoder level

Two-stage cascade note: the original ACAPULCO first runs a locating
network to crop a bounding box around the cerebellum, then passes the
crop to the parcellating U-Net.  For brainstem segmentation the ROI is
already known, so only the parcellating U-Net is implemented here.

``dataset.spatial_dims`` selects 2D (kernel 3×3) or 3D (kernel 3×3×3)
operations via ``get_ops()``.
"""

import logging
from typing import Any, List

import torch
import torch.nn as nn

from .._ops import SpatialOps, get_ops
from ..registry import register_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PreActResBlock(nn.Module):
    """Pre-activation residual block: (IN → ReLU → Conv) × 2 + shortcut.

    When ``in_ch != out_ch`` a 1×1 projection is used for the shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, ops: SpatialOps) -> None:
        super().__init__()
        self.norm1 = ops.InstanceNorm(in_ch, affine=True)
        self.conv1 = ops.Conv(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = ops.InstanceNorm(out_ch, affine=True)
        self.conv2 = ops.Conv(out_ch, out_ch, 3, padding=1, bias=False)
        self.act   = nn.ReLU(inplace=True)
        self.proj  = (
            ops.Conv(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj(x)
        out = self.conv1(self.act(self.norm1(x)))
        out = self.conv2(self.act(self.norm2(out)))
        return out + shortcut


class EncoderStage(nn.Module):
    """MaxPool downsample → PreActResBlock × num_blocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ops: SpatialOps,
        num_blocks: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.pool = ops.MaxPool(2) if downsample else nn.Identity()
        blocks: List[nn.Module] = [PreActResBlock(in_ch, out_ch, ops)]
        for _ in range(num_blocks - 1):
            blocks.append(PreActResBlock(out_ch, out_ch, ops))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pool(x))


class DecoderStage(nn.Module):
    """ConvTranspose upsample → concat skip → PreActResBlock × num_blocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        ops: SpatialOps,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.up    = ops.ConvTranspose(in_ch, in_ch, 2, stride=2)
        # After concat: in_ch + skip_ch
        blocks: List[nn.Module] = [PreActResBlock(in_ch + skip_ch, out_ch, ops)]
        for _ in range(num_blocks - 1):
            blocks.append(PreActResBlock(out_ch, out_ch, ops))
        self.blocks = nn.Sequential(*blocks)
        self._ops = ops

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = self._ops.interpolate(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Full ACAPULCO parcellating U-Net
# ---------------------------------------------------------------------------

@register_model("acapulco")
class ACAPULCONet(nn.Module):
    """ACAPULCO parcellating U-Net (pre-activation ResNet + instance norm).

    Config keys (under ``model``):
        enc_channels (list[int]): Channel widths per encoder stage.
                                  Default: [32, 64, 128, 256].
        blocks_per_stage (int):   ResBlocks per encoder/decoder stage.
                                  Default: 2.
    ``dataset.spatial_dims`` selects 2D or 3D.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        ops     = get_ops(cfg.dataset.spatial_dims)
        in_ch   = cfg.dataset.channels
        num_cls = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        # Default channel widths tuned to ~17 M params with 4 downsamples each:
        #   2D (4 enc stages + bottleneck pool = 4 pools, input 80×80→5×5):
        #       [44, 88, 176, 240]  → ~17.4 M
        #   3D (4 enc stages + bottleneck pool = 4 pools, input 80×80×96→5×5×6):
        #       [26, 52, 104, 140]  → ~17.1 M
        default_chs = [44, 88, 176, 240] if ops.dims == 2 else [26, 52, 104, 140]
        chs: List[int] = list(getattr(cfg.model, "enc_channels", default_chs))
        n_blk: int = int(getattr(cfg.model, "blocks_per_stage", 2))

        # Stem: project input channels to chs[0]
        self.stem = ops.Conv(in_ch, chs[0], 3, padding=1, bias=False)

        # Encoder (first stage has no downsampling — stem is at full res)
        self.enc_stages: nn.ModuleList = nn.ModuleList()
        ch_in = chs[0]
        for i, ch_out in enumerate(chs):
            self.enc_stages.append(
                EncoderStage(ch_in, ch_out, ops, n_blk, downsample=(i > 0))
            )
            ch_in = ch_out

        # Bottleneck
        bot_ch = chs[-1] * 2
        self.bottleneck = nn.Sequential(
            ops.MaxPool(2),
            PreActResBlock(chs[-1], bot_ch, ops),
            PreActResBlock(bot_ch, bot_ch, ops),
        )

        # Decoder (reversed)
        self.dec_stages: nn.ModuleList = nn.ModuleList()
        dec_in = bot_ch
        for ch in reversed(chs):
            self.dec_stages.append(
                DecoderStage(dec_in, ch, ch, ops, n_blk)
            )
            dec_in = ch

        # Head
        self.head = ops.Conv(chs[0], num_cls, 1)

        logger.info(
            f"ACAPULCONet {ops.dims}D | enc_channels={chs} | "
            f"bottleneck={bot_ch} | blocks/stage={n_blk} | "
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
