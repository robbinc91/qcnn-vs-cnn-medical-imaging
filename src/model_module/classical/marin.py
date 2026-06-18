"""MARIN-Net: Multi-scale Attended Residual Instance-Normalized Network.

Novel architecture combining the strongest elements of each model in the
brainstem segmentation benchmark:

  ACAPULCO   → pre-activation ordering (IN → PReLU → Conv) and
                instance normalization throughout (intensity-invariant MRI)
  MipaimUNet → CBAM channel+spatial attention gates on every skip connection
  CerebNet   → competitive element-wise maxout at the bottleneck (max instead
                of + for the residual shortcut, preserving the strongest
                abstract features at the deepest level)
  All above  → multi-scale dilated parallel branches: two Conv branches
                (dilation 1 and 2) fused before the integration conv,
                giving a 3×3 and 5×5 effective receptive field simultaneously
                with the same parameter count as a single 3×3 conv layer

Core building block — MultiScalePreActBlock:
  pre-norm/act → [Conv_dil1 ‖ Conv_dil2] → concat → pre-norm/act →
  integration Conv → residual + shortcut

Bottleneck variant — CompetitiveMultiScaleBlock:
  identical structure but residual is max(out, proj(x)) instead of + .

Skip gates — CBAMGate (Woo et al., 2018):
  channel attention (MLP on global avg+max) ×
  spatial attention (7×7 Conv on per-channel avg+max)

Works for 2D and 3D via ``get_ops(dataset.spatial_dims)``.
"""

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from .._ops import SpatialOps, get_ops
from ..registry import register_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MultiScalePreActBlock(nn.Module):
    """Pre-activation block with dual dilated branches and additive residual.

    Parallel branches (dilation 1 and 2) give local + slightly-wider context
    at no extra parameter cost vs a standard pre-act ResBlock.
    """

    def __init__(self, in_ch: int, out_ch: int, ops: SpatialOps) -> None:
        super().__init__()
        half = out_ch // 2

        # Pre-activation on input
        self.norm1 = ops.InstanceNorm(in_ch, affine=True)
        self.act1  = nn.PReLU(in_ch)

        # Parallel multi-scale branches (dil=1: local, dil=2: wider context)
        self.branch_local = ops.Conv(in_ch, half,       3, padding=1, bias=False)
        self.branch_wide  = ops.Conv(in_ch, out_ch - half, 3, padding=2,
                                     dilation=2, bias=False)

        # Pre-activation on merged features + integration conv
        self.norm2    = ops.InstanceNorm(out_ch, affine=True)
        self.act2     = nn.PReLU(out_ch)
        self.integrate = ops.Conv(out_ch, out_ch, 3, padding=1, bias=False)

        # Residual shortcut
        self.proj = (ops.Conv(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_act = self.act1(self.norm1(x))
        x_ms  = torch.cat([self.branch_local(x_act),
                            self.branch_wide(x_act)], dim=1)
        out   = self.integrate(self.act2(self.norm2(x_ms)))
        return out + self.proj(x)


class CompetitiveMultiScaleBlock(nn.Module):
    """Like MultiScalePreActBlock but uses element-wise max for the shortcut.

    Used at the bottleneck: competition between the transformed features and
    the projected input preserves the strongest activations at the deepest
    abstract level (from CerebNet / FastSurferCNN).
    """

    def __init__(self, in_ch: int, out_ch: int, ops: SpatialOps) -> None:
        super().__init__()
        half = out_ch // 2

        self.norm1        = ops.InstanceNorm(in_ch, affine=True)
        self.act1         = nn.PReLU(in_ch)
        self.branch_local = ops.Conv(in_ch, half,       3, padding=1, bias=False)
        self.branch_wide  = ops.Conv(in_ch, out_ch - half, 3, padding=2,
                                     dilation=2, bias=False)
        self.norm2        = ops.InstanceNorm(out_ch, affine=True)
        self.act2         = nn.PReLU(out_ch)
        self.integrate    = ops.Conv(out_ch, out_ch, 3, padding=1, bias=False)
        self.proj         = (ops.Conv(in_ch, out_ch, 1, bias=False)
                             if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_act = self.act1(self.norm1(x))
        x_ms  = torch.cat([self.branch_local(x_act),
                            self.branch_wide(x_act)], dim=1)
        out   = self.integrate(self.act2(self.norm2(x_ms)))
        return torch.max(out, self.proj(x))          # competitive max (CerebNet)


class CBAMGate(nn.Module):
    """Convolutional Block Attention Module gate (Woo et al. 2018).

    Applies channel attention then spatial attention; dimension-agnostic.
    Used on skip connections so the decoder attends to the most relevant
    encoder features (from MipaimUNet).
    """

    def __init__(self, channels: int, ops: SpatialOps, ratio: int = 8) -> None:
        super().__init__()
        self._ops  = ops
        mid        = max(1, channels // ratio)
        self.ch_fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sp_conv = ops.Conv(2, 1, 7, padding=3, bias=False)
        self._extra  = tuple([1] * ops.dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        # Channel attention
        avg_c  = self._ops.adaptive_avg_pool(x, 1).view(B, C)
        max_c  = self._ops.adaptive_max_pool(x, 1).view(B, C)
        ch_att = torch.sigmoid(self.ch_fc(avg_c) + self.ch_fc(max_c))
        x      = x * ch_att.view(B, C, *self._extra)
        # Spatial attention
        avg_s  = x.mean(dim=1, keepdim=True)
        max_s  = x.max(dim=1, keepdim=True).values
        sp_att = torch.sigmoid(self.sp_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_att


class MARINEncoderStage(nn.Module):
    """Optional MaxPool → MultiScalePreActBlock × n → CBAMGate.

    Returns both the raw block output (fed to the next stage) and the
    CBAM-attended version (stored as skip for the decoder).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ops: SpatialOps,
        n_blocks: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.pool = ops.MaxPool(2) if downsample else nn.Identity()
        blocks: List[nn.Module] = [MultiScalePreActBlock(in_ch, out_ch, ops)]
        for _ in range(n_blocks - 1):
            blocks.append(MultiScalePreActBlock(out_ch, out_ch, ops))
        self.blocks = nn.Sequential(*blocks)
        self.cbam   = CBAMGate(out_ch, ops)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (block_output, cbam_attended_skip)."""
        x    = self.blocks(self.pool(x))
        skip = self.cbam(x)
        return x, skip


class MARINDecoderStage(nn.Module):
    """ConvTranspose upsample → concat attended skip → MultiScalePreActBlock × n."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        ops: SpatialOps,
        n_blocks: int,
    ) -> None:
        super().__init__()
        self.up   = ops.ConvTranspose(in_ch, in_ch, 2, stride=2)
        blocks: List[nn.Module] = [
            MultiScalePreActBlock(in_ch + skip_ch, out_ch, ops)
        ]
        for _ in range(n_blocks - 1):
            blocks.append(MultiScalePreActBlock(out_ch, out_ch, ops))
        self.blocks = nn.Sequential(*blocks)
        self._ops   = ops

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = self._ops.interpolate(x, skip.shape[2:])
        return self.blocks(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Full MARIN-Net
# ---------------------------------------------------------------------------

@register_model("marin")
class MARINNet(nn.Module):
    """MARIN-Net: Multi-scale Attended Residual Instance-Normalized Network.

    Combines:
      - Pre-activation blocks + Instance Norm  (ACAPULCO)
      - CBAM attention gates on skip connections  (MipaimUNet)
      - Competitive maxout at the bottleneck  (CerebNet)
      - Dual dilated branches (dil=1 ‖ dil=2) in every block  (novel fusion)

    Config keys (under ``model``):
        enc_channels (list[int]): Channel widths per encoder stage.
            Auto-selected per spatial_dims if omitted:
              2D: [24, 48, 96, 128]  → ~5 M params
              3D: [24, 48, 96]       → ~7 M params
        blocks_per_stage (int): MultiScalePreActBlocks per stage. Default: 2.
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
        # Tuned to ~17 M params with 4 downsamples each:
        #   2D (4 enc stages + bottleneck pool = 4 pools, input 80×80→5×5):
        #       [44, 88, 176, 240] → ~17.4 M
        #   3D (4 enc stages + bottleneck pool = 4 pools, input 80×80×96→5×5×6):
        #       [26, 52, 104, 140] → ~17.1 M
        default_chs = [44, 88, 176, 240] if ops.dims == 2 else [26, 52, 104, 140]
        chs: List[int] = list(getattr(cfg.model, "enc_channels", default_chs))
        n_blk: int     = int(getattr(cfg.model, "blocks_per_stage", 2))

        assert all(c % 2 == 0 for c in chs), \
            "enc_channels must all be even (required by dual-branch block)"

        # Stem: project to first encoder channel width
        self.stem = ops.Conv(in_ch, chs[0], 3, padding=1, bias=False)

        # Encoder
        self.enc_stages: nn.ModuleList = nn.ModuleList()
        ch_in = chs[0]
        for i, ch_out in enumerate(chs):
            self.enc_stages.append(
                MARINEncoderStage(ch_in, ch_out, ops, n_blk, downsample=(i > 0))
            )
            ch_in = ch_out

        # Bottleneck (competitive maxout residual)
        bot_ch = chs[-1] * 2
        self.bottleneck = nn.Sequential(
            ops.MaxPool(2),
            CompetitiveMultiScaleBlock(chs[-1], bot_ch, ops),
            CompetitiveMultiScaleBlock(bot_ch, bot_ch, ops),
        )

        # Decoder
        self.dec_stages: nn.ModuleList = nn.ModuleList()
        dec_in = bot_ch
        for ch in reversed(chs):
            self.dec_stages.append(
                MARINDecoderStage(dec_in, ch, ch, ops, n_blk)
            )
            dec_in = ch

        # Classification head
        self.head = ops.Conv(chs[0], num_cls, 1)

        logger.info(
            f"MARINNet {ops.dims}D | enc_channels={chs} | bottleneck={bot_ch} | "
            f"blocks/stage={n_blk} | in={in_ch}ch → {num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: List[torch.Tensor] = []
        for enc in self.enc_stages:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.dec_stages, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)
