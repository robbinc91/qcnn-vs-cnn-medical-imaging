"""CerebNet-style segmentation network for 2D or 3D mode.

Based on FastSurferCNN / CerebNet (Faber et al., NeuroImage 2022).
Key design choices:
- Competitive Dense Blocks: 3× (Conv → BN → PReLU) with element-wise
  maxout against the block's projected input at each step.
- Uniform ``filters`` channels throughout (no channel doubling).
- MaxPool (with stored indices) for downsampling.
- MaxUnpool (with stored indices) for upsampling; competitive max-fusion
  replaces skip concatenation in the decoder.
- Kernel 5×5 in 2D (paper), 3×3×3 in 3D (memory-efficient adaptation).

``dataset.spatial_dims`` selects 2D or 3D mode via ``get_ops()``.
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

class CompetitiveDenseBlock(nn.Module):
    """3× (Conv → BN → PReLU) with competitive maxout against the block input.

    At every conv step the output is element-wise max'd with the projected
    block input (``skip``), preserving features across all three layers.
    """

    def __init__(self, in_ch: int, out_ch: int, ops: SpatialOps, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        # 1×1 projection so maxout channels always match
        self.proj = ops.Conv(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.conv1 = ops.Conv(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1   = ops.BatchNorm(out_ch)
        self.act1  = nn.PReLU()

        self.conv2 = ops.Conv(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = ops.BatchNorm(out_ch)
        self.act2  = nn.PReLU()

        self.conv3 = ops.Conv(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn3   = ops.BatchNorm(out_ch)
        self.act3  = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = torch.max(out, skip)

        out = self.act2(self.bn2(self.conv2(out)))
        out = torch.max(out, skip)

        out = self.act3(self.bn3(self.conv3(out)))
        out = torch.max(out, skip)

        return out


class CompetitiveEncoderBlock(nn.Module):
    """CompetitiveDenseBlock followed by MaxPool (with indices)."""

    def __init__(self, in_ch: int, out_ch: int, ops: SpatialOps, kernel_size: int) -> None:
        super().__init__()
        self.dense = CompetitiveDenseBlock(in_ch, out_ch, ops, kernel_size)
        # MaxPool with indices — works for both 2D and 3D
        if ops.dims == 2:
            self.pool: nn.Module = nn.MaxPool2d(2, return_indices=True)
        else:
            self.pool = nn.MaxPool3d(2, return_indices=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (pooled, pre_pool_features, pool_indices)."""
        features = self.dense(x)
        pooled, indices = self.pool(features)
        return pooled, features, indices


class CompetitiveDecoderBlock(nn.Module):
    """MaxUnpool (with indices) + competitive max-fusion + CompetitiveDenseBlock.

    The skip connection (encoder features) is fused via element-wise max
    instead of concatenation, keeping the channel count constant.
    """

    def __init__(self, channels: int, ops: SpatialOps, kernel_size: int) -> None:
        super().__init__()
        if ops.dims == 2:
            self.unpool: nn.Module = nn.MaxUnpool2d(2)
        else:
            self.unpool = nn.MaxUnpool3d(2)
        self.dense = CompetitiveDenseBlock(channels, channels, ops, kernel_size)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        x = self.unpool(x, indices, output_size=skip.shape)
        x = torch.max(x, skip)          # competitive fusion
        return self.dense(x)


# ---------------------------------------------------------------------------
# Full CerebNet model
# ---------------------------------------------------------------------------

@register_model("cerebnet")
class CerebNet(nn.Module):
    """CerebNet-style competitive-dense U-Net, 2D or 3D.

    Config keys (under ``model``):
        filters (int): Uniform channel width. Default 96.
        num_enc_stages (int): Number of encoder stages. Default 4.
    ``dataset.spatial_dims`` selects 2D or 3D ops.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        ops      = get_ops(cfg.dataset.spatial_dims)
        in_ch    = cfg.dataset.channels
        num_cls  = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        filters: int       = int(getattr(cfg.model, "filters", 96))
        num_enc: int       = int(getattr(cfg.model, "num_enc_stages", 4))
        # 5×5 for 2D (paper), 3×3×3 for 3D (memory-efficient)
        kernel_size: int   = 5 if ops.dims == 2 else 3

        # ---- Encoder --------------------------------------------------------
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        ch_in = in_ch
        for _ in range(num_enc):
            self.enc_blocks.append(
                CompetitiveEncoderBlock(ch_in, filters, ops, kernel_size)
            )
            ch_in = filters

        # ---- Bottleneck -----------------------------------------------------
        self.bottleneck = CompetitiveDenseBlock(filters, filters, ops, kernel_size)

        # ---- Decoder --------------------------------------------------------
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(num_enc):
            self.dec_blocks.append(
                CompetitiveDecoderBlock(filters, ops, kernel_size)
            )

        # ---- Head -----------------------------------------------------------
        self.head = ops.Conv(filters, num_cls, 1)

        logger.info(
            f"CerebNet {ops.dims}D | filters={filters} | enc_stages={num_enc} | "
            f"kernel={kernel_size}×{kernel_size}{'×'+str(kernel_size) if ops.dims==3 else ''} | "
            f"in={in_ch}ch → {num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features: List[torch.Tensor] = []
        enc_indices:  List[torch.Tensor] = []

        for enc in self.enc_blocks:
            x, features, indices = enc(x)
            enc_features.append(features)
            enc_indices.append(indices)

        x = self.bottleneck(x)

        for dec, skip, idx in zip(
            self.dec_blocks,
            reversed(enc_features),
            reversed(enc_indices),
        ):
            x = dec(x, skip, idx)

        return self.head(x)
