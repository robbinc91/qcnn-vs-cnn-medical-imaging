"""PyTorch port of mipaim_unet for 2-D or 3-D segmentation.

Original: https://github.com/robbinc91/mipaim_unet (TensorFlow/Keras)

``dataset.spatial_dims`` (2 or 3) selects all conv/pool/norm ops via the
shared ``_ops.get_ops()`` helper.  CBAM, Inception blocks, encoder, and
decoder all adapt automatically.
"""

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._ops import SpatialOps, get_ops
from ..registry import register_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _norm(channels: int, ops: SpatialOps, instance_norm: bool) -> nn.Module:
    return ops.InstanceNorm(channels, affine=True) if instance_norm else nn.Identity()


class ReducedDimInceptionBlock(nn.Module):
    """Reduced-dimension inception block (dimension-agnostic).

    Four parallel branches each producing ``out_ch // 4`` channels:
    1×1, 1×1→3×3, 1×1→3×3 (or 5×5), MaxPool(3,pad=1)→1×1.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ops: SpatialOps,
        only_3x3: bool = True,
        instance_norm: bool = True,
    ) -> None:
        super().__init__()
        assert out_ch % 4 == 0
        b = out_ch // 4

        self.b1 = nn.Sequential(
            ops.Conv(in_ch, b, 1, bias=False),
            _norm(b, ops, instance_norm),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            ops.Conv(in_ch, b, 1, bias=False),
            nn.ReLU(inplace=True),
            ops.Conv(b, b, 3, padding=1, bias=False),
            _norm(b, ops, instance_norm),
            nn.ReLU(inplace=True),
        )
        k = 3 if only_3x3 else 5
        self.b3 = nn.Sequential(
            ops.Conv(in_ch, b, 1, bias=False),
            nn.ReLU(inplace=True),
            ops.Conv(b, b, k, padding=k // 2, bias=False),
            _norm(b, ops, instance_norm),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            ops.MaxPool(3, stride=1, padding=1),
            ops.Conv(in_ch, b, 1, bias=False),
            _norm(b, ops, instance_norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (dimension-agnostic).

    Applies channel attention then spatial attention (Woo et al. 2018).
    """

    def __init__(self, channels: int, ops: SpatialOps, ratio: int = 8) -> None:
        super().__init__()
        self._ops = ops
        mid = max(1, channels // ratio)
        self.ch_fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        # Spatial attention conv: 2 input channels (avg + max), kernel 7
        self.sp_conv = ops.Conv(2, 1, kernel_size=7, padding=3, bias=False)

        # For view in channel attention
        self._extra_dims = tuple([1] * ops.dims)  # (1,1) or (1,1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]

        # Channel attention
        avg_c = self._ops.adaptive_avg_pool(x, 1).view(B, C)
        max_c = self._ops.adaptive_max_pool(x, 1).view(B, C)
        ch_att = torch.sigmoid(self.ch_fc(avg_c) + self.ch_fc(max_c))
        ch_att = ch_att.view(B, C, *self._extra_dims)
        x = x * ch_att

        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sp_att = torch.sigmoid(self.sp_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_att


class SimpleConvBlock(nn.Module):
    """Plain double-conv block (used in force_unet mode)."""

    def __init__(self, in_ch: int, out_ch: int, ops: SpatialOps, instance_norm: bool) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ops.Conv(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch, ops, instance_norm),
            nn.ReLU(inplace=True),
            ops.Conv(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch, ops, instance_norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_block(in_ch, out_ch, ops, only_3x3, instance_norm, force_unet):
    if force_unet:
        return SimpleConvBlock(in_ch, out_ch, ops, instance_norm)
    out_ch_aligned = ((out_ch + 3) // 4) * 4
    block: nn.Module = ReducedDimInceptionBlock(
        in_ch, out_ch_aligned, ops, only_3x3, instance_norm
    )
    if out_ch_aligned != out_ch:
        block = nn.Sequential(
            block,
            ops.Conv(out_ch_aligned, out_ch, 1, bias=False),
            _norm(out_ch, ops, instance_norm),
            nn.ReLU(inplace=True),
        )
    return block


def _make_skip_refinement(channels, n_iters, method, ops, only_3x3, instance_norm, force_unet):
    if n_iters == 0:
        return None
    layers: List[nn.Module] = []
    for _ in range(n_iters):
        if method == "attention":
            layers.append(CBAMBlock(channels, ops))
        elif method == "conv":
            layers.append(SimpleConvBlock(channels, channels, ops, instance_norm))
        else:
            layers.append(_make_block(channels, channels, ops, only_3x3, instance_norm, force_unet))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class InceptionEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        filters,
        ops,
        only_3x3,
        instance_norm,
        skip_treatment,
        skip_method,
        force_unet,
    ) -> None:
        super().__init__()
        assert len(filters) == 5
        self.blocks            = nn.ModuleList()
        self.pools             = nn.ModuleList()
        self.skip_refinements  = nn.ModuleList()

        ch_in = in_channels
        for i, f in enumerate(filters):
            self.blocks.append(_make_block(ch_in, f, ops, only_3x3, instance_norm, force_unet))
            if i < 4:
                self.pools.append(ops.MaxPool(2, stride=2))
            ref = _make_skip_refinement(
                f, skip_treatment, skip_method, ops, only_3x3, instance_norm, force_unet
            )
            self.skip_refinements.append(ref if ref is not None else nn.Identity())
            ch_in = f

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips: List[torch.Tensor] = []
        pool_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(self.skip_refinements[i](x))
            if i < 4:
                x = self.pools[pool_idx](x)
                pool_idx += 1
        return skips


class InceptionDecoderV2(nn.Module):
    def __init__(self, filters, num_classes, ops, only_3x3, instance_norm, force_unet) -> None:
        super().__init__()
        assert len(filters) == 5
        self._ops = ops

        self.bottleneck_block = _make_block(
            filters[4], filters[4], ops, only_3x3, instance_norm, force_unet
        )
        self.upsamplers  = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(3, -1, -1):
            self.upsamplers.append(
                ops.ConvTranspose(filters[i + 1], filters[i], kernel_size=2, stride=2, bias=False)
            )
            self.merge_blocks.append(
                _make_block(filters[i] * 2, filters[i], ops, only_3x3, instance_norm, force_unet)
            )

        self.head = ops.Conv(filters[0], num_classes, 1)

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        x = self.bottleneck_block(skips[4])
        for i, (up, merge) in enumerate(zip(self.upsamplers, self.merge_blocks)):
            skip = skips[3 - i]
            x = up(x)
            if x.shape != skip.shape:
                x = self._ops.interpolate(x, skip.shape[2:])
            x = merge(torch.cat([skip, x], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("mipaim_unet")
class MipaimUNet(nn.Module):
    """Medical Image Parcellation with Attention and Inception Modules U-Net.

    PyTorch port of https://github.com/robbinc91/mipaim_unet.
    Supports 2-D and 3-D modes via ``dataset.spatial_dims``.

    Config keys under ``model``:
        filters_dim (list[int]): 5 channel counts per encoder level.
            Default: [8, 16, 32, 64, 128].
        only_3x3_filters (bool): Use only 3×3 convolutions. Default: True.
        instance_normalization (bool): InstanceNorm throughout. Default: True.
        skip_connections_treatment_number (int): Skip refinement passes. Default: 3.
        skip_connections_method (str): 'attention', 'conv', or 'inception'. Default: 'attention'.
        force_unet (bool): Replace inception with plain conv. Default: False.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        m_cfg = cfg.model
        ops   = get_ops(cfg.dataset.spatial_dims)

        in_channels = cfg.dataset.channels
        num_classes = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )

        # Auto-select ~17 M defaults per spatial_dims if not set in config:
        #   2D: [90, 180, 360, 720, 1440] → ~17.3 M
        #   3D: [64, 128, 256, 512, 1024] → ~17.8 M
        _default_filters = (
            [90, 180, 360, 720, 1440] if ops.dims == 2 else [64, 128, 256, 512, 1024]
        )
        filters: List[int]  = list(m_cfg.get("filters_dim", _default_filters))
        only_3x3: bool      = m_cfg.get("only_3x3_filters", True)
        instance_norm: bool = m_cfg.get("instance_normalization", True)
        skip_treatment: int = m_cfg.get("skip_connections_treatment_number", 3)
        skip_method: str    = m_cfg.get("skip_connections_method", "attention")
        force_unet: bool    = m_cfg.get("force_unet", False)

        assert len(filters) == 5, "filters_dim must have exactly 5 values"

        self.encoder = InceptionEncoder(
            in_channels, filters, ops, only_3x3, instance_norm,
            skip_treatment, skip_method, force_unet,
        )
        self.decoder = InceptionDecoderV2(
            filters, num_classes, ops, only_3x3, instance_norm, force_unet,
        )

        logger.info(
            f"MipaimUNet {ops.dims}D | filters={filters} | only_3x3={only_3x3} | "
            f"skip={skip_treatment}×{skip_method} | in={in_channels}ch → out={num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
