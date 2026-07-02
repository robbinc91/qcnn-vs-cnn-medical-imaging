"""Quaternion MipaimUNet for 2-D or 3-D segmentation.

Quaternion port of :class:`MipaimUNet` (Medical Image Parcellation with
Attention and Inception Modules).  The reduced-dimension inception blocks and
CBAM skip refinement are preserved, with every convolution replaced by a
:class:`QuaternionConv`.  InstanceNorm / ReLU and the CBAM gate act "split" over
the ``4 * Q`` real channels; the four inception branches are merged with
:func:`quaternion_cat`.

``filters_dim`` are REAL widths (÷4 = quaternion); defaults are sized to the
~17 M field.  Quaternion channel counts are aligned up to a multiple of 4 inside
each inception block (so each of the four branches gets an integer width).
"""

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn

from .._ops import SpatialOps, get_ops
from ..registry import register_model
from ._quaternion_ops import QuaternionConv, quaternion_cat
from ._quaternion_blocks import QuaternionUp, QuaternionCBAM

logger = logging.getLogger(__name__)


def _qnorm(q: int, ops: SpatialOps, instance_norm: bool) -> nn.Module:
    return ops.InstanceNorm(4 * q, affine=True) if instance_norm else nn.Identity()


class QReducedDimInceptionBlock(nn.Module):
    """Quaternion reduced-dimension inception block (4 branches → quaternion_cat)."""

    def __init__(
        self, in_q: int, out_q: int, ops: SpatialOps, only_3x3: bool, instance_norm: bool
    ) -> None:
        super().__init__()
        assert out_q % 4 == 0, "inception out_q must be divisible by 4"
        b = out_q // 4
        self._b = b

        self.b1 = nn.Sequential(
            QuaternionConv(in_q, b, ops, 1, padding=0, bias=False),
            _qnorm(b, ops, instance_norm), nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            QuaternionConv(in_q, b, ops, 1, padding=0, bias=False), nn.ReLU(inplace=True),
            QuaternionConv(b, b, ops, 3, padding=1, bias=False),
            _qnorm(b, ops, instance_norm), nn.ReLU(inplace=True),
        )
        k = 3 if only_3x3 else 5
        self.b3 = nn.Sequential(
            QuaternionConv(in_q, b, ops, 1, padding=0, bias=False), nn.ReLU(inplace=True),
            QuaternionConv(b, b, ops, k, padding=k // 2, bias=False),
            _qnorm(b, ops, instance_norm), nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            ops.MaxPool(3, stride=1, padding=1),
            QuaternionConv(in_q, b, ops, 1, padding=0, bias=False),
            _qnorm(b, ops, instance_norm), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self._b
        return quaternion_cat(
            [self.b1(x), self.b2(x), self.b3(x), self.b4(x)], [b, b, b, b]
        )


class QSimpleConvBlock(nn.Module):
    """Quaternion double-conv block (force_unet mode / conv skip refinement)."""

    def __init__(self, in_q: int, out_q: int, ops: SpatialOps, instance_norm: bool) -> None:
        super().__init__()
        self.net = nn.Sequential(
            QuaternionConv(in_q, out_q, ops, 3, padding=1, bias=False),
            _qnorm(out_q, ops, instance_norm), nn.ReLU(inplace=True),
            QuaternionConv(out_q, out_q, ops, 3, padding=1, bias=False),
            _qnorm(out_q, ops, instance_norm), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_block(in_q, out_q, ops, only_3x3, instance_norm, force_unet) -> nn.Module:
    if force_unet:
        return QSimpleConvBlock(in_q, out_q, ops, instance_norm)
    aligned = ((out_q + 3) // 4) * 4
    block: nn.Module = QReducedDimInceptionBlock(in_q, aligned, ops, only_3x3, instance_norm)
    if aligned != out_q:
        block = nn.Sequential(
            block,
            QuaternionConv(aligned, out_q, ops, 1, padding=0, bias=False),
            _qnorm(out_q, ops, instance_norm), nn.ReLU(inplace=True),
        )
    return block


def _make_skip_refinement(q, n_iters, method, ops, only_3x3, instance_norm, force_unet):
    if n_iters == 0:
        return None
    layers: List[nn.Module] = []
    for _ in range(n_iters):
        if method == "attention":
            layers.append(QuaternionCBAM(q, ops))
        elif method == "conv":
            layers.append(QSimpleConvBlock(q, q, ops, instance_norm))
        else:
            layers.append(_make_block(q, q, ops, only_3x3, instance_norm, force_unet))
    return nn.Sequential(*layers)


class QInceptionEncoder(nn.Module):
    def __init__(self, in_q, q_filters, ops, only_3x3, instance_norm,
                 skip_treatment, skip_method, force_unet) -> None:
        super().__init__()
        assert len(q_filters) == 5
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.skip_refinements = nn.ModuleList()
        ch_in = in_q
        for i, f in enumerate(q_filters):
            self.blocks.append(_make_block(ch_in, f, ops, only_3x3, instance_norm, force_unet))
            if i < 4:
                self.pools.append(ops.MaxPool(2, stride=2))
            ref = _make_skip_refinement(f, skip_treatment, skip_method, ops,
                                        only_3x3, instance_norm, force_unet)
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


class QInceptionDecoder(nn.Module):
    def __init__(self, q_filters, real_filters, num_classes, ops,
                 only_3x3, instance_norm, force_unet) -> None:
        super().__init__()
        assert len(q_filters) == 5
        self.bottleneck_block = _make_block(
            q_filters[4], q_filters[4], ops, only_3x3, instance_norm, force_unet
        )
        self.upsamplers = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()
        self._skip_q: List[int] = []
        for i in range(3, -1, -1):
            self.upsamplers.append(QuaternionUp(q_filters[i + 1], q_filters[i], ops))
            self.merge_blocks.append(
                _make_block(q_filters[i] * 2, q_filters[i], ops, only_3x3, instance_norm, force_unet)
            )
            self._skip_q.append(q_filters[i])
        self.head = ops.Conv(real_filters[0], num_classes, 1)

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        x = self.bottleneck_block(skips[4])
        for i, (up, merge) in enumerate(zip(self.upsamplers, self.merge_blocks)):
            skip = skips[3 - i]
            q = self._skip_q[i]
            x = up(x, skip.shape[2:])
            x = quaternion_cat([skip, x], [q, q])
            x = merge(x)
        return self.head(x)


@register_model("quaternion_mipaim")
class QuaternionMipaimUNet(nn.Module):
    """Quaternion MipaimUNet (inception + CBAM attention), 2-D / 3-D.

    Config keys (under ``model``):
        filters_dim (list[int]): 5 REAL widths per level (÷4 = quaternion).
            Auto-selected per spatial_dims if omitted:
              2D: [128, 256, 512, 1152, 2048]  → ~17.4 M
              3D: [88, 176, 352, 704, 1408]    → ~16.5 M
        only_3x3_filters (bool): only 3x3 convs. Default: True.
        instance_normalization (bool): split InstanceNorm. Default: True.
        skip_connections_treatment_number (int): skip refinement passes. Default: 3.
        skip_connections_method (str): 'attention' | 'conv' | 'inception'. Default: 'attention'.
        force_unet (bool): plain conv instead of inception. Default: False.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        m_cfg = cfg.model
        ops = get_ops(cfg.dataset.spatial_dims)
        in_ch = cfg.dataset.channels
        num_cls = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        _default = (
            [128, 256, 512, 1152, 2048] if ops.dims == 2
            else [88, 176, 352, 704, 1408]
        )
        filters: List[int] = list(m_cfg.get("filters_dim", _default))
        only_3x3: bool = m_cfg.get("only_3x3_filters", True)
        inst_norm: bool = m_cfg.get("instance_normalization", True)
        skip_treat: int = m_cfg.get("skip_connections_treatment_number", 3)
        skip_method: str = m_cfg.get("skip_connections_method", "attention")
        force_unet: bool = m_cfg.get("force_unet", False)

        assert len(filters) == 5, "filters_dim must have exactly 5 values"
        assert all(f % 4 == 0 for f in filters), \
            f"filters_dim must be divisible by 4, got {filters}"
        q_filters: List[int] = [f // 4 for f in filters]

        # Real stem lifts input into quaternion space at level-0 width.
        self.input_stem = ops.Conv(in_ch, filters[0], 3, padding=1, bias=False)

        self.encoder = QInceptionEncoder(
            q_filters[0], q_filters, ops, only_3x3, inst_norm,
            skip_treat, skip_method, force_unet,
        )
        self.decoder = QInceptionDecoder(
            q_filters, filters, num_cls, ops, only_3x3, inst_norm, force_unet,
        )

        logger.info(
            f"QuaternionMipaimUNet {ops.dims}D | real_filters={filters} | "
            f"quaternion_filters={q_filters} | only_3x3={only_3x3} | "
            f"skip={skip_treat}×{skip_method} | in={in_ch}ch → {num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_stem(x)
        return self.decoder(self.encoder(x))
