"""SwinUNETR wrapper for 2-D or 3-D segmentation.

Uses MONAI's production SwinUNETR which natively supports ``spatial_dims=2``
and ``spatial_dims=3`` via the same API.  The forward method auto-pads the
input to the required multiple of 2^(stages+1) and crops the output back.

Reference: Hatamizadeh et al., MICCAI 2022.  https://arxiv.org/abs/2201.01266
"""

import logging
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_model

logger = logging.getLogger(__name__)


@register_model("swin_unetr")
class SwinUNETR(nn.Module):
    """MONAI SwinUNETR for 2-D or 3-D segmentation.

    Config keys under ``model``:
        feature_size (int): Base embedding dimension. Default: 24.
        depths (list[int]): Swin blocks per stage. Default: [2, 2, 2, 2].
        num_heads (list[int]): Attention heads per stage. Default: [3, 6, 12, 24].
        window_size (int): Attention window size (applies to each spatial dim). Default: 7.
        drop_rate (float): Dropout in attention/MLP. Default: 0.0.
        attn_drop_rate (float): Dropout on attention weights. Default: 0.0.
        use_checkpoint (bool): Gradient checkpointing. Default: True.
    ``dataset.spatial_dims`` selects 2-D or 3-D mode.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        try:
            from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
        except ImportError as exc:
            raise ImportError(
                "MONAI is required for SwinUNETR. Install: pip install 'monai[einops]'"
            ) from exc

        m_cfg          = cfg.model
        spatial_dims   = cfg.dataset.spatial_dims
        in_channels    = cfg.dataset.channels
        num_classes    = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )

        # Auto-select ~17 M defaults per spatial_dims if not set in config:
        #   2D: feature_size=36, depths=[2,4,4,4] → ~16.8 M
        #   3D: feature_size=24, depths=[2,4,4,4] → ~17.1 M
        _default_fs     = 36 if spatial_dims == 2 else 24
        _default_depths = [2, 4, 4, 4]
        feature_size: int    = m_cfg.get("feature_size", _default_fs)
        depths: List[int]    = list(m_cfg.get("depths", _default_depths))
        num_heads: List[int] = list(m_cfg.get("num_heads", [3, 6, 12, 24]))
        window_size: int     = m_cfg.get("window_size", 7)
        drop_rate: float     = m_cfg.get("drop_rate", 0.0)
        attn_drop: float     = m_cfg.get("attn_drop_rate", 0.0)
        use_ckpt: bool       = m_cfg.get("use_checkpoint", True)

        self.model = MonaiSwinUNETR(
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop,
            use_checkpoint=use_ckpt,
            spatial_dims=spatial_dims,
        )

        self._spatial_dims = spatial_dims
        # divisor = 2^(num_stages+1); for 4-stage model: 2^5 = 32
        self._divisor = 2 ** (self.model.swinViT.num_layers + 1)

        logger.info(
            f"SwinUNETR (MONAI) {spatial_dims}D | "
            f"in={in_channels}ch → out={num_classes} classes | "
            f"feature_size={feature_size} | depths={depths} | window={window_size}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auto-pad spatial dims to the next multiple of divisor, then crop."""
        orig_shape = x.shape[2:]          # spatial dims only
        d = self._divisor

        # Compute padding for each spatial dim (last dim first for F.pad)
        pads = []
        for s in reversed(orig_shape):
            pads += [0, (-s % d)]         # (pad_before=0, pad_after)
        if any(p > 0 for p in pads):
            x = F.pad(x, pads)

        out = self.model(x)

        # Crop back to original spatial size
        slices = [slice(None), slice(None)] + [slice(0, s) for s in orig_shape]
        return out[slices]
