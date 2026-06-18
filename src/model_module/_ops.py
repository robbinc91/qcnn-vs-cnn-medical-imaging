"""Dimension-agnostic spatial operations helper.

All models import ``get_ops(spatial_dims)`` and use the returned ``SpatialOps``
object instead of hard-coded Conv2d / Conv3d classes.  This lets every model
run in either 2-D (slice-level) or 3-D (volumetric) mode purely by changing
``dataset.spatial_dims`` in the config.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpatialOps:
    dims: int
    Conv: type
    ConvTranspose: type
    MaxPool: type
    BatchNorm: type
    InstanceNorm: type
    Dropout: type
    interp_mode: str        # 'bilinear' | 'trilinear'
    interp_kwargs: dict     # {'align_corners': False}

    # ------------------------------------------------------------------
    # Adaptive pooling
    # ------------------------------------------------------------------
    def adaptive_avg_pool(self, x: torch.Tensor, output_size) -> torch.Tensor:
        if self.dims == 2:
            return F.adaptive_avg_pool2d(x, output_size)
        return F.adaptive_avg_pool3d(x, output_size)

    def adaptive_max_pool(self, x: torch.Tensor, output_size) -> torch.Tensor:
        if self.dims == 2:
            return F.adaptive_max_pool2d(x, output_size)
        return F.adaptive_max_pool3d(x, output_size)

    # ------------------------------------------------------------------
    # Quantum-bottleneck helpers
    # ------------------------------------------------------------------
    def global_avg_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over all spatial dims → (B, C), works for 2-D and 3-D."""
        return x.mean(dim=list(range(2, x.ndim)))

    def spatial_broadcast(self, vec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Reshape (B, C) to (B, C, 1, 1) or (B, C, 1, 1, 1) to match target's ndim."""
        return vec.view(vec.shape[0], vec.shape[1], *([1] * (target.ndim - 2)))

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def interpolate(
        self,
        x: torch.Tensor,
        size: Union[Tuple, torch.Size],
    ) -> torch.Tensor:
        return F.interpolate(x, size=size, mode=self.interp_mode, **self.interp_kwargs)


def get_ops(spatial_dims: int) -> SpatialOps:
    """Return the spatial ops set for ``spatial_dims`` (2 or 3)."""
    if spatial_dims == 2:
        return SpatialOps(
            dims=2,
            Conv=nn.Conv2d,
            ConvTranspose=nn.ConvTranspose2d,
            MaxPool=nn.MaxPool2d,
            BatchNorm=nn.BatchNorm2d,
            InstanceNorm=nn.InstanceNorm2d,
            Dropout=nn.Dropout2d,
            interp_mode="bilinear",
            interp_kwargs={"align_corners": False},
        )
    if spatial_dims == 3:
        return SpatialOps(
            dims=3,
            Conv=nn.Conv3d,
            ConvTranspose=nn.ConvTranspose3d,
            MaxPool=nn.MaxPool3d,
            BatchNorm=nn.BatchNorm3d,
            InstanceNorm=nn.InstanceNorm3d,
            Dropout=nn.Dropout3d,
            interp_mode="trilinear",
            interp_kwargs={"align_corners": False},
        )
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")
