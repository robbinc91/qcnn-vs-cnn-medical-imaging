"""Quaternion-valued operations for 2-D / 3-D convolutional networks.

A quaternion feature map with ``Q`` quaternion channels is stored as an ordinary
real tensor with ``4 * Q`` channels in **component-major** layout::

    [ r_0 .. r_{Q-1} | i_0 .. i_{Q-1} | j_0 .. j_{Q-1} | k_0 .. k_{Q-1} ]

Every helper here preserves that layout so quaternion blocks can be stacked
freely.  Convolution is implemented as a single real ``F.conv{2,3}d`` whose
kernel is assembled from the four component weights via the Hamilton product,
which is what gives quaternion CNNs their characteristic 4x weight sharing
(a quaternion conv has 1/4 the parameters of a real conv between the same number
of real channels).

Dimension is selected through the shared :class:`SpatialOps` object from
``_ops.get_ops(spatial_dims)`` so the same code runs in 2-D and 3-D.

References:
  - Parcollet et al., "Quaternion Convolutional Neural Networks" (ECCV 2018)
  - Gaudet & Maida, "Deep Quaternion Networks" (IJCNN 2018)
  - Zhu et al., "Quaternion Convolutional Neural Networks" — Hamilton kernel form
"""

import logging
import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._ops import SpatialOps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def quaternion_init_(
    r: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    criterion: str = "he",
) -> None:
    """In-place quaternion weight init (Parcollet et al.).

    Each weight is drawn as ``modulus * (cos φ + sin φ · u)`` where ``u`` is a
    random unit pure quaternion, ``φ`` a uniform phase, and ``modulus`` follows a
    chi distribution with 4 dof scaled so the layer respects the He/Glorot
    variance criterion.  ``chi(4)`` is generated as the L2 norm of four standard
    normals, avoiding a SciPy dependency.
    """
    out_q, in_q = r.shape[0], r.shape[1]
    receptive = 1
    for d in r.shape[2:]:
        receptive *= d
    fan_in = in_q * receptive
    fan_out = out_q * receptive
    if criterion == "glorot":
        s = 1.0 / math.sqrt(2.0 * (fan_in + fan_out))
    elif criterion == "he":
        s = 1.0 / math.sqrt(2.0 * fan_in)
    else:
        raise ValueError(f"Unknown init criterion: {criterion}")

    shape = r.shape
    # modulus ~ chi(4) scaled by s  ==  s * ||N(0,1)^4||_2
    modulus = s * torch.randn(4, *shape).pow(2).sum(dim=0).sqrt()
    phase = (torch.rand(shape) * 2.0 - 1.0) * math.pi
    # random unit pure-quaternion axis (v_i, v_j, v_k) per weight element
    v = torch.randn(3, *shape)
    v = v / (v.pow(2).sum(dim=0, keepdim=True).sqrt() + 1e-12)

    with torch.no_grad():
        r.copy_(modulus * torch.cos(phase))
        i.copy_(modulus * v[0] * torch.sin(phase))
        j.copy_(modulus * v[1] * torch.sin(phase))
        k.copy_(modulus * v[2] * torch.sin(phase))


# ---------------------------------------------------------------------------
# Quaternion convolution
# ---------------------------------------------------------------------------

class QuaternionConv(nn.Module):
    """Quaternion convolution for 2-D or 3-D inputs.

    Maps ``in_q`` quaternion channels (``4 * in_q`` real) to ``out_q`` quaternion
    channels (``4 * out_q`` real).  The four learnable component kernels are
    combined into one real kernel through the Hamilton product, then applied with
    a single ``F.conv2d`` / ``F.conv3d`` call.

    Hamilton real-matrix form of ``W ⊗ X`` (W = r + iI + jJ + kK)::

        [ r  -i  -j  -k ]
        [ i   r  -k   j ]
        [ j   k   r  -i ]
        [ k  -j   i   r ]
    """

    def __init__(
        self,
        in_q: int,
        out_q: int,
        ops: SpatialOps,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        ksize = (kernel_size,) * ops.dims

        self.r = nn.Parameter(torch.empty(out_q, in_q, *ksize))
        self.i = nn.Parameter(torch.empty(out_q, in_q, *ksize))
        self.j = nn.Parameter(torch.empty(out_q, in_q, *ksize))
        self.k = nn.Parameter(torch.empty(out_q, in_q, *ksize))
        self.bias = nn.Parameter(torch.zeros(4 * out_q)) if bias else None

        self._conv = F.conv2d if ops.dims == 2 else F.conv3d
        quaternion_init_(self.r, self.i, self.j, self.k)

    def _hamilton_kernel(self) -> torch.Tensor:
        cat_r = torch.cat([self.r, -self.i, -self.j, -self.k], dim=1)
        cat_i = torch.cat([self.i,  self.r, -self.k,  self.j], dim=1)
        cat_j = torch.cat([self.j,  self.k,  self.r, -self.i], dim=1)
        cat_k = torch.cat([self.k, -self.j,  self.i,  self.r], dim=1)
        return torch.cat([cat_r, cat_i, cat_j, cat_k], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._hamilton_kernel()
        return self._conv(
            x, weight, self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation,
        )


# ---------------------------------------------------------------------------
# Quaternion-aware concat (for U-Net skip connections)
# ---------------------------------------------------------------------------

def quaternion_cat(tensors: Sequence[torch.Tensor], q_sizes: Sequence[int]) -> torch.Tensor:
    """Concatenate quaternion tensors while keeping component-major layout.

    A naive ``torch.cat`` along the channel axis would interleave the components
    (r1,i1,j1,k1,r2,...) and corrupt the quaternion structure.  This splits each
    input into its four component blocks and concatenates component-wise, so the
    result is a valid quaternion tensor with ``sum(q_sizes)`` quaternion channels.
    """
    parts: List[List[torch.Tensor]] = [[], [], [], []]
    for t, _q in zip(tensors, q_sizes):
        chunks = torch.chunk(t, 4, dim=1)  # r, i, j, k blocks
        for c in range(4):
            parts[c].append(chunks[c])
    return torch.cat([torch.cat(p, dim=1) for p in parts], dim=1)
