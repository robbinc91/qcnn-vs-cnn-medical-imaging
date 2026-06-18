"""Hybrid quantum-classical U-Net for 2-D or 3-D segmentation.

Architecturally identical to ``pennylane_qcnn`` (same PennyLane
AngleEmbedding + StronglyEntanglingLayers circuit, same classical conv
encoder/decoder, same additive bottleneck); it differs only in the default
number of ansatz layers. As in the other two models, the quantum circuit sees
only a global-average-pooled channel vector and its output is broadcast and
**added** back to the bottleneck feature map (``x = x + q_feat``) — a global,
spatially-uniform channel-recalibration side-branch, not a true bottleneck.
All three "quantum"/"hybrid" models in this project are hybrid in this sense;
none is a pure quantum network.

``dataset.spatial_dims`` (2 or 3) controls the encoder/decoder dimensionality;
the quantum branch is dimension-agnostic.
"""

import logging
from typing import Any, List

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .._ops import get_ops
from ..registry import register_model

logger = logging.getLogger(__name__)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ops, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            ops.Conv(in_ch, out_ch, 3, padding=1, bias=False),
            ops.BatchNorm(out_ch), nn.ReLU(inplace=True),
            ops.Conv(out_ch, out_ch, 3, padding=1, bias=False),
            ops.BatchNorm(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(ops.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("hybrid_qcnn")
class HybridQCNN(nn.Module):
    """Hybrid Quantum-Classical U-Net (2-D or 3-D).

    Config keys under ``model``:
        classical_encoder.conv_channels (list[int]): Encoder widths.
            Default: [16, 32, 64].
        quantum.num_qubits (int): Qubits / bottleneck dim. Default: 8.
        quantum.num_layers (int): Ansatz repetitions. Default: 4.
        quantum.diff_method (str): PennyLane diff method. Default: 'best'.
        quantum.device (str): PennyLane device. Default: 'default.qubit'.
        classical_decoder.dropout (float): Decoder dropout. Default: 0.2.
    ``dataset.spatial_dims`` selects 2-D or 3-D mode.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        enc_cfg = cfg.model.classical_encoder
        q_cfg   = cfg.model.quantum
        dec_cfg = cfg.model.classical_decoder

        ops  = get_ops(cfg.dataset.spatial_dims)
        chs: List[int] = list(enc_cfg.conv_channels)
        in_ch   = cfg.dataset.channels
        num_cls = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        dp  = float(dec_cfg.dropout)
        nq  = q_cfg.num_qubits

        # Encoder
        self.enc_blocks = nn.ModuleList()
        ch_in = in_ch
        for ch in chs:
            self.enc_blocks.append(_DoubleConv(ch_in, ch, ops))
            ch_in = ch
        self.pool = ops.MaxPool(2)

        # Quantum bottleneck
        self.q_proj_in = nn.Linear(chs[-1], nq)

        dev = qml.device(q_cfg.device, wires=nq)

        @qml.qnode(dev, diff_method=q_cfg.diff_method, interface="torch")
        def quantum_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[float]:
            qml.AngleEmbedding(inputs, wires=range(nq))
            qml.StronglyEntanglingLayers(weights, wires=range(nq))
            return [qml.expval(qml.PauliZ(i)) for i in range(nq)]

        self.quantum_circuit = quantum_circuit

        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=q_cfg.num_layers, n_wires=nq
        )
        self.q_weights  = nn.Parameter(torch.randn(weight_shape) * 0.1)
        self.q_proj_out = nn.Sequential(nn.Linear(nq, chs[-1]), nn.ReLU(inplace=True))

        # Decoder
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch_in = chs[-1]
        for ch in reversed(chs):
            self.up_convs.append(ops.ConvTranspose(ch_in, ch, 2, stride=2))
            self.dec_blocks.append(_DoubleConv(ch * 2, ch, ops, dp))
            ch_in = ch

        self.head = ops.Conv(chs[0], num_cls, 1)
        self._ops = ops

        logger.info(
            f"HybridQCNN U-Net {ops.dims}D | enc={chs} | qubits={nq} | "
            f"layers={q_cfg.num_layers} | out={num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        skips: List[torch.Tensor] = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        q_in  = torch.tanh(self.q_proj_in(self._ops.global_avg_pool(x))).float() * np.pi
        q_out = torch.stack([
            torch.stack(self.quantum_circuit(q_in[i], self.q_weights))
            for i in range(B)
        ]).float()
        q_feat = self._ops.spatial_broadcast(self.q_proj_out(q_out), x)
        x = x + q_feat

        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = self._ops.interpolate(x, skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)
