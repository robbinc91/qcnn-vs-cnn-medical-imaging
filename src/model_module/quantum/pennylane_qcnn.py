"""PennyLane hybrid quantum-classical U-Net for 2-D or 3-D segmentation.

This is a *hybrid* model: a standard classical convolutional U-Net whose
deepest features are modulated by a variational quantum circuit. The circuit
operates on a global-average-pooled channel vector (all spatial structure is
discarded), and its output is broadcast and **added** back to the bottleneck
feature map (``x = x + q_feat``). The quantum branch is therefore a global,
spatially-uniform channel-recalibration *side-branch* (similar in spirit to an
additive Squeeze-and-Excite), NOT a bottleneck that the representation flows
through — the convolutional bottleneck features pass through unchanged and the
network can route around the circuit. The name is kept for registry
compatibility; "quantum bottleneck" should be read in this restricted sense.

``dataset.spatial_dims`` (2 or 3) controls the encoder/decoder dimensionality;
the circuit is dimension-agnostic (one call per sample).
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


class QuantumConvLayer(nn.Module):
    """PennyLane variational circuit (dimension-agnostic)."""

    def __init__(
        self,
        num_qubits: int,
        entanglement: str = "circular",
        ansatz: str = "strongly_entangling",
        diff_method: str = "best",
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.ansatz     = ansatz

        dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[float]:
            qml.AngleEmbedding(inputs, wires=range(num_qubits))
            if ansatz == "strongly_entangling":
                qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            elif ansatz == "basic_entangler":
                qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            else:
                for lw in weights:
                    for i in range(num_qubits):
                        qml.RY(lw[i, 0], wires=i)
                        qml.RZ(lw[i, 1], wires=i)
                    for i in range(num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if entanglement == "circular" and num_qubits > 2:
                        qml.CNOT(wires=[num_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def init_weights(self, num_layers: int) -> nn.Parameter:
        if self.ansatz == "strongly_entangling":
            shape = qml.StronglyEntanglingLayers.shape(
                n_layers=num_layers, n_wires=self.num_qubits
            )
        elif self.ansatz == "basic_entangler":
            shape = qml.BasicEntanglerLayers.shape(
                n_layers=num_layers, n_wires=self.num_qubits
            )
        else:
            shape = (num_layers, self.num_qubits, 2)
        return nn.Parameter(torch.randn(shape, dtype=torch.float32) * 0.1)


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


@register_model("qcnn_pennylane")
class PennyLaneQCNN(nn.Module):
    """Quantum-Bottleneck U-Net using PennyLane (2-D or 3-D).

    Config keys under ``model``:
        enc_channels (list[int]): Encoder widths. Default: [8, 16, 32].
        quantum.num_qubits (int): Qubits / bottleneck dim. Default: 8.
        quantum.num_layers (int): Ansatz layers. Default: 4.
        quantum.entanglement (str): 'linear' or 'circular'.
        quantum.ansatz (str): 'strongly_entangling', 'basic_entangler', 'custom'.
        quantum.diff_method (str): PennyLane diff method. Default: 'best'.
        quantum.device (str): PennyLane device. Default: 'default.qubit'.
        decoder_dropout (float): Decoder dropout. Default: 0.2.
    ``dataset.spatial_dims`` selects 2-D or 3-D mode.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        q_cfg = cfg.model.quantum
        ops   = get_ops(cfg.dataset.spatial_dims)
        chs: List[int] = list(cfg.model.get("enc_channels", [8, 16, 32]))
        in_ch   = cfg.dataset.channels
        num_cls = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        dp  = float(cfg.model.get("decoder_dropout", 0.2))
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
        self.q_layer   = QuantumConvLayer(
            num_qubits=nq,
            entanglement=q_cfg.entanglement,
            ansatz=q_cfg.ansatz,
            diff_method=q_cfg.diff_method,
            device_name=q_cfg.device,
        )
        self.q_weights  = self.q_layer.init_weights(q_cfg.num_layers)
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
            f"PennyLaneQCNN U-Net {ops.dims}D | enc={chs} | qubits={nq} | "
            f"layers={q_cfg.num_layers} | out={num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        skips: List[torch.Tensor] = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Quantum bottleneck — dimension-agnostic global avg pool
        q_in  = torch.tanh(self.q_proj_in(self._ops.global_avg_pool(x))).float() * np.pi
        q_out = torch.stack([
            torch.stack(self.q_layer.circuit(q_in[i], self.q_weights))
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
