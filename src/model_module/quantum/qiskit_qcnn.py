"""Qiskit hybrid quantum-classical U-Net for 2-D or 3-D segmentation.

This is a *hybrid* model: a classical convolutional U-Net whose deepest features
are modulated by a Qiskit EstimatorQNN. The circuit sees only a global-average-
pooled channel vector (spatial structure discarded) and its output is broadcast
and **added** back to the bottleneck feature map (``x = x + q_feat``). The
quantum branch is thus a global, spatially-uniform channel-recalibration
*side-branch*, NOT a bottleneck the representation flows through; the
convolutional features pass through unchanged. The name is kept for registry
compatibility.

``dataset.spatial_dims`` (2 or 3) controls the encoder/decoder dimensionality;
the EstimatorQNN / TorchConnector branch is dimension-agnostic.
"""

import logging
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .._ops import get_ops
from ..registry import register_model

logger = logging.getLogger(__name__)


def _build_circuit(num_qubits, feature_map_name, ansatz_name, entanglement, reps):
    if feature_map_name == "zz_feature_map":
        feature_map = ZZFeatureMap(
            feature_dimension=num_qubits, reps=2, entanglement=entanglement
        )
    else:
        from qiskit.circuit.library import PauliFeatureMap
        feature_map = PauliFeatureMap(
            feature_dimension=num_qubits, reps=2, entanglement=entanglement
        )

    if ansatz_name == "real_amplitudes":
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    elif ansatz_name == "efficient_su2":
        from qiskit.circuit.library import EfficientSU2
        ansatz = EfficientSU2(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    else:
        from qiskit.circuit.library import TwoLocal
        ansatz = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cz",
            reps=reps,
            entanglement=entanglement,
        )
    return feature_map, ansatz


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


@register_model("qcnn_qiskit")
class QiskitQCNN(nn.Module):
    """Quantum-Bottleneck U-Net using Qiskit + TorchConnector (2-D or 3-D).

    Config keys under ``model``:
        enc_channels (list[int]): Encoder widths. Default: [8, 16, 32].
        quantum.num_qubits (int): Qubits / output dim. Default: 8.
        quantum.feature_map (str): 'zz_feature_map' or 'pauli_feature_map'.
        quantum.ansatz (str): 'real_amplitudes', 'efficient_su2', 'two_local'.
        quantum.entanglement (str): 'circular', 'linear', 'full'.
        quantum.reps (int): Ansatz repetitions. Default: 2.
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

        feature_map, ansatz = _build_circuit(
            num_qubits=nq,
            feature_map_name=q_cfg.feature_map,
            ansatz_name=q_cfg.ansatz,
            entanglement=q_cfg.entanglement,
            reps=q_cfg.reps,
        )
        qc = QuantumCircuit(nq)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        observables = [
            SparsePauliOp.from_sparse_list([("Z", [i], 1.0)], num_qubits=nq)
            for i in range(nq)
        ]
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
            observables=observables,
        )
        self.qnn = TorchConnector(qnn)

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
            f"QiskitQCNN U-Net {ops.dims}D | enc={chs} | qubits={nq} | "
            f"ansatz={q_cfg.ansatz} | out={num_cls} classes"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for enc in self.enc_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        q_in  = torch.tanh(self.q_proj_in(self._ops.global_avg_pool(x))).float() * np.pi
        q_out = self.qnn(q_in)
        q_feat = self._ops.spatial_broadcast(self.q_proj_out(q_out), x)
        x = x + q_feat

        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = self._ops.interpolate(x, skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)
