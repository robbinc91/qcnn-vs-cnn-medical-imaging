"""Quantum CNN implementation using Qiskit."""

import logging
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

from ..registry import register_model

logger = logging.getLogger(__name__)


def build_qiskit_circuit(
    num_qubits: int,
    feature_map_name: str,
    ansatz_name: str,
    entanglement: str,
    reps: int,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Build Qiskit feature map and ansatz circuits.

    Returns:
        Tuple of (feature_map, ansatz) quantum circuits.
    """
    # Feature map
    if feature_map_name == "zz_feature_map":
        feature_map = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=2,
            entanglement=entanglement,
        )
    else:
        # Pauli feature map
        from qiskit.circuit.library import PauliFeatureMap
        feature_map = PauliFeatureMap(
            feature_dimension=num_qubits,
            reps=2,
            entanglement=entanglement,
        )

    # Ansatz
    if ansatz_name == "real_amplitudes":
        ansatz = RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    elif ansatz_name == "efficient_su2":
        from qiskit.circuit.library import EfficientSU2
        ansatz = EfficientSU2(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
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


class QiskitClassicalPreprocessor(nn.Module):
    """Classical feature extraction before Qiskit quantum circuit."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        pre_cfg = cfg.model.classical_pre
        in_channels = cfg.dataset.channels

        layers: List[nn.Module] = []
        for out_ch in pre_cfg.conv_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_ch, pre_cfg.kernel_size, padding=1),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_ch

        layers.append(nn.AdaptiveAvgPool2d(pre_cfg.pool_to_size))
        layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

        self.output_dim = pre_cfg.conv_channels[-1] * pre_cfg.pool_to_size ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("qcnn_qiskit")
class QiskitQCNN(nn.Module):
    """Quantum CNN using Qiskit with TorchConnector for hybrid training.

    Architecture:
        [Classical Preprocessor] -> Qiskit QNN -> [Classical Head]
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        q_cfg = cfg.model.quantum

        # Classical preprocessing
        if cfg.model.classical_pre.enabled:
            self.preprocessor = QiskitClassicalPreprocessor(cfg)
            pre_dim = self.preprocessor.output_dim
        else:
            self.preprocessor = None
            pre_dim = cfg.dataset.channels * cfg.dataset.image_size ** 2

        # Linear projection to qubit dimension
        self.project = nn.Linear(pre_dim, q_cfg.num_qubits)

        # Build quantum circuit
        feature_map, ansatz = build_qiskit_circuit(
            num_qubits=q_cfg.num_qubits,
            feature_map_name=q_cfg.feature_map,
            ansatz_name=q_cfg.ansatz,
            entanglement=q_cfg.entanglement,
            reps=q_cfg.reps,
        )

        qc = QuantumCircuit(q_cfg.num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        # Create QNN
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
        )

        self.qnn = TorchConnector(qnn)

        # Output head
        num_classes = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        self.head = nn.Linear(q_cfg.num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        else:
            x = x.flatten(1)

        x = torch.tanh(self.project(x)) * np.pi
        x = self.qnn(x)
        return self.head(x)
