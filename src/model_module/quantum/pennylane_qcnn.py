"""Quantum CNN implementation using PennyLane."""

import logging
from typing import Any, List

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..registry import register_model

logger = logging.getLogger(__name__)


class QuantumConvLayer(nn.Module):
    """Single quantum convolutional layer using PennyLane.

    Implements a parameterized quantum circuit acting as a convolutional filter.
    Each layer applies entangling gates between neighboring qubits followed by
    single-qubit rotations, mimicking local feature extraction in classical CNNs.
    """

    def __init__(
        self,
        num_qubits: int,
        entanglement: str = "linear",
        ansatz: str = "strongly_entangling",
        diff_method: str = "best",
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits

        dev = qml.device(device_name, wires=num_qubits)

        @qml.qnode(dev, diff_method=diff_method, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[float]:
            # Encode classical data into quantum state
            qml.AngleEmbedding(inputs, wires=range(num_qubits))

            # Variational layers
            if ansatz == "strongly_entangling":
                qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            elif ansatz == "basic_entangler":
                qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
            else:
                # Custom: alternating RY + CNOT
                for layer_weights in weights:
                    for i in range(num_qubits):
                        qml.RY(layer_weights[i, 0], wires=i)
                        qml.RZ(layer_weights[i, 1], wires=i)
                    for i in range(num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if entanglement == "circular" and num_qubits > 2:
                        qml.CNOT(wires=[num_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit
        self.ansatz = ansatz

    def init_weights(self, num_layers: int) -> nn.Parameter:
        """Initialize quantum circuit weights."""
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

        return nn.Parameter(
            torch.randn(shape, dtype=torch.float32) * 0.1
        )


class ClassicalPreprocessor(nn.Module):
    """Classical feature extraction to reduce input dimensionality for quantum circuit."""

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


@register_model("qcnn_pennylane")
class PennyLaneQCNN(nn.Module):
    """Quantum CNN using PennyLane with optional classical pre/post processing.

    Architecture:
        [Classical Conv Preprocessor] -> Quantum Circuit -> [Classical FC Head]
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        q_cfg = cfg.model.quantum

        # Classical preprocessing
        if cfg.model.classical_pre.enabled:
            self.preprocessor = ClassicalPreprocessor(cfg)
            pre_dim = self.preprocessor.output_dim
        else:
            self.preprocessor = None
            pre_dim = cfg.dataset.channels * cfg.dataset.image_size ** 2

        # Linear projection to match qubit count
        self.project = nn.Linear(pre_dim, q_cfg.num_qubits)

        # Quantum layer
        self.q_layer = QuantumConvLayer(
            num_qubits=q_cfg.num_qubits,
            entanglement=q_cfg.entanglement,
            ansatz=q_cfg.ansatz,
            diff_method=q_cfg.diff_method,
            device_name=q_cfg.device,
        )
        self.q_weights = self.q_layer.init_weights(q_cfg.num_layers)

        # Output head
        num_classes = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )
        fc_dims = list(cfg.model.classifier.fc_dims)
        if fc_dims:
            layers: List[nn.Module] = []
            prev = q_cfg.num_qubits
            for dim in fc_dims:
                layers.extend([nn.Linear(prev, dim), nn.ReLU(inplace=True)])
                prev = dim
            layers.append(nn.Linear(prev, num_classes))
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(q_cfg.num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Classical preprocessing
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        else:
            x = x.flatten(1)

        # Project to qubit dimension
        x = torch.tanh(self.project(x)) * np.pi

        # Quantum circuit (process each sample)
        q_out = torch.stack([
            torch.stack(self.q_layer.circuit(x[i], self.q_weights))
            for i in range(batch_size)
        ])

        return self.head(q_out)
