"""Hybrid Quantum-Classical CNN: Classical encoder -> Quantum layer -> Classical decoder."""

import logging
from typing import Any, List

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..registry import register_model

logger = logging.getLogger(__name__)


@register_model("hybrid_qcnn")
class HybridQCNN(nn.Module):
    """Hybrid quantum-classical CNN with deeper classical bookends.

    This follows the QCQ-CNN paradigm:
        Classical Encoder (Conv layers) -> Quantum Processor -> Classical Decoder (FC layers)

    The classical encoder extracts spatial features and reduces dimensionality.
    The quantum layer processes compressed features using entanglement.
    The classical decoder maps quantum measurements to class predictions.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        enc_cfg = cfg.model.classical_encoder
        q_cfg = cfg.model.quantum
        dec_cfg = cfg.model.classical_decoder

        # --- Classical Encoder ---
        encoder_layers: List[nn.Module] = []
        in_ch = cfg.dataset.channels
        for out_ch in enc_cfg.conv_channels:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, enc_cfg.kernel_size, padding=enc_cfg.kernel_size // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(enc_cfg.pool_size),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # Compute encoder output size
        feat_size = cfg.dataset.image_size
        for _ in enc_cfg.conv_channels:
            feat_size = feat_size // enc_cfg.pool_size
        encoder_out_dim = enc_cfg.conv_channels[-1] * feat_size * feat_size

        # Projection to match qubit count
        self.flatten = nn.Flatten()
        self.project = nn.Linear(encoder_out_dim, q_cfg.num_qubits)

        # --- Quantum Layer ---
        dev = qml.device(q_cfg.device, wires=q_cfg.num_qubits)

        @qml.qnode(dev, diff_method=q_cfg.diff_method, interface="torch")
        def quantum_circuit(
            inputs: torch.Tensor, weights: torch.Tensor
        ) -> List[float]:
            # Angle encoding
            qml.AngleEmbedding(inputs, wires=range(q_cfg.num_qubits))

            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(q_cfg.num_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(q_cfg.num_qubits)]

        self.quantum_circuit = quantum_circuit

        weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=q_cfg.num_layers, n_wires=q_cfg.num_qubits
        )
        self.q_weights = nn.Parameter(
            torch.randn(weight_shape, dtype=torch.float32) * 0.1
        )

        # --- Classical Decoder ---
        num_classes = (
            len(cfg.dataset.binary_classes)
            if cfg.dataset.binary_classes
            else cfg.dataset.num_classes
        )

        decoder_layers: List[nn.Module] = []
        prev_dim = q_cfg.num_qubits
        for dim in dec_cfg.fc_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dec_cfg.dropout),
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, num_classes))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Classical encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = torch.tanh(self.project(x)) * np.pi

        # Quantum processing
        q_out = torch.stack([
            torch.stack(self.quantum_circuit(x[i], self.q_weights))
            for i in range(batch_size)
        ])

        # Classical decoding
        return self.decoder(q_out)
