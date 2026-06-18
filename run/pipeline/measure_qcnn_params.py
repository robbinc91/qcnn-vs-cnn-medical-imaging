"""Measure 2D PennyLaneQCNN parameter counts for candidate 4-stage enc_channels.

The quantum/hybrid models share the same conv U-Net backbone, so the count is
representative for all three. Goal: find the 4-stage config closest to ~17M.
"""
import sys
from omegaconf import OmegaConf
from src.model_module import ModelFactory


def count(enc_channels):
    cfg = OmegaConf.create({
        "dataset": {"spatial_dims": 2, "channels": 1, "num_classes": 4,
                    "binary_classes": None},
        "model": {
            "name": "qcnn_pennylane", "type": "quantum", "framework": "pennylane",
            "enc_channels": list(enc_channels), "decoder_dropout": 0.2,
            "quantum": {"num_qubits": 8, "num_layers": 4, "entanglement": "circular",
                        "ansatz": "strongly_entangling", "diff_method": "best",
                        "device": "default.qubit"},
        },
    })
    model = ModelFactory(cfg)
    return sum(p.numel() for p in model.parameters())


CANDIDATES = [
    [80, 160, 320],            # current (3-stage baseline)
    [64, 128, 256, 352],       # classical_cnn 2D widths
    [80, 160, 320, 480],
    [80, 160, 320, 512],
    [80, 160, 320, 640],
    [96, 192, 384, 512],
]

print(f"{'enc_channels':28}{'params':>12}{'M':>9}")
for c in CANDIDATES:
    n = count(c)
    print(f"{str(c):28}{n:>12,}{n/1e6:>8.2f}M")
