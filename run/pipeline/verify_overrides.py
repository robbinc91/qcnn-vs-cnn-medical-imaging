"""Verify the exact CLI overrides bring all three 2D models to ~17M params.

Uses Hydra to compose each model's real config (matching the training entry
point), applies the 4-stage channel override, and counts parameters in 2D.
"""
from hydra import initialize, compose
from src.model_module import ModelFactory

CASES = [
    ("qcnn_pennylane_brainstem", "model.enc_channels=[80,160,320,480]"),
    ("qcnn_qiskit_brainstem", "model.enc_channels=[80,160,320,480]"),
    ("hybrid_qcnn_brainstem", "model.classical_encoder.conv_channels=[80,160,320,480]"),
]

with initialize(version_base=None, config_path="../../conf"):
    print(f"{'model':22}{'baseline(3-stage)':>20}{'override(4-stage)':>20}")
    for model_cfg, override in CASES:
        base = compose(config_name="config", overrides=[
            "+experiment=exp_brainstem", f"model={model_cfg}", "dataset.spatial_dims=2"])
        n_base = sum(p.numel() for p in ModelFactory(base).parameters())
        over = compose(config_name="config", overrides=[
            "+experiment=exp_brainstem", f"model={model_cfg}", "dataset.spatial_dims=2", override])
        n_over = sum(p.numel() for p in ModelFactory(over).parameters())
        print(f"{model_cfg.replace('_brainstem',''):22}{n_base/1e6:>18.2f}M{n_over/1e6:>18.2f}M")
