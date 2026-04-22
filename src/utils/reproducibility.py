"""Reproducibility utilities for experiment consistency."""

import hashlib
import logging
import os
import platform
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def log_environment() -> Dict[str, Any]:
    """Record environment information for reproducibility."""
    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_model": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "gpu_count": torch.cuda.device_count(),
        "platform": platform.platform(),
    }

    try:
        import pennylane as qml
        env_info["pennylane_version"] = qml.__version__
    except ImportError:
        env_info["pennylane_version"] = "not installed"

    try:
        import qiskit
        env_info["qiskit_version"] = qiskit.__version__
    except ImportError:
        env_info["qiskit_version"] = "not installed"

    logger.info(f"Environment: {env_info}")
    return env_info


def get_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for dataset versioning."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:12]


def get_device(device_cfg: str = "auto") -> torch.device:
    """Resolve device string to torch.device."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total:,}")
    return total
