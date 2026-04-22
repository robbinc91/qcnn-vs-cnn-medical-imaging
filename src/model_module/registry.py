"""Model registry with factory pattern."""

import logging
from typing import Any, Dict, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model: {name}")
        return cls
    return decorator


def ModelFactory(cfg: Any) -> nn.Module:
    """Create a model from config.

    Args:
        cfg: Hydra config with model section.

    Returns:
        Instantiated model.
    """
    # Import all model modules to trigger registration
    from . import classical, hybrid, quantum  # noqa: F401

    name = cfg.model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[name](cfg)
    logger.info(f"Created model: {name}")
    return model
