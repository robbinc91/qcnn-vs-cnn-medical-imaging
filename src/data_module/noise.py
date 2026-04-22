"""Noise augmentation for robustness experiments."""

import logging
from typing import Any

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)


class NoiseAugmentation:
    """Apply configurable noise to images for robustness testing."""

    def __init__(self, noise_type: str, **kwargs: Any) -> None:
        self.noise_type = noise_type
        self.kwargs = kwargs

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "gaussian":
            return self._add_gaussian(img)
        elif self.noise_type == "salt_pepper":
            return self._add_salt_pepper(img)
        elif self.noise_type == "rotation":
            return self._add_rotation(img)
        return img

    def _add_gaussian(self, img: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with specified std."""
        std = self.kwargs.get("std", 0.1)
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, -1.0, 1.0)

    def _add_salt_pepper(self, img: torch.Tensor) -> torch.Tensor:
        """Add salt-and-pepper noise with specified probability."""
        prob = self.kwargs.get("prob", 0.05)
        noise = torch.rand_like(img)
        img = img.clone()
        img[noise < prob / 2] = -1.0  # pepper
        img[noise > 1 - prob / 2] = 1.0  # salt
        return img

    def _add_rotation(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random rotation within specified degrees."""
        degrees = self.kwargs.get("degrees", 10)
        return transforms.functional.rotate(img, angle=float(torch.randint(-degrees, degrees + 1, (1,))))
