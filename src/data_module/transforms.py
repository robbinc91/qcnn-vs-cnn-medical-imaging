"""Image transforms for preprocessing pipeline."""

from typing import Any

from torchvision import transforms


def get_transforms(dataset_cfg: Any) -> transforms.Compose:
    """Build transform pipeline based on dataset config."""
    transform_list = [transforms.ToTensor()]

    if dataset_cfg.channels == 1:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    elif dataset_cfg.channels == 3:
        transform_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

    return transforms.Compose(transform_list)
