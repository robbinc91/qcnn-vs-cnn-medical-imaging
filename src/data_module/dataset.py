"""Dataset loading with factory pattern and optional subsampling."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .transforms import get_transforms

logger = logging.getLogger(__name__)

DATASET_FACTORY: Dict[str, Type[datasets.VisionDataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class in the factory."""
    def decorator(cls: Type[datasets.VisionDataset]) -> Type[datasets.VisionDataset]:
        DATASET_FACTORY[name] = cls
        return cls
    return decorator


# Register torchvision datasets
DATASET_FACTORY["mnist"] = datasets.MNIST
DATASET_FACTORY["fashion_mnist"] = datasets.FashionMNIST
DATASET_FACTORY["cifar10"] = datasets.CIFAR10


def DatasetFactory(cfg: Any) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from config.

    Args:
        cfg: Hydra config with dataset and training sections.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    ds_name = cfg.dataset.name
    if ds_name not in DATASET_FACTORY:
        raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(DATASET_FACTORY.keys())}")

    ds_cls = DATASET_FACTORY[ds_name]
    transform = get_transforms(cfg.dataset)

    train_ds = ds_cls(
        root=cfg.dataset.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = ds_cls(
        root=cfg.dataset.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Filter to binary classes if specified
    binary_classes = cfg.dataset.get("binary_classes")
    if binary_classes is not None:
        train_ds = _filter_classes(train_ds, binary_classes)
        test_ds = _filter_classes(test_ds, binary_classes)
        logger.info(f"Filtered to classes {binary_classes}")

    # Subsample if specified
    subsample = cfg.dataset.get("subsample")
    if subsample is not None:
        train_ds = _subsample(train_ds, subsample)
        test_ds = _subsample(test_ds, min(subsample // 4, len(test_ds)))
        logger.info(f"Subsampled to {len(train_ds)} train, {len(test_ds)} test")

    # Split train into train/val
    val_size = int(len(train_ds) * cfg.dataset.val_split)
    train_size = len(train_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [train_size, val_size]
    )

    logger.info(
        f"Dataset {ds_name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def _filter_classes(
    dataset: datasets.VisionDataset,
    classes: List[int],
) -> Subset:
    """Filter dataset to only include specified classes, relabeling to 0..N-1."""
    targets = np.array(dataset.targets)
    mask = np.isin(targets, classes)
    indices = np.where(mask)[0].tolist()

    # Relabel: map original class IDs to 0, 1, ...
    class_map = {c: i for i, c in enumerate(sorted(classes))}
    for idx in indices:
        dataset.targets[idx] = class_map[dataset.targets[idx]]

    return Subset(dataset, indices)


def _subsample(dataset: Dataset, n: int) -> Subset:
    """Randomly subsample n examples from dataset."""
    n = min(n, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False).tolist()
    return Subset(dataset, indices)
