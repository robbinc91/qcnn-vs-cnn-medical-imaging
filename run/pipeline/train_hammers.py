"""Training entry point for HAMMERS parietal-lobe segmentation (2-D and 3-D).

Usage:
    python run/pipeline/train_hammers.py experiment=exp_hammers \
        model=classical_cnn_hammers dataset.spatial_dims=2
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data_module.hammers import (
    HammersDataFactory,
    HammersVolumeDataFactory,
    CLASS_NAMES,
)
from src.model_module import ModelFactory
from src.trainer_module import Trainer
from src.utils.metrics import compute_dice_per_class, pixel_accuracy
from src.utils.reproducibility import get_device, log_environment, set_seed
from src.utils.visualization import plot_training_curves

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    set_seed(cfg.seed)
    env_info = log_environment()
    device = get_device(cfg.device)

    if cfg.dataset.name != "hammers":
        raise ValueError(
            f"train_hammers.py expects dataset=hammers, got: {cfg.dataset.name}."
        )

    spatial_dims = cfg.dataset.spatial_dims
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Device: {device} | spatial_dims: {spatial_dims}")
    logger.info(f"Output dir: {output_dir}")

    with open(output_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Data
    if spatial_dims == 2:
        train_loader, val_loader, test_loader, class_weights = HammersDataFactory(cfg)
    else:
        train_loader, val_loader, test_loader, class_weights = HammersVolumeDataFactory(cfg)

    # Model
    model = ModelFactory(cfg)
    logger.info(f"Model: {cfg.model.name}")

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, cfg, device,
        class_weights=class_weights,
    )
    results = trainer.train()

    results["model_name"] = cfg.model.name
    results["model_type"] = cfg.model.get("type", "unknown")
    results["dataset"] = cfg.dataset.name
    results["spatial_dims"] = spatial_dims

    trainer.save_results(results, str(output_dir))

    plot_training_curves(
        results["history"],
        save_path=str(output_dir / "training_curves.png"),
        title=f"{cfg.model.name} ({spatial_dims}D) - HAMMERS",
    )

    test_preds, test_masks = _collect_predictions(model, test_loader, device)
    dice_per_class = compute_dice_per_class(
        test_masks, test_preds, cfg.dataset.num_classes
    )
    pix_acc = pixel_accuracy(test_masks, test_preds)

    logger.info(
        f"Results | {cfg.model.name} {spatial_dims}D | "
        f"test_mean_dice={results.get('test_mean_dice', 0):.4f} | "
        f"test_pixel_acc={pix_acc:.4f} | "
        + " | ".join(
            f"{CLASS_NAMES[int(k.split('_')[-1])] if k.split('_')[-1].isdigit() and int(k.split('_')[-1]) < len(CLASS_NAMES) else k}={v:.4f}"
            for k, v in sorted(dice_per_class.items())
        )
        + f" | params={results['num_parameters']:,} | "
        f"time={results['training_time_seconds']:.1f}s"
    )

    with open(output_dir / "config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    return float(results.get("test_mean_dice", 0.0))


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    model.eval()
    preds, targets = [], []
    for data, mask in loader:
        data = data.to(device)
        with torch.amp.autocast('cuda', enabled=True):
            out = model(data)
        preds.append(out.argmax(dim=1).cpu().numpy().ravel())
        targets.append(mask.numpy().ravel())
    return np.concatenate(preds), np.concatenate(targets)


if __name__ == "__main__":
    main()
