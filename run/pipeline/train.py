"""Main training entry point with Hydra configuration."""

import logging
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_module import DatasetFactory
from src.model_module import ModelFactory
from src.trainer_module import Trainer
from src.utils.reproducibility import get_device, log_environment, set_seed
from src.utils.visualization import plot_training_curves

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train a model and return test accuracy for Hydra sweeps."""
    # Reproducibility
    set_seed(cfg.seed)
    env_info = log_environment()
    device = get_device(cfg.device)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Device: {device}")

    # Save environment info
    output_dir = Path(hydra.utils.get_original_cwd()) if not Path(".hydra").exists() else Path(".")
    with open("environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Data
    train_loader, val_loader, test_loader = DatasetFactory(cfg)

    # Model
    model = ModelFactory(cfg)

    # Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, cfg, device)
    results = trainer.train()

    # Save outputs
    trainer.save_results(results, ".")
    plot_training_curves(results["history"], save_path="training_curves.png")

    # Save config snapshot
    with open("config_resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
    return results["test_accuracy"]


if __name__ == "__main__":
    main()
