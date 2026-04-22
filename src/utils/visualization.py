"""Visualization utilities for experiment results."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and optionally save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> None:
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    if "train_accuracy" in history:
        axes[1].plot(history["train_accuracy"], label="Train Acc")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")
    plt.close(fig)


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
    title: str = "Model Comparison",
) -> None:
    """Bar chart comparing a metric across multiple models."""
    models = list(results.keys())
    values = [results[m].get(metric, 0.0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, values, color=sns.color_palette("Set2", len(models)))

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {save_path}")
    plt.close(fig)


def plot_param_vs_accuracy(
    model_data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Parameter Efficiency",
) -> None:
    """Scatter plot of parameter count vs accuracy for different models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for entry in model_data:
        marker = "s" if entry.get("type") == "classical" else "o"
        color = "#2196F3" if entry.get("type") == "classical" else "#E91E63"
        ax.scatter(
            entry["params"],
            entry["accuracy"],
            s=100,
            marker=marker,
            color=color,
            label=entry["name"],
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Parameter efficiency plot saved to {save_path}")
    plt.close(fig)
