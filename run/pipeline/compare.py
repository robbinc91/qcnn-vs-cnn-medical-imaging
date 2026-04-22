"""Compare results across multiple experiment runs."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.visualization import plot_comparison, plot_param_vs_accuracy

logger = logging.getLogger(__name__)


def collect_results(output_dir: str = "run/outputs") -> List[Dict[str, Any]]:
    """Scan output directories for results.json files."""
    results = []
    for result_file in Path(output_dir).rglob("results.json"):
        with open(result_file) as f:
            data = json.load(f)

        # Also load the config if available
        config_file = result_file.parent / "config_resolved.yaml"
        if config_file.exists():
            data["config_path"] = str(config_file)

        data["run_dir"] = str(result_file.parent)
        results.append(data)

    logger.info(f"Collected {len(results)} experiment results")
    return results


def build_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison DataFrame from collected results."""
    rows = []
    for r in results:
        rows.append({
            "run_dir": r.get("run_dir", ""),
            "model": r.get("model_name", "unknown"),
            "num_parameters": r.get("num_parameters", 0),
            "test_accuracy": r.get("test_accuracy", 0),
            "test_f1": r.get("test_f1_macro", 0),
            "training_time_s": r.get("training_time_seconds", 0),
            "best_epoch": r.get("best_epoch", 0),
        })
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)


def main() -> None:
    """Generate comparison report."""
    logging.basicConfig(level=logging.INFO)

    results = collect_results()
    if not results:
        logger.warning("No results found in run/outputs/")
        return

    df = build_comparison_table(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    # Save CSV
    df.to_csv("run/outputs/comparison.csv", index=False)

    # Generate plots
    model_results = {r.get("model_name", f"run_{i}"): r for i, r in enumerate(results)}
    plot_comparison(
        model_results,
        metric="test_accuracy",
        save_path="run/outputs/comparison_accuracy.png",
        title="Test Accuracy Comparison",
    )

    # Parameter efficiency plot
    model_data = [
        {
            "name": r.get("model_name", f"run_{i}"),
            "params": r.get("num_parameters", 0),
            "accuracy": r.get("test_accuracy", 0),
            "type": r.get("model_type", "classical"),
        }
        for i, r in enumerate(results)
    ]
    plot_param_vs_accuracy(
        model_data,
        save_path="run/outputs/param_efficiency.png",
        title="Parameter Count vs Test Accuracy",
    )

    logger.info("Comparison report saved to run/outputs/")


if __name__ == "__main__":
    main()
