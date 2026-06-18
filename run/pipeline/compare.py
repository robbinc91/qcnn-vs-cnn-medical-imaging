"""Collect and compare segmentation results across experiment runs."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

METRICS = [
    "test_mean_dice",
    "test_dice_class_1",   # medulla
    "test_dice_class_2",   # pons
    "test_dice_class_3",   # mesencephalon
    "test_pixel_accuracy",
]


def collect_results(output_dir: str = "run/outputs") -> List[Dict[str, Any]]:
    results = []
    for result_file in sorted(Path(output_dir).rglob("results.json")):
        with open(result_file) as f:
            data = json.load(f)
        data["run_dir"] = str(result_file.parent)
        results.append(data)
    logger.info(f"Collected {len(results)} experiment results")
    return results


def build_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model":         r.get("model_name", "unknown"),
            "spatial_dims":  r.get("spatial_dims", "?"),
            "params":        f"{r.get('num_parameters', 0) / 1e6:.2f} M",
            "mean_dice":     f"{r.get('test_mean_dice', 0):.4f}",
            "medulla":       f"{r.get('test_dice_class_1', 0):.4f}",
            "pons":          f"{r.get('test_dice_class_2', 0):.4f}",
            "mesencephalon": f"{r.get('test_dice_class_3', 0):.4f}",
            "pixel_acc":     f"{r.get('test_pixel_accuracy', 0):.4f}",
            "best_epoch":    r.get("best_epoch", 0),
            "time_min":      f"{r.get('training_time_seconds', 0) / 60:.1f}",
        })
    df = pd.DataFrame(rows)
    if "mean_dice" in df.columns:
        df = df.sort_values("mean_dice", ascending=False)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    results = collect_results()
    if not results:
        print("No results found in run/outputs/. Run training first.")
        return

    df = build_comparison_table(results)

    print("\n" + "=" * 90)
    print("BRAINSTEM SEGMENTATION — COMPARISON")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    out = Path("run/outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison.csv", index=False)
    print(f"\nSaved: {out / 'comparison.csv'}")


if __name__ == "__main__":
    main()
