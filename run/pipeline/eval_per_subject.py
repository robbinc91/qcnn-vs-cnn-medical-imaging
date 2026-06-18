"""Per-subject test-set evaluation for all trained models.

For each model output dir (run/outputs/<model>_<dims>d and the archived 6M 2D
quantum runs), reconstruct the model from its saved Hydra config + best
checkpoint, run CPU inference on each of the 6 held-out test subjects, and
record per-subject Dice (mean + per class). Output: long-format CSV for stats.

CPU-only (CUDA hidden) so it never contends with other GPU jobs.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU, avoid busy GPUs

import glob
import csv
import numpy as np
import torch
from omegaconf import OmegaConf

from src.model_module import ModelFactory
from src.data_module.brainstem import (
    BrainstemSliceDataset, BrainstemVolumeDataset, get_subject_splits,
)
from src.utils.metrics import compute_dice_per_class

DATA_DIR = "datatrain/"
NUM_CLASSES = 4


def load_model(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    model = ModelFactory(cfg)
    ckpt = os.path.join(run_dir, "best_model.pth")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def subject_dice(model, dims, sid, min_vox):
    if dims == 2:
        ds = BrainstemSliceDataset(DATA_DIR, [sid], augmentation=None,
                                   min_brainstem_voxels=min_vox)
    else:
        ds = BrainstemVolumeDataset(DATA_DIR, [sid], augmentation=None)
    preds, masks = [], []
    for i in range(len(ds)):
        img, mask = ds[i]
        out = model(img.unsqueeze(0))            # (1,C,...)
        preds.append(out.argmax(1).cpu().numpy().ravel())
        masks.append(mask.numpy().ravel())
    if not preds:
        return None
    yp = np.concatenate(preds); yt = np.concatenate(masks)
    per = compute_dice_per_class(yt, yp, NUM_CLASSES)
    return {
        "mean_dice": float(np.mean(list(per.values()))),
        "dice_c1": per["dice_class_1"], "dice_c2": per["dice_class_2"],
        "dice_c3": per["dice_class_3"], "n_items": len(ds),
    }


def main():
    run_dirs = sorted(glob.glob("run/outputs/*/"))
    run_dirs += sorted(glob.glob("run/archive_6M_2d/*/"))
    rows = []
    for rd in run_dirs:
        if not os.path.isfile(os.path.join(rd, "best_model.pth")):
            continue
        try:
            model, cfg = load_model(rd)
        except Exception as exc:
            print(f"SKIP {rd}: {exc}")
            continue
        dims = int(cfg.dataset.spatial_dims)
        name = cfg.model.name
        typ = cfg.model.get("type", "unknown")
        params = sum(p.numel() for p in model.parameters())
        budget = "6M" if params < 1.0e7 else "17M"
        min_vox = int(cfg.dataset.get("min_brainstem_voxels", 20))
        test_ids = get_subject_splits(int(cfg.seed))["test"]
        tag = f"{name}_{dims}d_{budget}" if typ in ("quantum", "hybrid") else f"{name}_{dims}d"
        for sid in test_ids:
            r = subject_dice(model, dims, sid, min_vox)
            if r is None:
                continue
            rows.append(dict(run=tag, model=name, dims=dims, type=typ,
                             budget=budget, params=params, subject=sid, **r))
        print(f"done {tag}: {params/1e6:.2f}M, {len(test_ids)} subjects")

    os.makedirs("run/outputs/analysis", exist_ok=True)
    out = "run/outputs/analysis/per_subject_dice.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
