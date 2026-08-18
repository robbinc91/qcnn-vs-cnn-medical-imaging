"""Per-subject test-set evaluation for all HAMMERS sweep3 models.

For each model output dir (runhammers/<model>_<dims>d), reconstruct the model
from its saved Hydra config + best checkpoint, run inference on each of the 5
held-out test subjects, and record per-subject Dice (mean + per class).
Output: long-format CSV for stats. GPU used when available (GPU1 is free).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import glob
import csv
import numpy as np
import torch
from omegaconf import OmegaConf

from src.model_module import ModelFactory
from src.data_module.hammers import (
    HammersSliceDataset, HammersVolumeDataset, get_subject_splits,
)
from src.utils.metrics import compute_dice_per_class

DATA_DIR = "datatrain_hammers/"
NUM_CLASSES = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    model = ModelFactory(cfg)
    ckpt = os.path.join(run_dir, "best_model.pth")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def subject_dice(model, dims, sid, min_fg):
    if dims == 2:
        ds = HammersSliceDataset(DATA_DIR, [sid], augmentation=None,
                                 min_foreground_voxels=min_fg)
    else:
        ds = HammersVolumeDataset(DATA_DIR, [sid], augmentation=None)
    preds, masks = [], []
    for i in range(len(ds)):
        img, mask = ds[i]
        out = model(img.unsqueeze(0).to(device))
        preds.append(out.argmax(1).cpu().numpy().ravel())
        masks.append(mask.numpy().ravel())
    if not preds:
        return None
    yp = np.concatenate(preds); yt = np.concatenate(masks)
    per = compute_dice_per_class(yt, yp, NUM_CLASSES)
    dices = [per["dice_class_%d" % c] for c in range(1, NUM_CLASSES)]
    res = {"mean_dice": float(np.mean(dices)), "n_items": len(ds)}
    for c in range(1, NUM_CLASSES):
        res["dice_c%d" % c] = per["dice_class_%d" % c]
    return res


def main():
    run_dirs = sorted(glob.glob("runhammers/*/"))
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
        min_fg = int(cfg.dataset.get("min_foreground_voxels", 20))
        test_ids = get_subject_splits(int(cfg.seed))["test"]
        tag = f"{name}_{dims}d"
        for sid in test_ids:
            r = subject_dice(model, dims, sid, min_fg)
            if r is None:
                continue
            rows.append(dict(run=tag, model=name, dims=dims, type=typ,
                             budget=budget, params=params, subject=sid, **r))
        print(f"done {tag}: {params/1e6:.2f}M, {len(test_ids)} subjects", flush=True)

    os.makedirs("runhammers/analysis", exist_ok=True)
    out = "runhammers/analysis/per_subject_dice.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
