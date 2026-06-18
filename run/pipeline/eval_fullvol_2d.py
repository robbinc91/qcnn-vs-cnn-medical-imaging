"""Honest full-volume 2D evaluation (C2 fix).

Scores existing 2D checkpoints on ALL axial slices of each test subject
(min_brainstem_voxels=0), so false positives on empty/non-brainstem slices are
penalised — unlike the training-time eval which filtered to foreground-bearing
slices and gave optimistic Dice. Uses the as-trained checkpoints (no retraining).

Per-subject Dice = pixel-pooled over all 96 slices of the subject.
Output: run/outputs/analysis/per_subject_dice_fullvol_2d.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time, glob, csv
import numpy as np
import torch
from omegaconf import OmegaConf

from src.model_module import ModelFactory
from src.data_module.brainstem import BrainstemSliceDataset, get_subject_splits
from src.utils.metrics import compute_dice_per_class

DATA_DIR = "datatrain/"
NUM_CLASSES = 4
OUT = "run/outputs/analysis/per_subject_dice_fullvol_2d.csv"
torch.set_num_threads(10)


def log(*a): print(*a, flush=True)


def load_model(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    model = ModelFactory(cfg)
    state = torch.load(os.path.join(run_dir, "best_model.pth"), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg


@torch.no_grad()
def subject_dice(model, sid, chunk=32):
    # min_brainstem_voxels=0 -> ALL slices of the subject (full volume)
    ds = BrainstemSliceDataset(DATA_DIR, [sid], None, 0)
    n = len(ds)
    if n == 0:
        return None
    preds, masks = [], []
    for s in range(0, n, chunk):
        items = [ds[i] for i in range(s, min(s + chunk, n))]
        imgs = torch.stack([it[0] for it in items])
        out = model(imgs)
        preds.append(out.argmax(1).cpu().numpy().ravel())
        masks.append(np.stack([it[1].numpy() for it in items]).ravel())
    yp = np.concatenate(preds); yt = np.concatenate(masks)
    per = compute_dice_per_class(yt, yp, NUM_CLASSES)
    return dict(mean_dice=float(np.mean(list(per.values()))),
                dice_c1=per["dice_class_1"], dice_c2=per["dice_class_2"],
                dice_c3=per["dice_class_3"], n_slices=n)


def order_key(rd):
    is_arch = "archive_6M" in rd
    try:
        cfg = OmegaConf.load(os.path.join(rd, ".hydra", "config.yaml"))
        slow = 1 if cfg.model.get("type", "x") in ("quantum", "hybrid") else 0
    except Exception:
        slow = 9
    return (is_arch, slow, rd)


def main():
    os.makedirs("run/outputs/analysis", exist_ok=True)
    dirs = sorted(glob.glob("run/outputs/*_2d/")) + sorted(glob.glob("run/archive_6M_2d/*/"))
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "best_model.pth"))]
    # keep only 2D runs
    keep = []
    for d in dirs:
        cfg = OmegaConf.load(os.path.join(d, ".hydra", "config.yaml"))
        if int(cfg.dataset.spatial_dims) == 2:
            keep.append(d)
    keep.sort(key=order_key)

    fields = ["run", "model", "type", "budget", "params", "subject",
              "mean_dice", "dice_c1", "dice_c2", "dice_c3", "n_slices"]
    f = open(OUT, "w", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()

    for rd in keep:
        t0 = time.time()
        try:
            model, cfg = load_model(rd)
        except Exception as exc:
            log(f"SKIP {rd}: {exc}"); continue
        name = cfg.model.name; typ = cfg.model.get("type", "unknown")
        params = sum(p.numel() for p in model.parameters())
        budget = "6M" if params < 1.0e7 else "17M"
        ids = get_subject_splits(int(cfg.seed))["test"]
        tag = f"{name}_2d_{budget}" if typ in ("quantum", "hybrid") else f"{name}_2d"
        vals = []
        for sid in ids:
            r = subject_dice(model, sid)
            if r is None:
                continue
            vals.append(r["mean_dice"])
            w.writerow(dict(run=tag, model=name, type=typ, budget=budget,
                            params=params, subject=sid, **r)); f.flush()
        log(f"== {tag:22} fullvol Dice={np.mean(vals):.4f}+/-{np.std(vals,ddof=1):.4f} "
            f"({time.time()-t0:.0f}s)")
    f.close()
    log(f"\nALL DONE -> {OUT}")


if __name__ == "__main__":
    main()
