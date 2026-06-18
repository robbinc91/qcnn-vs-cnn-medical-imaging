"""Per-subject test-set evaluation (instrumented, incremental, CPU-only).

Writes one CSV row per (model, subject) as it goes and flushes, so progress is
visible and partial results survive. Fast classical models first, then quantum.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time
import glob
import csv
import json
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
OUT = "run/outputs/analysis/per_subject_dice.csv"
torch.set_num_threads(10)


def log(*a):
    print(*a, flush=True)


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
def subject_dice(model, dims, sid, min_vox, chunk=32):
    if dims == 2:
        ds = BrainstemSliceDataset(DATA_DIR, [sid], None, min_vox)
    else:
        ds = BrainstemVolumeDataset(DATA_DIR, [sid], None)
    n = len(ds)
    if n == 0:
        return None
    preds, masks = [], []
    # Batch slices/volumes into chunked forward passes (huge speedup vs B=1).
    for start in range(0, n, chunk):
        items = [ds[i] for i in range(start, min(start + chunk, n))]
        imgs = torch.stack([it[0] for it in items])     # (B,1,H,W[,D])
        out = model(imgs)                                # (B,C,H,W[,D])
        preds.append(out.argmax(1).cpu().numpy().ravel())
        masks.append(np.stack([it[1].numpy() for it in items]).ravel())
    yp = np.concatenate(preds); yt = np.concatenate(masks)
    per = compute_dice_per_class(yt, yp, NUM_CLASSES)
    return dict(mean_dice=float(np.mean(list(per.values()))),
                dice_c1=per["dice_class_1"], dice_c2=per["dice_class_2"],
                dice_c3=per["dice_class_3"], n_items=len(ds))


def order_key(rd):
    # classical & 3D-quantum fast first; 2D quantum slow last; archived last
    is_arch = "archive_6M" in rd
    try:
        cfg = OmegaConf.load(os.path.join(rd, ".hydra", "config.yaml"))
        typ = cfg.model.get("type", "x"); dims = int(cfg.dataset.spatial_dims)
    except Exception:
        typ, dims = "z", 9
    slow = 1 if (typ in ("quantum", "hybrid") and dims == 2) else 0
    return (is_arch, slow, rd)


def main():
    os.makedirs("run/outputs/analysis", exist_ok=True)
    dirs = sorted(glob.glob("run/outputs/*/")) + sorted(glob.glob("run/archive_6M_2d/*/"))
    dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "best_model.pth"))]
    dirs.sort(key=order_key)

    fields = ["run", "model", "dims", "type", "budget", "params",
              "subject", "mean_dice", "dice_c1", "dice_c2", "dice_c3", "n_items"]
    f = open(OUT, "w", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()

    for rd in dirs:
        t0 = time.time()
        try:
            model, cfg = load_model(rd)
        except Exception as exc:
            log(f"SKIP {rd}: {exc}"); continue
        dims = int(cfg.dataset.spatial_dims); name = cfg.model.name
        typ = cfg.model.get("type", "unknown")
        params = sum(p.numel() for p in model.parameters())
        budget = "6M" if params < 1.0e7 else "17M"
        min_vox = int(cfg.dataset.get("min_brainstem_voxels", 20))
        ids = get_subject_splits(int(cfg.seed))["test"]
        tag = f"{name}_{dims}d_{budget}" if typ in ("quantum", "hybrid") else f"{name}_{dims}d"
        log(f">> {tag} ({params/1e6:.2f}M) load={time.time()-t0:.1f}s")
        for sid in ids:
            ts = time.time()
            try:
                r = subject_dice(model, dims, sid, min_vox)
            except Exception as exc:
                log(f"   subj {sid} ERROR: {exc}"); continue
            if r is None:
                continue
            w.writerow(dict(run=tag, model=name, dims=dims, type=typ,
                            budget=budget, params=params, subject=sid, **r))
            f.flush()
            log(f"   subj {sid}: dice={r['mean_dice']:.4f} "
                f"n={r['n_items']} ({time.time()-ts:.1f}s)")
        log(f"== {tag} done in {time.time()-t0:.1f}s")
    f.close()
    log(f"\nALL DONE -> {OUT}")


if __name__ == "__main__":
    main()
