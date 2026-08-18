"""Generate segmentation overlay screenshots for all HAMMERS sweep3 models (GPU1).

Loads each model checkpoint, runs inference on test subject slices, and
saves 3-panel montages (MRI / Ground Truth / Prediction).

Naming: <model>_<dims>d_subj<S>_slice<Z>.png (2D) / _axial_z<Z>.png (3D)
"""

import os, sys, glob, time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # use GPU 1 (free)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from omegaconf import OmegaConf

sys.path.insert(0, "/home/ai/claude_research/qcnn-vs-cnn-medical-imaging")
from src.model_module import ModelFactory
from src.data_module.hammers import (
    HammersSliceDataset, HammersVolumeDataset, get_subject_splits,
)

DATA_DIR = "datatrain_hammers/"
OUTPUT_DIR = "runhammers/analysis/segmentation_screenshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                 "#ff7f00", "#a65628", "#f781bf", "#999999"]
REGION_NAMES_ES = [
    "Parietal Superior Izq.", "Parietal Superior Der.",
    "Supramarginal Izq.", "Supramarginal Der.",
    "Angular Izq.", "Angular Der.",
    "Postcentral Izq.", "Postcentral Der.",
]
CMAP = ListedColormap(["#000000"] + REGION_COLORS)
LEGEND = [Patch(facecolor=REGION_COLORS[i], label=REGION_NAMES_ES[i])
          for i in range(8)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(output_dir):
    cfg = OmegaConf.load(os.path.join(output_dir, ".hydra", "config.yaml"))
    model = ModelFactory(cfg)
    ckpt = os.path.join(output_dir, "best_model.pth")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def infer_2d(model, cfg, output_tag):
    seed = int(cfg.get("seed", 42))
    min_fg = int(cfg.dataset.get("min_foreground_voxels", 20))
    test_ids = get_subject_splits(seed)["test"]

    for sid in test_ids:
        subj_ds = HammersSliceDataset(DATA_DIR, [sid], augmentation=None,
                                      min_foreground_voxels=min_fg)
        n_slices = len(subj_ds.masks)
        if n_slices == 0:
            continue
        picks = [0, n_slices // 2, n_slices - 1] if n_slices >= 3 else list(range(n_slices))
        for pick_idx in picks:
            img, gt_mask = subj_ds[pick_idx]
            img = img.unsqueeze(0).to(device)
            out = model(img)
            pred_mask = out.argmax(1).squeeze(0).cpu().numpy()
            img_2d = img.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
            save_screenshot(
                img_2d,
                gt_mask.cpu().numpy(),
                pred_mask,
                f"{output_tag}_subj{sid}_slice{pick_idx}",
                f"Sujeto {sid} | Corte {pick_idx}/{n_slices-1}"
            )


@torch.no_grad()
def infer_3d(model, cfg, output_tag):
    seed = int(cfg.get("seed", 42))
    test_ids = get_subject_splits(seed)["test"]

    for sid in test_ids:
        subj_ds = HammersVolumeDataset(DATA_DIR, [sid], augmentation=None)
        if len(subj_ds.volumes) == 0:
            continue
        vol_img, vol_gt = subj_ds[0]
        vol_img_np = vol_img.squeeze(0).cpu().numpy()

        out = model(vol_img.unsqueeze(0).to(device))
        vol_pred = out.argmax(1).squeeze(0).cpu().numpy()

        D = vol_img_np.shape[2]
        mid = D // 2
        picks = [max(0, mid - 24), mid, min(D - 1, mid + 24)]

        for z in picks:
            gt_np = vol_gt[:, :, z].cpu().numpy() if torch.is_tensor(vol_gt) else vol_gt[:, :, z]
            save_screenshot(
                vol_img_np[:, :, z],
                gt_np,
                vol_pred[:, :, z],
                f"{output_tag}_subj{sid}_axial_z{z:02d}",
                f"Sujeto {sid} | Axial z={z}/{D-1}"
            )


def save_screenshot(mri_slice, gt_mask, pred_mask, name, title_str):
    lo, hi = np.percentile(mri_slice, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].imshow(mri_slice, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("MRI", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(mri_slice, cmap="gray", vmin=lo, vmax=hi)
    axes[1].imshow(gt_mask, cmap=CMAP, alpha=0.35, vmin=0, vmax=8)
    axes[1].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(mri_slice, cmap="gray", vmin=lo, vmax=hi)
    axes[2].imshow(pred_mask, cmap=CMAP, alpha=0.35, vmin=0, vmax=8)
    axes[2].set_title("Prediccion", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    fig.legend(handles=LEGEND, loc="lower center",
               ncol=4, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title_str, fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main():
    base_dir = "/home/ai/claude_research/qcnn-vs-cnn-medical-imaging/runhammers"
    model_dirs = sorted(glob.glob(os.path.join(base_dir, "*/")))

    t_start = time.time()
    for md in model_dirs:
        name = os.path.basename(md.rstrip("/"))
        if name.startswith(".") or name == "analysis":
            continue
        if not os.path.isfile(os.path.join(md, "best_model.pth")):
            continue
        if not os.path.isdir(os.path.join(md, ".hydra")):
            continue

        print(f"[{time.time()-t_start:.0f}s] Processing: {name}", flush=True)

        try:
            model, cfg = load_model(md)
        except Exception as e:
            print(f"  SKIP {name}: {e}", flush=True)
            continue

        dims = int(cfg.dataset.spatial_dims)
        t0 = time.time()
        try:
            if dims == 2:
                infer_2d(model, cfg, name)
            else:
                infer_3d(model, cfg, name)
        except Exception as e:
            print(f"  ERROR {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        n_files = len(glob.glob(os.path.join(OUTPUT_DIR, f"{name}_*.png")))
        print(f"  Done in {elapsed:.1f}s ({n_files} screenshots)", flush=True)

    total = time.time() - t_start
    n_total = len(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
    print(f"\nAll done in {total:.0f}s. {n_total} screenshots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
