"""Generate the headline C2 figure: full-volume vs foreground-only 2D Dice."""
import csv, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = "run/outputs/analysis"


def per_run_mean(path, key="mean_dice"):
    rows = list(csv.DictReader(open(path)))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["run"]].append(float(r[key]))
    return {k: (np.mean(v), np.std(v, ddof=1)) for k, v in by.items()}

fg = per_run_mean(f"{A}/per_subject_dice.csv")          # foreground-only
fv = per_run_mean(f"{A}/per_subject_dice_fullvol_2d.csv")  # full-volume

# 9 parameter-matched 2D models (quantum/hybrid at 17M)
models = [
    ("qcnn_qiskit_2d_17M", "Qiskit-QCNN", "q"),
    ("hybrid_qcnn_2d_17M", "Hybrid-QCNN", "q"),
    ("qcnn_pennylane_2d_17M", "PennyLane-QCNN", "q"),
    ("cerebnet_2d", "CerebNet", "c"),
    ("classical_cnn_2d", "ClassicalCNN", "c"),
    ("swin_unetr_2d", "SwinUNETR", "c"),
    ("mipaim_unet_2d", "MipaimUNet", "c"),
    ("acapulco_2d", "ACAPULCO", "c"),
    ("marin_2d", "MARIN", "c"),
]
# sort by full-volume mean descending
models.sort(key=lambda m: -fv[m[0]][0])
labels = [m[1] for m in models]
fg_m = [fg[m[0]][0] for m in models]
fv_m = [fv[m[0]][0] for m in models]
fv_s = [fv[m[0]][1] for m in models]
fg_s = [fg[m[0]][1] for m in models]
colors = ["#d62728" if m[2] == "q" else "#1f77b4" for m in models]

x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.2))
b1 = ax.bar(x - w/2, fg_m, w, yerr=fg_s, capsize=2, color="#bbbbbb",
            label="Foreground-only (filtered)")
b2 = ax.bar(x + w/2, fv_m, w, yerr=fv_s, capsize=2, color=colors,
            label="Full-volume (honest)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("per-subject mean Dice (n=6)")
ax.set_ylim(0.80, 0.95)
ax.axhline(np.mean(fv_m), ls="--", lw=0.8, color="k", alpha=0.4)
ax.set_title("2D brainstem segmentation: filtered vs honest full-volume evaluation")
# legend: grey = filtered; red = quantum/hybrid full-vol; blue = classical full-vol
from matplotlib.patches import Patch
leg = [Patch(color="#bbbbbb", label="Foreground-only (filtered)"),
       Patch(color="#d62728", label="Full-volume - quantum/hybrid"),
       Patch(color="#1f77b4", label="Full-volume - classical")]
ax.legend(handles=leg, fontsize=8, loc="lower left")
fig.tight_layout()
os.makedirs("paper/figures", exist_ok=True)
fig.savefig("paper/figures/fig_fullvol_vs_fg_2d.png", dpi=160)
fig.savefig("paper/figures/fig_fullvol_vs_fg_2d.pdf")
print("wrote paper/figures/fig_fullvol_vs_fg_2d.{png,pdf}")
