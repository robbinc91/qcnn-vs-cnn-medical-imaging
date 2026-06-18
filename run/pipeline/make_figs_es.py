"""Spanish-labelled versions of the two paper figures."""
import csv, collections, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

A = "run/outputs/analysis"
os.makedirs("paper/figures", exist_ok=True)


def per_run_mean(path):
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[r["run"]].append(float(r["mean_dice"]))
    return {k: (np.mean(v), np.std(v, ddof=1)) for k, v in by.items()}

# ---- Figure 1: full-volume vs foreground-only (Spanish) ----
fg = per_run_mean(f"{A}/per_subject_dice.csv")
fv = per_run_mean(f"{A}/per_subject_dice_fullvol_2d.csv")
models = [
    ("qcnn_qiskit_2d_17M", "Qiskit-QCNN", "q"), ("hybrid_qcnn_2d_17M", "Hybrid-QCNN", "q"),
    ("qcnn_pennylane_2d_17M", "PennyLane-QCNN", "q"), ("cerebnet_2d", "CerebNet", "c"),
    ("classical_cnn_2d", "ClassicalCNN", "c"), ("swin_unetr_2d", "SwinUNETR", "c"),
    ("mipaim_unet_2d", "MipaimUNet", "c"), ("acapulco_2d", "ACAPULCO", "c"),
    ("marin_2d", "MARIN", "c"),
]
models.sort(key=lambda m: -fv[m[0]][0])
labels = [m[1] for m in models]
fg_m = [fg[m[0]][0] for m in models]; fg_s = [fg[m[0]][1] for m in models]
fv_m = [fv[m[0]][0] for m in models]; fv_s = [fv[m[0]][1] for m in models]
colors = ["#d62728" if m[2] == "q" else "#1f77b4" for m in models]
x = np.arange(len(models)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(x - w/2, fg_m, w, yerr=fg_s, capsize=2, color="#bbbbbb")
ax.bar(x + w/2, fv_m, w, yerr=fv_s, capsize=2, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Dice medio por sujeto (n=6)")
ax.set_ylim(0.80, 0.95)
ax.axhline(np.mean(fv_m), ls="--", lw=0.8, color="k", alpha=0.4)
ax.set_title("Segmentacion 2D del tronco encefalico: evaluacion filtrada vs volumen completo")
leg = [Patch(color="#bbbbbb", label="Solo cortes con estructura (filtrado)"),
       Patch(color="#d62728", label="Volumen completo - cuantico/hibrido"),
       Patch(color="#1f77b4", label="Volumen completo - clasico")]
ax.legend(handles=leg, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("paper/figures/fig_fullvol_vs_fg_2d_es.png", dpi=160)
print("wrote fig_fullvol_vs_fg_2d_es.png")

# ---- Figure 2: accuracy vs compute cost (Spanish) ----
fig, ax = plt.subplots(figsize=(7, 5))
for rd in glob.glob("run/outputs/*_2d/"):
    rj = os.path.join(rd, "results.json")
    if not os.path.isfile(rj):
        continue
    d = json.load(open(rj))
    if int(d.get("spatial_dims", 0)) != 2:
        continue
    t = d["training_time_seconds"] / 60.0
    col = {"quantum": "C3", "hybrid": "C1", "classical": "C0"}.get(d.get("model_type"), "C7")
    ax.scatter(t, d["test_mean_dice"], c=col, s=40)
    ax.annotate(d["model_name"], (t, d["test_mean_dice"]), fontsize=7)
ax.set_xscale("log")
ax.set_xlabel("tiempo de entrenamiento (min, escala log)")
ax.set_ylabel("Dice medio (conjunto de prueba)")
ax.set_title("2D: precision frente a costo computacional")
fig.tight_layout()
fig.savefig("paper/figures/fig_cost_2d_es.png", dpi=160)
print("wrote fig_cost_2d_es.png")
