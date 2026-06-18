"""Statistical analysis of per-subject Dice across all models.

Inputs : run/outputs/analysis/per_subject_dice.csv (long format, 6 test subjects)
Outputs: run/outputs/analysis/stats_report.txt
         run/outputs/analysis/fig_*.png
         run/outputs/analysis/summary_table.csv

Given only 6 held-out test subjects, formal per-subject tests are
underpowered (two-sided Wilcoxon at n=6 cannot reach p<0.05). We therefore
report descriptive stats + bootstrap 95% CIs of paired differences + paired
Cohen's d as the primary evidence, with Wilcoxon/Friedman p-values as
secondary, Holm-corrected, and clearly flagged as exploratory.
"""
import os
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

A = "run/outputs/analysis"
CSV = os.path.join(A, "per_subject_dice.csv")
RNG = np.random.RandomState(42)


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            r["dims"] = int(r["dims"]); r["subject"] = int(r["subject"])
            r["params"] = int(r["params"])
            for k in ("mean_dice", "dice_c1", "dice_c2", "dice_c3"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def vec(rows, run):
    """Per-subject mean_dice vector ordered by subject id, for a run tag."""
    d = {r["subject"]: r["mean_dice"] for r in rows if r["run"] == run}
    return np.array([d[s] for s in sorted(d)])


def boot_ci(diff, n=10000):
    means = [RNG.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def paired(rows, a, b, lines):
    x, y = vec(rows, a), vec(rows, b)
    if len(x) != len(y) or len(x) == 0:
        lines.append(f"  {a} vs {b}: MISSING"); return None
    diff = x - y
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    lo, hi = boot_ci(diff)
    try:
        w_p = stats.wilcoxon(x, y).pvalue if np.any(diff != 0) else 1.0
    except Exception:
        w_p = float("nan")
    lines.append(f"  {a:22} vs {b:22} | Δ={diff.mean():+.4f} "
                 f"95%CI[{lo:+.4f},{hi:+.4f}] d={d:+.2f} wilcoxon_p={w_p:.3f}")
    return {"pair": f"{a} vs {b}", "delta": diff.mean(), "ci": (lo, hi),
            "d": d, "p": w_p}


def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        v = (m - rank) * pvals[i]
        prev = max(prev, v); adj[i] = min(prev, 1.0)
    return adj


def main():
    rows = load()
    runs = sorted(set(r["run"] for r in rows))
    lines = ["=" * 78, "STATISTICAL ANALYSIS — per-subject Dice (n=6 test subjects)", "=" * 78]

    # ---- descriptive ----
    lines.append("\n## Descriptive (mean +/- std over 6 subjects)\n")
    summ = []
    for run in runs:
        sub = [r for r in rows if r["run"] == run]
        md = np.array([r["mean_dice"] for r in sub])
        typ = sub[0]["type"]; dims = sub[0]["dims"]; params = sub[0]["params"]
        summ.append(dict(run=run, dims=dims, type=typ, params=params,
                         mean=md.mean(), std=md.std(ddof=1), n=len(md)))
    summ.sort(key=lambda z: (z["dims"], -z["mean"]))
    for dims in (2, 3):
        lines.append(f"--- {dims}D ---")
        for s in [x for x in summ if x["dims"] == dims]:
            lines.append(f"  {s['run']:26} {s['type']:9} {s['params']/1e6:6.2f}M  "
                         f"Dice={s['mean']:.4f}±{s['std']:.4f}")

    # ---- Friedman omnibus per dims (only same-budget 17M / classical runs) ----
    lines.append("\n## Friedman omnibus (do models differ within a dimension?)\n")
    for dims in (2, 3):
        sel = [s["run"] for s in summ if s["dims"] == dims and "_6M" not in s["run"]]
        mat = [vec(rows, r) for r in sel]
        if all(len(v) == 6 for v in mat) and len(mat) >= 3:
            chi, p = stats.friedmanchisquare(*mat)
            lines.append(f"  {dims}D: chi2={chi:.2f}, p={p:.3f}  ({len(sel)} models)")

    # ---- headline paired comparisons ----
    lines.append("\n## Paired comparisons (bootstrap CI primary; Wilcoxon exploratory)\n")
    fam = []
    lines.append(" [2D] best quantum vs best classical & param-matched field:")
    for a, b in [("qcnn_qiskit_2d_17M", "marin_2d"),
                 ("qcnn_qiskit_2d_17M", "mipaim_unet_2d"),
                 ("hybrid_qcnn_2d_17M", "marin_2d")]:
        r = paired(rows, a, b, lines); fam.append(r) if r else None
    lines.append("\n [2D] 6M vs 17M (does a wider classical backbone help the QCNN?):")
    for a, b in [("qcnn_pennylane_2d_17M", "qcnn_pennylane_2d_6M"),
                 ("qcnn_qiskit_2d_17M", "qcnn_qiskit_2d_6M"),
                 ("hybrid_qcnn_2d_17M", "hybrid_qcnn_2d_6M")]:
        r = paired(rows, a, b, lines); fam.append(r) if r else None
    lines.append("\n [3D] best quantum vs best classical:")
    for a, b in [("qcnn_pennylane_3d_17M", "mipaim_unet_3d"),
                 ("qcnn_pennylane_3d_17M", "acapulco_3d")]:
        r = paired(rows, a, b, lines); fam.append(r) if r else None

    # ---- pooled quantum vs classical, per subject ----
    lines.append("\n## Pooled quantum/hybrid vs classical (per-subject means)\n")
    for dims in (2, 3):
        q = [s["run"] for s in summ if s["dims"] == dims and s["type"] in ("quantum", "hybrid")
             and "_6M" not in s["run"]]
        c = [s["run"] for s in summ if s["dims"] == dims and s["type"] == "classical"]
        subs = sorted(set(r["subject"] for r in rows if r["dims"] == dims))
        qm = np.array([np.mean([vec(rows, run)[i] for run in q]) for i in range(len(subs))])
        cm = np.array([np.mean([vec(rows, run)[i] for run in c]) for i in range(len(subs))])
        diff = qm - cm
        lo, hi = boot_ci(diff)
        try:
            p = stats.wilcoxon(qm, cm).pvalue
        except Exception:
            p = float("nan")
        lines.append(f"  {dims}D: quantum={qm.mean():.4f} classical={cm.mean():.4f} "
                     f"Δ={diff.mean():+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p={p:.3f}")

    # ---- Holm correction over headline family ----
    fam = [f for f in fam if f and not np.isnan(f["p"])]
    if fam:
        adj = holm([f["p"] for f in fam])
        lines.append("\n## Holm-corrected p-values (headline family)\n")
        for f, a in zip(fam, adj):
            lines.append(f"  {f['pair']:48} raw={f['p']:.3f} holm={a:.3f}")

    # ---- figures ----
    # Fig 1: per-subject boxplots per dims
    for dims in (2, 3):
        sel = [s for s in summ if s["dims"] == dims and "_6M" not in s["run"]]
        sel.sort(key=lambda z: -z["mean"])
        data = [vec(rows, s["run"]) for s in sel]
        labels = [s["run"].replace(f"_{dims}d", "").replace("_17M", "") for s in sel]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylabel("per-subject mean Dice"); ax.set_title(f"{dims}D models (n=6 subjects)")
        ax.tick_params(axis="x", rotation=45); fig.tight_layout()
        fig.savefig(f"{A}/fig_box_{dims}d.png", dpi=130); plt.close(fig)

    # Fig 2: 6M vs 17M paired (2D quantum/hybrid)
    fig, ax = plt.subplots(figsize=(7, 5))
    for run6, run17, name in [("qcnn_pennylane_2d_6M", "qcnn_pennylane_2d_17M", "pennylane"),
                              ("qcnn_qiskit_2d_6M", "qcnn_qiskit_2d_17M", "qiskit"),
                              ("hybrid_qcnn_2d_6M", "hybrid_qcnn_2d_17M", "hybrid")]:
        v6, v17 = vec(rows, run6), vec(rows, run17)
        if len(v6) and len(v17):
            ax.plot([0, 1], [v6.mean(), v17.mean()], "-o", label=name)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["6.1M (3-stage)", "16.9M (4-stage)"])
    ax.set_ylabel("mean Dice"); ax.set_title("2D QCNN: backbone width vs Dice")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{A}/fig_6M_vs_17M.png", dpi=130); plt.close(fig)

    # Fig 3: training time vs Dice (cost of quantum) from results.json
    fig, ax = plt.subplots(figsize=(7, 5))
    import glob
    for rd in glob.glob("run/outputs/*/"):
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
    ax.set_xscale("log"); ax.set_xlabel("training time (min, log)"); ax.set_ylabel("test mean Dice")
    ax.set_title("2D: accuracy vs compute cost")
    fig.tight_layout(); fig.savefig(f"{A}/fig_cost_2d.png", dpi=130); plt.close(fig)

    # ---- write ----
    with open(f"{A}/summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "dims", "type", "params", "mean", "std", "n"])
        w.writeheader(); w.writerows(summ)
    report = "\n".join(lines)
    open(f"{A}/stats_report.txt", "w").write(report)
    print(report)
    print(f"\nFigures + report written to {A}/")


if __name__ == "__main__":
    main()
