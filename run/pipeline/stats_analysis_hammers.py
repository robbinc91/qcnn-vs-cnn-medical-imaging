"""Statistical analysis of per-subject Dice across all HAMMERS sweep3 models.

Inputs : runhammers/analysis/per_subject_dice.csv (long format, 5 test subjects)
Outputs: runhammers/analysis/stats_report.txt
         runhammers/analysis/fig_*.png
         runhammers/analysis/summary_table.csv

Given only 5 held-out test subjects, formal per-subject tests are
underpowered (two-sided Wilcoxon at n=5 cannot reach p<0.05). Descriptive
stats + bootstrap 95% CIs of paired differences + paired Cohen's d are the
primary evidence, with Wilcoxon/Friedman p-values as secondary,
Holm-corrected, and clearly flagged as exploratory.
"""
import os
import csv
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

A = "runhammers/analysis"
CSV = os.path.join(A, "per_subject_dice.csv")
RNG = np.random.RandomState(42)

CLASS_NAMES = [
    "left_superior_parietal", "right_superior_parietal",
    "left_supramarginal", "right_supramarginal",
    "left_angular", "right_angular",
    "left_postcentral", "right_postcentral",
]

# quaternion family -> classical counterpart (paired by architecture)
QUAT_PAIRS_2D = [("quaternion_marin_2d", "marin_2d"),
                 ("quaternion_mipaim_2d", "mipaim_unet_2d"),
                 ("quaternion_acapulco_2d", "acapulco_2d"),
                 ("quaternion_unet_2d", "swin_unetr_2d")]
QUAT_PAIRS_3D = [("quaternion_marin_3d", "marin_3d"),
                 ("quaternion_mipaim_3d", "mipaim_unet_3d"),
                 ("quaternion_acapulco_3d", "acapulco_3d"),
                 ("quaternion_unet_3d", "swin_unetr_3d")]


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            r["dims"] = int(r["dims"]); r["subject"] = int(r["subject"])
            r["params"] = int(r["params"])
            for k in ["mean_dice"] + ["dice_c%d" % c for c in range(1, 9)]:
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
    lines.append(f"  {a:26} vs {b:26} | Δ={diff.mean():+.4f} "
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
    lines = ["=" * 82,
             "STATISTICAL ANALYSIS — HAMMERS per-subject Dice (n=5 test subjects)",
             "=" * 82]

    # ---- descriptive ----
    lines.append("\n## Descriptive (mean +/- std over 5 subjects)\n")
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

    # ---- Friedman omnibus per dims ----
    lines.append("\n## Friedman omnibus (do models differ within a dimension?)\n")
    for dims in (2, 3):
        sel = [s["run"] for s in summ if s["dims"] == dims]
        mat = [vec(rows, r) for r in sel]
        if all(len(v) == 5 for v in mat) and len(mat) >= 3:
            chi, p = stats.friedmanchisquare(*mat)
            lines.append(f"  {dims}D: chi2={chi:.2f}, p={p:.3f}  ({len(sel)} models)")

    # ---- headline paired comparisons ----
    lines.append("\n## Paired comparisons (bootstrap CI primary; Wilcoxon exploratory)\n")
    fam = []
    lines.append(" [2D] best quaternion vs its classical base; best quantum vs best classical:")
    for a, b in [("quaternion_marin_2d", "marin_2d"),
                 ("qcnn_pennylane_2d", "marin_2d"),
                 ("hybrid_qcnn_2d", "marin_2d")]:
        r = paired(rows, a, b, lines)
        fam.append(r) if r else None
    lines.append("\n [3D] best quaternion vs its classical base; best quantum vs best classical:")
    for a, b in [("quaternion_marin_3d", "marin_3d"),
                 ("qcnn_pennylane_3d", "marin_3d"),
                 ("qcnn_qiskit_3d", "classical_cnn_3d")]:
        r = paired(rows, a, b, lines)
        fam.append(r) if r else None

    # ---- quaternion vs classical family ----
    lines.append("\n## Quaternion vs classical counterpart (per architecture)\n")
    for dims, pairs in ((2, QUAT_PAIRS_2D), (3, QUAT_PAIRS_3D)):
        lines.append(f" [{dims}D]")
        for a, b in pairs:
            r = paired(rows, a, b, lines)
            fam.append(r) if r else None

    # ---- pooled quantum/hybrid vs classical; quaternion vs classical ----
    lines.append("\n## Pooled comparisons (per-subject means)\n")
    for dims in (2, 3):
        q = [s["run"] for s in summ if s["dims"] == dims and s["type"] in ("quantum", "hybrid")]
        c = [s["run"] for s in summ if s["dims"] == dims and s["type"] == "classical"]
        qu = [s["run"] for s in summ if s["dims"] == dims and s["type"] == "quaternion"]
        subs = sorted(set(r["subject"] for r in rows if r["dims"] == dims))
        def pooled(tag_runs):
            return np.array([np.mean([vec(rows, run)[i] for run in tag_runs])
                             for i in range(len(subs))])
        qm, cm = pooled(q), pooled(c)
        diff = qm - cm
        lo, hi = boot_ci(diff)
        try:
            p = stats.wilcoxon(qm, cm).pvalue if np.any(diff != 0) else 1.0
        except Exception:
            p = float("nan")
        lines.append(f"  {dims}D quantum/hybrid={qm.mean():.4f} classical={cm.mean():.4f} "
                     f"Δ={diff.mean():+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p={p:.3f}")
        if qu:
            um, cm2 = pooled(qu), pooled(c)
            diff2 = um - cm2
            lo2, hi2 = boot_ci(diff2)
            try:
                p2 = stats.wilcoxon(um, cm2).pvalue if np.any(diff2 != 0) else 1.0
            except Exception:
                p2 = float("nan")
            lines.append(f"  {dims}D quaternion={um.mean():.4f} classical={cm2.mean():.4f} "
                         f"Δ={diff2.mean():+.4f} 95%CI[{lo2:+.4f},{hi2:+.4f}] p={p2:.3f}")

    # ---- Holm correction over family ----
    fam = [f for f in fam if f and not np.isnan(f["p"])]
    if fam:
        adj = holm([f["p"] for f in fam])
        lines.append("\n## Holm-corrected p-values (headline family)\n")
        for f, a in zip(fam, adj):
            lines.append(f"  {f['pair']:56} raw={f['p']:.3f} holm={a:.3f}")

    # ---- figures ----
    # Fig 1: per-subject boxplots per dims
    for dims in (2, 3):
        sel = [s for s in summ if s["dims"] == dims]
        sel.sort(key=lambda z: -z["mean"])
        data = [vec(rows, s["run"]) for s in sel]
        labels = [s["run"].replace(f"_{dims}d", "") for s in sel]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("per-subject mean Dice"); ax.set_title(f"HAMMERS {dims}D models (n=5 subjects)")
        fig.tight_layout()
        fig.savefig(f"{A}/fig_box_{dims}d.png", dpi=130); plt.close(fig)

    # Fig 2: quaternion vs classical slopegraph per dims
    for dims, pairs in ((2, QUAT_PAIRS_2D), (3, QUAT_PAIRS_3D)):
        fig, ax = plt.subplots(figsize=(7, 5))
        for a, b in pairs:
            x = vec(rows, b).mean(); y = vec(rows, a).mean()
            ax.plot([0, 1], [x, y], "-o", label=b.replace(f"_{dims}d", ""))
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["classical", "quaternion"])
        ax.set_ylabel("per-subject mean Dice")
        ax.set_title(f"HAMMERS {dims}D: quaternion vs classical base")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(f"{A}/fig_quat_vs_class_{dims}d.png", dpi=130); plt.close(fig)

    # Fig 3: training time vs Dice (cost of quantum) per dims
    for dims in (2, 3):
        fig, ax = plt.subplots(figsize=(7, 5))
        for rd in sorted(glob.glob("runhammers/*/")):
            rj = os.path.join(rd, "results.json")
            if not os.path.isfile(rj):
                continue
            d = json.load(open(rj))
            if int(d.get("spatial_dims", 0)) != dims:
                continue
            t = d["training_time_seconds"] / 3600.0
            col = {"quantum": "C3", "hybrid": "C1", "classical": "C0", "quaternion": "C2"}.get(
                d.get("model_type"), "C7")
            ax.scatter(t, d["test_mean_dice"], c=col, s=40)
            ax.annotate(d["model_name"], (t, d["test_mean_dice"]), fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel("training time (h, log)"); ax.set_ylabel("test mean Dice")
        ax.set_title(f"HAMMERS {dims}D: accuracy vs compute cost")
        fig.tight_layout(); fig.savefig(f"{A}/fig_cost_{dims}d.png", dpi=130); plt.close(fig)

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
