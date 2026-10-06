"""Cross-task report: MLP vs GNN on every AirfRANS task that has results.

Reads results/{mlp,gnn}_<task>_test.csv (+ data/processed/scalars.csv and manifest.json if
available, for the drag breakdown and extrapolation plots) and writes to results/report/:
  results.md              tables + figure links, ready to paste into the GitHub README
  summary_scalars.csv     Cd / Cl metrics per task and model
  summary_fields.csv      GNN field errors per task
  fig_parity.png          predicted vs true Cd, every task x model
  fig_cd_error.png        Cd error per task, MLP vs GNN
  fig_field_errors.png    GNN relative L2 error per field and task
  fig_extrapolation.png   Cd error vs Reynolds / AoA for the extrapolation tasks

Usage:  python report.py
"""
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import CHORD, NU, PROC_DIR, RESULTS_DIR

TASKS = ["full", "scarce", "reynolds", "aoa"]
TASK_LABEL = {"full": "full (800 train)", "scarce": "scarce (200 train)",
              "reynolds": "reynolds (extrapolation)", "aoa": "aoa (extrapolation)"}
MODELS = ["MLP", "GNN"]
FIELDS = ["u_x", "u_y", "p", "nut"]
OUT = RESULTS_DIR / "report"

# Validated categorical palette (slots 1-4), text and grid tokens.
COLORS = {"MLP": "#2a78d6", "GNN": "#eb6834"}
FIELD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
})


def load_runs():
    runs = {}
    for task in TASKS:
        for model in MODELS:
            f = RESULTS_DIR / f"{model.lower()}_{task}_test.csv"
            if f.exists():
                runs[task, model] = pd.read_csv(f).set_index("name")
    return runs


def scalar_metrics(df):
    out = {}
    for q in ("cd", "cl"):
        err = df[f"{q}_pred"] - df[f"{q}_true"]
        out[f"{q.upper()} MAE"] = np.abs(err).mean()
        out[f"{q.upper()} RMSE"] = np.sqrt((err**2).mean())
        out[f"{q.upper()} Spearman"] = spearmanr(df[f"{q}_true"], df[f"{q}_pred"]).statistic
    out["CD mean rel err [%]"] = 100 * (np.abs(df.cd_pred - df.cd_true) / np.abs(df.cd_true)).mean()
    return out


def to_markdown(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        idx = " / ".join(idx) if isinstance(idx, tuple) else idx
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def fig_parity(runs, tasks):
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.4 * len(tasks), 6.6), squeeze=False)
    for r, model in enumerate(MODELS):
        vals = [v for t in tasks if (t, model) in runs for v in
                (runs[t, model].cd_true, runs[t, model].cd_pred)]
        if not vals:
            continue
        lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
        pad = 0.05 * (hi - lo)
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls="--", zorder=1)
            if (task, model) not in runs:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color=INK_2)
                continue
            df = runs[task, model]
            m = scalar_metrics(df)
            ax.scatter(df.cd_true, df.cd_pred, s=14, color=COLORS[model], alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.5, zorder=2)
            ax.text(0.04, 0.96, f"MAE {m['CD MAE']:.2e}\nρ = {m['CD Spearman']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=9, color=INK)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(TASK_LABEL[task])
            if r == 1:
                ax.set_xlabel("Cd, CFD")
            if c == 0:
                ax.set_ylabel(f"{model}: Cd predicted")
    fig.suptitle("Drag coefficient on the test set: MLP (direct regression) vs GNN (integrated fields)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_parity.png", dpi=150)
    plt.close(fig)


def fig_cd_error(summary, runs, tasks):
    """Dot plot on a log axis: the two models differ by more than an order of magnitude."""
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(tasks) + 1.4))
    y = np.arange(len(tasks))[::-1]
    for model, dy in zip(MODELS, (0.12, -0.12)):
        xs = [summary.loc[(t, model), "CD MAE"] if (t, model) in summary.index else np.nan
              for t in tasks]
        ax.scatter(xs, y + dy, s=64, color=COLORS[model], label=model, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        for xi, yi in zip(xs, y + dy):
            if np.isfinite(xi):
                ax.annotate(f"{xi:.1e}", (xi, yi), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK_2)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, [TASK_LABEL[t] for t in tasks])
    ax.set_ylim(-0.6, len(tasks) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cd mean absolute error on the test set (log scale)")
    typical = np.median(np.concatenate([df.cd_true.values for df in runs.values()]))
    ax.set_title(f"Drag error per task (median test Cd = {typical:.4f})", loc="left", pad=22)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cd_error.png", dpi=150)
    plt.close(fig)


def fig_field_errors(fields, tasks):
    tasks = [t for t in tasks if t in fields.index]
    fig, ax = plt.subplots(figsize=(1.8 * len(tasks) + 2.5, 3.8))
    w = 0.8 / len(FIELDS)
    x = np.arange(len(tasks))
    for i, (f, color) in enumerate(zip(FIELDS, FIELD_COLORS)):
        vals = fields.loc[tasks, f"relL2_{f}"].values
        ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=color, label=f, zorder=2)
    ax.set_xticks(x, [TASK_LABEL[t] for t in tasks], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("relative L2 error")
    ax.set_title("GNN field error per task (lower is better)", loc="left")
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig_field_errors.png", dpi=150)
    plt.close(fig)


def fig_extrapolation(runs, ref, manifest):
    """|Cd error| against the varied parameter, with the training range shaded."""
    panels = [(t, var) for t, var in (("reynolds", "Re"), ("aoa", "aoa_deg"))
              if any((t, m) in runs for m in MODELS)]
    if not panels or ref is None:
        return False
    ref = ref.assign(Re=ref.U * CHORD / NU)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    for ax, (task, var) in zip(axes[0], panels):
        train_vals = ref.loc[ref.index.intersection(manifest[f"{task}_train"]), var]
        ax.axvspan(train_vals.min(), train_vals.max(), color=GRID, alpha=0.6, zorder=0,
                   label="training range")
        for model in MODELS:
            if (task, model) not in runs:
                continue
            df = runs[task, model].join(ref[[var]])
            ax.scatter(df[var], 100 * np.abs(df.cd_pred - df.cd_true) / df.cd_true, s=14,
                       color=COLORS[model], alpha=0.75, label=model, zorder=2,
                       edgecolor=SURFACE, linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("Reynolds number" if var == "Re" else "angle of attack [deg]")
        ax.set_ylabel("|Cd error| [% of Cd]")
        ax.set_title(f"{TASK_LABEL[task]}: Cd error vs {'Reynolds' if var == 'Re' else 'AoA'}",
                     loc="left")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(OUT / "fig_extrapolation.png", dpi=150)
    plt.close(fig)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no results found in results/ - run the training jobs first")
    tasks = [t for t in TASKS if any((t, m) in runs for m in MODELS)]
    print("tasks with results:", tasks)

    summary = pd.DataFrame({k: scalar_metrics(v) for k, v in runs.items()}).T
    summary.index.names = ["task", "model"]
    summary = summary.reindex([(t, m) for t in tasks for m in MODELS if (t, m) in runs])

    ref = manifest = None
    if (PROC_DIR / "scalars.csv").exists() and (PROC_DIR / "manifest.json").exists():
        ref = pd.read_csv(PROC_DIR / "scalars.csv").set_index("name")
        manifest = json.loads((PROC_DIR / "manifest.json").read_text())

    field_rows, breakdown_rows = {}, {}
    for task in tasks:
        if (task, "GNN") not in runs:
            continue
        g = runs[task, "GNN"]
        field_rows[task] = g[[f"relL2_{f}" for f in FIELDS]].mean()
        if ref is not None:
            j = g.join(ref[["cd_p", "cd_v"]])
            breakdown_rows[task] = {
                "pressure part MAE": np.abs(j.cd_p_pred - j.cd_p).mean(),
                "viscous part MAE": np.abs(j.cd_v_pred - j.cd_v).mean(),
                "true pressure part (mean)": j.cd_p.mean(),
                "true viscous part (mean)": j.cd_v.mean(),
            }
    fields = pd.DataFrame(field_rows).T
    fields.index.name = "task"
    breakdown = pd.DataFrame(breakdown_rows).T
    breakdown.index.name = "task"

    summary.to_csv(OUT / "summary_scalars.csv")
    fields.to_csv(OUT / "summary_fields.csv")
    if len(breakdown):
        breakdown.to_csv(OUT / "summary_gnn_drag_breakdown.csv")

    fig_parity(runs, tasks)
    fig_cd_error(summary, runs, tasks)
    if len(fields):
        fig_field_errors(fields, tasks)
    has_extrap = fig_extrapolation(runs, ref, manifest)

    num = lambda v: f"{v:.3g}" if abs(v) >= 1e-2 or v == 0 else f"{v:.2e}"
    md = ["# Results", "",
          "Test-set metrics. **MLP**: (U, AoA, airfoil shape) → (Cd, Cl) directly. "
          "**GNN**: predicts the (u, p, ν_t) fields; Cd/Cl obtained by integrating predicted "
          "surface pressure and wall shear.", "",
          "## Drag and lift", "", to_markdown(summary, num), "",
          "![parity](fig_parity.png)", "", "![drag error](fig_cd_error.png)", ""]
    if len(fields):
        md += ["## GNN field accuracy (relative L2 error)", "", to_markdown(fields, num), "",
               "![field errors](fig_field_errors.png)", ""]
    if len(breakdown):
        md += ["## Where the GNN drag error comes from", "", to_markdown(breakdown, num), ""]
    if has_extrap:
        md += ["## Extrapolation tasks", "", "![extrapolation](fig_extrapolation.png)", ""]
    (OUT / "results.md").write_text("\n".join(md), encoding="utf8")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", num):
        print(summary)
        if len(fields):
            print(fields)
        if len(breakdown):
            print(breakdown)
    print("written to", OUT)


if __name__ == "__main__":
    main()
