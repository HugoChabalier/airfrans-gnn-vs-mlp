"""Compare drag prediction: MLP direct regression vs. Cd integrated from GNN-predicted fields.

Usage:  python compare.py [--task scarce]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS_DIR


def metrics(df):
    err = df.cd_pred - df.cd_true
    return {"n": len(df), "Cd MAE": np.abs(err).mean(), "Cd RMSE": np.sqrt((err**2).mean()),
            "Cd mean rel err": (np.abs(err) / np.abs(df.cd_true)).mean(),
            "Cd Spearman": spearmanr(df.cd_true, df.cd_pred).statistic,
            "Cl MAE": np.abs(df.cl_pred - df.cl_true).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="scarce")
    args = ap.parse_args()

    runs = {}
    for model in ("mlp", "gnn"):
        f = RESULTS_DIR / f"{model}_{args.task}_test.csv"
        if f.exists():
            runs[model.upper()] = pd.read_csv(f)
    if not runs:
        raise SystemExit("no results found - run train_mlp.py / train_gnn.py first")

    table = pd.DataFrame({k: metrics(v) for k, v in runs.items()}).T
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    table.to_csv(RESULTS_DIR / f"compare_{args.task}.csv")

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.6), squeeze=False)
    for ax, (k, df) in zip(axes[0], runs.items()):
        ax.scatter(df.cd_true, df.cd_pred, s=10, alpha=0.7)
        lim = [min(df.cd_true.min(), df.cd_pred.min()), max(df.cd_true.max(), df.cd_pred.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Cd (CFD)")
        ax.set_ylabel("Cd (predicted)")
        ax.set_title(f"{k}: MAE {metrics(df)['Cd MAE']:.4f}, "
                     f"ρ={metrics(df)['Cd Spearman']:.3f}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"compare_{args.task}_cd.png", dpi=120)
    print("saved", RESULTS_DIR / f"compare_{args.task}_cd.png")


if __name__ == "__main__":
    main()
