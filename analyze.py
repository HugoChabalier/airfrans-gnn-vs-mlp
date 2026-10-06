"""Break the GNN's integrated-drag error into pressure and viscous parts, and check lift.

Usage:  python analyze.py [--task full]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import PROC_DIR, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="full")
    args = ap.parse_args()

    ref = pd.read_csv(PROC_DIR / "scalars.csv").set_index("name")
    gnn = pd.read_csv(RESULTS_DIR / f"gnn_{args.task}_test.csv").set_index("name")
    mlp = pd.read_csv(RESULTS_DIR / f"mlp_{args.task}_test.csv").set_index("name")
    df = gnn.join(ref[["cd_p", "cd_v", "U", "aoa_deg"]])

    rows = []
    for label, true, pred in [("Cd total", df.cd_true, df.cd_pred),
                              ("Cd pressure part", df.cd_p, df.cd_p_pred),
                              ("Cd viscous part", df.cd_v, df.cd_v_pred),
                              ("Cl", df.cl_true, df.cl_pred)]:
        err = pred - true
        rows.append({"quantity": label, "true mean": true.mean(), "pred mean": pred.mean(),
                     "MAE": np.abs(err).mean(), "MAE / mean|true|": np.abs(err).mean() / np.abs(true).mean(),
                     "Spearman": spearmanr(true, pred).statistic})
    print(f"GNN, task={args.task}: integrated quantities from predicted fields")
    print(pd.DataFrame(rows).set_index("quantity").to_string(float_format=lambda v: f"{v:.4g}"))

    print("\nGNN field errors (relative L2, mean over test sims):")
    print(df[[c for c in df.columns if c.startswith("relL2_")]].mean().to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nMLP for reference:")
    for q in ("cd", "cl"):
        err = mlp[f"{q}_pred"] - mlp[f"{q}_true"]
        print(f"  {q}: MAE {np.abs(err).mean():.4g}  Spearman {spearmanr(mlp[f'{q}_true'], mlp[f'{q}_pred']).statistic:.4f}")

    # Does the GNN drag error grow with angle of attack (separation) or speed?
    df["abs_err"] = np.abs(df.cd_pred - df.cd_true)
    print("\nGNN |Cd error| vs angle of attack:")
    print(df.groupby(pd.cut(df.aoa_deg, [-5, 0, 5, 10, 15])).abs_err.agg(["count", "mean"])
          .to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
