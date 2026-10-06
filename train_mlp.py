"""MLP baseline: (inlet speed, angle of attack, airfoil shape descriptor) -> (Cd, Cl).

Usage:  python train_mlp.py [--task scarce]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import PROC_DIR, RESULTS_DIR, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="scarce", choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    df = pd.read_csv(PROC_DIR / "scalars.csv").set_index("name")
    train, val, test = load_split(args.task, args.seed)
    feat_cols = ["U", "aoa_deg"] + [c for c in df.columns if c.startswith("g")]
    tgt_cols = ["cd", "cl"]

    def arrays(names):
        sub = df.loc[names]
        return sub[feat_cols].values.astype(np.float32), sub[tgt_cols].values.astype(np.float32)

    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = arrays(train), arrays(val), arrays(test)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = Ytr.mean(0), Ytr.std(0)
    t = lambda a, m, s: torch.from_numpy((a - m) / s)
    Xtr_t, Ytr_t, Xva_t, Xte_t = t(Xtr, xm, xs), t(Ytr, ym, ys), t(Xva, xm, xs), t(Xte, xm, xs)

    h = args.hidden
    model = nn.Sequential(nn.Linear(len(feat_cols), h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                          nn.Linear(h, h), nn.GELU(), nn.Linear(h, len(tgt_cols)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    predict = lambda X: model(X).detach().numpy() * ys + ym
    best, best_state = np.inf, None
    for ep in range(args.epochs):  # full-batch training: the dataset is tiny
        model.train()
        loss = ((model(Xtr_t) - Ytr_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        model.eval()
        val_mae = np.abs(predict(Xva_t)[:, 0] - Yva[:, 0]).mean()
        if val_mae < best:
            best, best_state = val_mae, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 500 == 0:
            print(f"epoch {ep:5d}  train mse {loss.item():.4f}  val Cd MAE {val_mae:.5f}")
    model.load_state_dict(best_state)

    P = predict(Xte_t)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame({"name": test, "cd_true": Yte[:, 0], "cd_pred": P[:, 0],
                        "cl_true": Yte[:, 1], "cl_pred": P[:, 1]})
    out.to_csv(RESULTS_DIR / f"mlp_{args.task}_test.csv", index=False)
    rel = np.abs(out.cd_pred - out.cd_true) / np.abs(out.cd_true)
    print(f"[MLP] test Cd MAE {np.abs(out.cd_pred - out.cd_true).mean():.5f}  "
          f"mean rel err {rel.mean():.3%}  |  Cl MAE {np.abs(out.cl_pred - out.cl_true).mean():.4f}")


if __name__ == "__main__":
    main()
