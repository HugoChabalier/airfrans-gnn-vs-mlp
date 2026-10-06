"""Simple message-passing GNN predicting (u, v, p, nut) fields on AirfRANS k-NN graphs.

After training, drag/lift are also obtained by integrating the *predicted* surface fields,
so they can be compared with the direct MLP regression (compare.py).

Usage:  python train_gnn.py [--task scarce] [--epochs 40] [--n-nodes 8000]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

from common import (RESULTS_DIR, AirfransGraphs, compute_stats, dim_targets, force_coefficients,
                    load_sample, load_split, make_graph, subsample,
                    surface_order)

CHANNELS = ["u_x", "u_y", "p", "nut"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class EdgeConv(MessagePassing):
    """h_i <- h_i + MLP(h_i, mean_j MLP(h_i, h_j, e_ij))."""

    def __init__(self, h, e):
        super().__init__(aggr="mean")
        self.msg = mlp(2 * h + e, h, h)
        self.upd = mlp(2 * h, h, h)
        self.norm = nn.LayerNorm(h)

    def forward(self, h, edge_index, edge_attr):
        m = self.propagate(edge_index, h=h, e=edge_attr)
        return self.norm(h + self.upd(torch.cat([h, m], dim=1)))

    def message(self, h_i, h_j, e):
        return self.msg(torch.cat([h_i, h_j, e], dim=1))


class FieldGNN(nn.Module):
    def __init__(self, in_dim=7, edge_dim=3, hidden=64, layers=5, out_dim=4):
        super().__init__()
        self.enc = mlp(in_dim, hidden, hidden)
        self.convs = nn.ModuleList(EdgeConv(hidden, edge_dim) for _ in range(layers))
        self.dec = mlp(hidden, hidden, out_dim)

    def forward(self, data):
        h = self.enc(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)
        return self.dec(h)


@torch.no_grad()
def evaluate(model, names, stats, args, max_n=None):
    """Field errors (relative L2 per channel, dimensional) + Cd/Cl from predicted fields."""
    model.eval()
    rows = []
    for i, name in enumerate(names[:max_n]):
        d = subsample(load_sample(name), args.n_nodes, np.random.default_rng(1000 + i))
        pred = model(make_graph(d, stats, args.k).to(DEVICE)).cpu().numpy() * stats["y_std"] + stats["y_mean"]
        y_pred = dim_targets(d["x"], pred)
        err = np.linalg.norm(y_pred - d["y"], axis=0) / (np.linalg.norm(d["y"], axis=0) + 1e-12)
        c_true = force_coefficients(d["pos"], d["x"], d["y"], d["surf"], d["wall_nb"])
        c_pred = force_coefficients(d["pos"], d["x"], y_pred, d["surf"], d["wall_nb"])
        rows.append({"name": name, **{f"relL2_{c}": e for c, e in zip(CHANNELS, err)},
                     "cd_true": c_true["cd"], "cd_pred": c_pred["cd"],
                     "cd_p_pred": c_pred["cd_p"], "cd_v_pred": c_pred["cd_v"],
                     "cl_true": c_true["cl"], "cl_pred": c_pred["cl"]})
    return pd.DataFrame(rows)


def plot_sample(model, name, stats, args, path):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.path import Path
    model.eval()
    d = subsample(load_sample(name), args.n_nodes, np.random.default_rng(0))
    with torch.no_grad():
        pred = model(make_graph(d, stats, args.k).to(DEVICE)).cpu().numpy() * stats["y_std"] + stats["y_mean"]
    y_pred = dim_targets(d["x"], pred)
    tri = mtri.Triangulation(d["pos"][:, 0], d["pos"][:, 1])
    # Drop triangles whose centroid lies inside the airfoil outline.
    s, n_s = d["pos"][d["surf"]], d["x"][d["surf"], 3:5]
    outline = Path(s[surface_order(s, n_s)])
    centroids = d["pos"][tri.triangles].mean(axis=1)
    tri.set_mask(outline.contains_points(centroids))
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for r, (c, label) in enumerate([(0, "u_x [m/s]"), (1, "u_y [m/s]"), (2, "p/rho [m²/s²]")]):
        vmin, vmax = np.percentile(d["y"][:, c], [1, 99])
        for col, (field, title) in enumerate([(d["y"][:, c], "CFD"), (y_pred[:, c], "GNN")]):
            im = axes[r, col].tripcolor(tri, field, shading="gouraud", cmap="viridis",
                                        vmin=vmin, vmax=vmax)
            axes[r, col].set_title(f"{title}  {label}")
            fig.colorbar(im, ax=axes[r, col])
        im = axes[r, 2].tripcolor(tri, np.abs(y_pred[:, c] - d["y"][:, c]), shading="gouraud",
                                  cmap="magma")
        axes[r, 2].set_title(f"|error|  {label}")
        fig.colorbar(im, ax=axes[r, 2])
    for ax in axes.flat:
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
    fig.suptitle(name)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="scarce", choices=["full", "scarce", "reynolds", "aoa"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-nodes", type=int, default=8000, help="nodes sampled per mesh")
    ap.add_argument("--k", type=int, default=10, help="neighbours in the k-NN graph")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=None, help="cap #training sims (quick tests)")
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0, help="data-loading processes (use >0 on HPC)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    train, val, test = load_split(args.task, args.seed)
    train = train[: args.max_train]
    print(f"task={args.task}: {len(train)} train / {len(val)} val / {len(test)} test sims")
    stats = compute_stats(train)

    loader = DataLoader(AirfransGraphs(train, stats, args.n_nodes, args.k, seed=args.seed),
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers)
    model = FieldGNN(hidden=args.hidden, layers=args.layers).to(DEVICE)
    print("device:", DEVICE)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader))

    RESULTS_DIR.mkdir(exist_ok=True)
    ckpt = RESULTS_DIR / f"gnn_{args.task}.pt"
    best = np.inf
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            loss = ((model(batch) - batch.y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v = evaluate(model, val, stats, args)
        v_score = v[[f"relL2_{c}" for c in CHANNELS[:3]]].mean().mean()
        print(f"epoch {ep:3d}  train loss {tot / len(loader):.4f}  val relL2(u,v,p) {v_score:.4f}"
              f"  val Cd MAE {np.abs(v.cd_pred - v.cd_true).mean():.5f}  ({time.time() - t0:.0f}s)")
        if v_score < best:
            best = v_score
            torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args)}, ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False)["model"])
    res = evaluate(model, test, stats, args, args.max_test)
    res.to_csv(RESULTS_DIR / f"gnn_{args.task}_test.csv", index=False)
    print("[GNN] test relative L2 error per field:")
    print(res[[f"relL2_{c}" for c in CHANNELS]].mean().to_string())
    rel = np.abs(res.cd_pred - res.cd_true) / np.abs(res.cd_true)
    print(f"[GNN] Cd from integrated predicted fields: MAE {np.abs(res.cd_pred - res.cd_true).mean():.5f}"
          f"  mean rel err {rel.mean():.3%}")
    plot_sample(model, test[0], stats, args, RESULTS_DIR / f"gnn_{args.task}_fields.png")
    print("saved", RESULTS_DIR / f"gnn_{args.task}_fields.png")


if __name__ == "__main__":
    main()
