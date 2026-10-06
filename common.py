"""Shared paths, physics post-processing and dataset utilities for the AirfRANS project."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

# Keep the (large) data outside OneDrive so it does not get synced.
DATA_ROOT = Path(os.environ.get("AIRFRANS_DATA", Path.home() / "data" / "airfrans"))
RAW_DIR = DATA_ROOT / "raw"
PROC_DIR = DATA_ROOT / "processed"
RESULTS_DIR = Path(__file__).parent / "results"

NU = 1.56e-5  # kinematic viscosity of air used in AirfRANS [m^2/s]
CHORD = 1.0


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def load_split(task: str, rng_seed: int = 0, val_frac: float = 0.1):
    """Return (train, val, test) lists of simulation names for an AirfRANS task.

    task in {"full", "scarce", "reynolds", "aoa"}. The scarce task shares the full test set.
    A validation set is carved out of the training set for model selection.
    """
    manifest = json.loads((PROC_DIR / "manifest.json").read_text())
    train = list(manifest[f"{task}_train"])
    test = list(manifest["full_test" if task == "scarce" else f"{task}_test"])
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(train)
    n_val = max(1, int(round(val_frac * len(train))))
    return train[n_val:], train[:n_val], test


# ---------------------------------------------------------------------------
# Physics: force coefficients from fields
# ---------------------------------------------------------------------------
def split_surface(pos_s: np.ndarray, n_s: np.ndarray):
    """Indices of the upper and lower airfoil surfaces, each sorted by increasing x.

    The sign of the normal's y-component flips exactly at the leading and trailing edges
    (where the normal is horizontal), so it separates the two sides for any thickness/camber.
    """
    a, b = np.where(n_s[:, 1] >= 0)[0], np.where(n_s[:, 1] < 0)[0]
    if pos_s[a, 1].mean() < pos_s[b, 1].mean():  # normals point into the airfoil
        a, b = b, a
    return a[np.argsort(pos_s[a, 0])], b[np.argsort(pos_s[b, 0])]


def surface_order(pos_s: np.ndarray, n_s: np.ndarray) -> np.ndarray:
    """Order airfoil surface points as a closed loop: TE -> upper -> LE -> lower -> TE."""
    upper, lower = split_surface(pos_s, n_s)
    return np.concatenate([upper[::-1], lower])


def force_coefficients(pos, x, y, surf, wall_nb):
    """Drag/lift coefficients by integrating pressure and wall shear over the airfoil.

    pos (N,2); x (N,5) = [u_inf_x, u_inf_y, sdf, n_x, n_y]; y (N,4) = [u_x, u_y, p/rho, nut];
    surf (N,) bool; wall_nb (n_surf,) index of the first off-wall node above each surface node
    (in the order of np.where(surf)).
    Wall shear uses tau = nu * u_t / d with the first off-wall node (wall-resolved mesh, y+ ~ 1).
    Returns dict with cd, cl and pressure/viscous parts of cd.
    """
    s_idx = np.where(surf)[0]
    order = surface_order(pos[s_idx], x[s_idx, 3:5])
    s_idx, nb = s_idx[order], wall_nb[order]

    P = pos[s_idx]
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    ds = 0.5 * (seg + np.roll(seg, 1))

    n = x[s_idx, 3:5].astype(np.float64)
    # Orient normals into the fluid (towards the off-wall neighbour).
    if np.sum((pos[nb] - P) * n) < 0:
        n = -n

    p = y[s_idx, 2]
    f_p = -(p[:, None] * n * ds[:, None]).sum(0)

    u_nb = y[nb, :2].astype(np.float64)
    u_t = u_nb - (u_nb * n).sum(1, keepdims=True) * n
    d = np.maximum(x[nb, 2].astype(np.float64), 1e-12)
    tau = NU * u_t / d[:, None]
    f_v = (tau * ds[:, None]).sum(0)

    u_inf = x[0, :2].astype(np.float64)
    U = np.linalg.norm(u_inf)
    e_d = u_inf / U
    e_l = np.array([-e_d[1], e_d[0]])
    q = 0.5 * U**2 * CHORD
    return {
        "cd": float((f_p + f_v) @ e_d / q),
        "cl": float((f_p + f_v) @ e_l / q),
        "cd_p": float(f_p @ e_d / q),
        "cd_v": float(f_v @ e_d / q),
    }


def geometry_descriptor(pos_s: np.ndarray, n_s: np.ndarray, n_stations: int = 16) -> np.ndarray:
    """Fixed-size airfoil shape descriptor: upper & lower y at cosine-spaced chord stations."""
    xs = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_stations + 2)[1:-1])) * CHORD
    upper, lower = split_surface(pos_s, n_s)
    return np.concatenate([np.interp(xs, pos_s[i, 0], pos_s[i, 1]) for i in (upper, lower)]
                          ).astype(np.float32)


# ---------------------------------------------------------------------------
# Graph dataset
# ---------------------------------------------------------------------------
def load_sample(name: str) -> dict:
    return torch.load(PROC_DIR / "sims" / f"{name}.pt", weights_only=False)


def subsample(d: dict, n_nodes: int, rng: np.random.Generator) -> dict:
    """Random node subset that always keeps surface nodes and their off-wall neighbours."""
    N = d["pos"].shape[0]
    surf = d["surf"]
    must = np.unique(np.concatenate([np.where(surf)[0], d["wall_nb"]]))
    if n_nodes is None or n_nodes >= N:
        keep = np.arange(N)
    else:
        rest = np.setdiff1d(np.arange(N), must)
        extra = rng.choice(rest, max(0, n_nodes - len(must)), replace=False)
        keep = np.sort(np.concatenate([must, extra]))
    remap = -np.ones(N, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    out = {k: d[k][keep] for k in ("pos", "x", "y", "surf")}
    out["wall_nb"] = remap[d["wall_nb"]]
    return out


def knn_edges(pos: np.ndarray, k: int) -> torch.Tensor:
    """Directed k-NN graph (edges j -> i for the k nearest neighbours j of every node i)."""
    _, nbr = cKDTree(pos).query(pos, k=k + 1)
    dst = np.repeat(np.arange(len(pos)), k)
    src = nbr[:, 1:].reshape(-1)
    return torch.from_numpy(np.stack([src, dst])).long()


def nondim_targets(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[u/U, v/U, Cp = p/(0.5 U^2), nut/nu] - removes most of the inlet-speed dependence."""
    U = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
    return np.concatenate([y[:, :2] / U, y[:, 2:3] / (0.5 * U**2), y[:, 3:4] / NU], axis=1)


def dim_targets(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    U = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
    return np.concatenate([t[:, :2] * U, t[:, 2:3] * 0.5 * U**2, t[:, 3:4] * NU], axis=1)


def node_features(pos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """[x, y, u_inf_x, u_inf_y, sdf, n_x, n_y]."""
    return np.concatenate([pos, x], axis=1).astype(np.float32)


def edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Scale-free edge features: unit direction + log10 distance (mesh spacing spans ~6 decades)."""
    rel = pos[edge_index[0]] - pos[edge_index[1]]
    dist = rel.norm(dim=1, keepdim=True)
    return torch.cat([rel / (dist + 1e-12), torch.log10(dist + 1e-8)], dim=1)


class AirfransGraphs(torch.utils.data.Dataset):
    """Loads preprocessed simulations and turns them into k-NN graphs on the fly."""

    def __init__(self, names, stats, n_nodes=8000, k=10, seed=0, fixed=False):
        self.names, self.n_nodes, self.k = list(names), n_nodes, k
        self.stats = stats
        self.fixed = fixed  # same subsample every time (for evaluation)
        self.seed = seed

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        # Fresh entropy when training so DataLoader workers (forked copies) do not repeat subsamples.
        rng = np.random.default_rng(self.seed + i) if self.fixed else np.random.default_rng()
        d = subsample(load_sample(self.names[i]), self.n_nodes, rng)
        data = make_graph(d, self.stats, self.k)
        data.name = self.names[i]
        return data


def make_graph(d: dict, stats: dict, k: int) -> Data:
    """Build a normalised k-NN graph (features, edge features, targets) from a (sub)sampled sim."""
    feat = (node_features(d["pos"], d["x"]) - stats["x_mean"]) / stats["x_std"]
    tgt = (nondim_targets(d["x"], d["y"]) - stats["y_mean"]) / stats["y_std"]
    pos = torch.from_numpy(d["pos"]).float()
    ei = knn_edges(d["pos"], k)
    return Data(x=torch.from_numpy(feat).float(), y=torch.from_numpy(tgt).float(), pos=pos,
                edge_index=ei, edge_attr=edge_features(pos, ei), surf=torch.from_numpy(d["surf"]))


def compute_stats(names, max_samples=100):
    """Per-channel mean/std of node features and (non-dimensional) targets over training sims."""
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for name in names[:max_samples]:
        d = subsample(load_sample(name), 4000, rng)
        xs.append(node_features(d["pos"], d["x"]))
        ys.append(nondim_targets(d["x"], d["y"]))
    X, Y = np.concatenate(xs), np.concatenate(ys)
    std = lambda a: np.where(a.std(0) < 1e-8, 1.0, a.std(0)).astype(np.float32)
    return {"x_mean": X.mean(0).astype(np.float32), "x_std": std(X),
            "y_mean": Y.mean(0).astype(np.float32), "y_std": std(Y)}
