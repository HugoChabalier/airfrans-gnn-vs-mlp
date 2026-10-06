"""Download AirfRANS (PyG version) and convert it into compact per-simulation files.

The stock torch_geometric AirfRANS loader loads all 1000 simulations (~180k nodes each) into RAM
at once, which does not fit on an 8 GB machine. Here the raw file is memory-mapped and each
simulation is processed independently:
  * reference drag/lift coefficients are computed on the FULL mesh,
  * a geometry descriptor (for the MLP baseline) is extracted from the airfoil surface,
  * the mesh is subsampled to --n-nodes nodes (always keeping surface + first off-wall nodes,
    so force coefficients can still be recomputed exactly from the subsample).

Usage:  python prepare_data.py [--n-nodes 32000] [--delete-raw] [--download-only]
        python prepare_data.py --rebuild-scalars   # recompute Cd/Cl/geometry only
"""
import argparse
import json
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from common import PROC_DIR, RAW_DIR, force_coefficients, geometry_descriptor, surface_order

URL = "https://data.isir.upmc.fr/extrality/pytorch_geometric/AirfRANS.zip"


def download(url, dest):
    """Resumable download with a progress bar."""
    done = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r:
        if done and r.status != 206:  # server ignored Range -> restart
            done = 0
        total = done + int(r.headers.get("Content-Length", 0))
        if done >= total:
            return
        with open(dest, "ab" if done else "wb") as f, tqdm(
            total=total, initial=done, unit="B", unit_scale=True, desc="download"
        ) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def fetch_raw():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pt, manifest = RAW_DIR / "AirfRANS.pt", RAW_DIR / "manifest.json"
    if pt.exists() and manifest.exists():
        return pt, manifest
    zpath = RAW_DIR / "AirfRANS.zip"
    expected = int(urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"))
                   .headers["Content-Length"])
    if not zpath.exists() or zpath.stat().st_size < expected:
        download(URL, zpath)
    print("extracting ...")
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():
            fname = member.split("/")[-1]
            if fname in ("AirfRANS.pt", "manifest.json"):
                with z.open(member) as src, open(RAW_DIR / fname, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
    zpath.unlink()
    return pt, manifest


def process_one(d, n_nodes, rng):
    pos, x, y = (np.asarray(d[k], dtype=np.float32) for k in ("pos", "x", "y"))
    surf = np.asarray(d["surf"]).astype(bool)

    s_idx, v_idx = np.where(surf)[0], np.where(~surf)[0]
    _, j = cKDTree(pos[v_idx]).query(pos[s_idx])
    wall_nb = v_idx[j]  # first off-wall node above each surface node

    coefs = force_coefficients(pos, x, y, surf, wall_nb)

    must = np.unique(np.concatenate([s_idx, wall_nb]))
    rest = np.setdiff1d(np.arange(len(pos)), must)
    keep = np.sort(np.concatenate([must, rng.choice(rest, max(0, n_nodes - len(must)), replace=False)]))
    remap = -np.ones(len(pos), dtype=np.int64)
    remap[keep] = np.arange(len(keep))

    sub = {"pos": pos[keep], "x": x[keep], "y": y[keep], "surf": surf[keep],
           "wall_nb": remap[wall_nb], "n_full": len(pos)}
    # Sanity check: coefficients from the subsample must match the full mesh.
    c_sub = force_coefficients(sub["pos"], sub["x"], sub["y"], sub["surf"], sub["wall_nb"])
    assert abs(c_sub["cd"] - coefs["cd"]) < 1e-6 * max(1.0, abs(coefs["cd"])), (c_sub, coefs)
    return sub


def scalar_row(name, sub):
    """Reference Cd/Cl + MLP inputs. Uses only surface + first off-wall nodes, which every
    processed file keeps from the full mesh, so this equals the full-mesh result."""
    s = sub["surf"]
    pos_s, n_s = sub["pos"][s], sub["x"][s, 3:5]
    loop = pos_s[surface_order(pos_s, n_s)]
    u_inf = sub["x"][0, :2]
    return {
        "name": name,
        "U": float(np.linalg.norm(u_inf)),
        "aoa_deg": float(np.degrees(np.arctan2(u_inf[1], u_inf[0]))),
        **force_coefficients(sub["pos"], sub["x"], sub["y"], s, sub["wall_nb"]),
        "perimeter": float(np.linalg.norm(np.roll(loop, -1, 0) - loop, axis=1).sum()),
        **{f"g{i}": float(v) for i, v in enumerate(geometry_descriptor(pos_s, n_s))},
    }


def save_scalars(rows):
    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "scalars.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["U", "aoa_deg", "cd", "cd_p", "cd_v", "cl", "perimeter"]].describe())
    # A NACA airfoil of chord 1 has a perimeter of ~2.0-2.2; anything else means a broken surface loop.
    odd = df[(df.perimeter < 1.95) | (df.perimeter > 2.3)]
    if len(odd):
        print(f"WARNING: {len(odd)} sims with suspicious surface perimeter:")
        print(odd[["name", "perimeter", "cd", "cl"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=32000, help="nodes kept per simulation")
    ap.add_argument("--delete-raw", action="store_true", help="delete raw AirfRANS.pt afterwards")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N sims (debug)")
    ap.add_argument("--download-only", action="store_true",
                    help="only download + extract (run on a login node with internet access)")
    ap.add_argument("--rebuild-scalars", action="store_true",
                    help="recompute scalars.csv from already processed sims (no raw data needed)")
    args = ap.parse_args()

    if args.rebuild_scalars:
        names = sorted(f.stem for f in (PROC_DIR / "sims").glob("*.pt"))
        save_scalars([scalar_row(n, torch.load(PROC_DIR / "sims" / f"{n}.pt", weights_only=False))
                      for n in tqdm(names, desc="rebuild")])
        return

    pt, manifest_path = fetch_raw()
    if args.download_only:
        print("raw data ready in", RAW_DIR)
        return
    manifest = json.loads(manifest_path.read_text())
    names = manifest["full_train"] + manifest["full_test"]

    print("memory-mapping raw file ...")
    raw = torch.load(pt, mmap=True, weights_only=False)
    assert len(raw) == len(names), (len(raw), len(names))

    (PROC_DIR / "sims").mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_path, PROC_DIR / "manifest.json")

    rng = np.random.default_rng(0)
    rows = []
    for k, name in enumerate(tqdm(names[: args.limit], desc="process")):
        out = PROC_DIR / "sims" / f"{name}.pt"
        sub = process_one(raw[k], args.n_nodes, rng)
        torch.save(sub, out)
        rows.append(scalar_row(name, sub))
    save_scalars(rows)

    if args.delete_raw:
        del raw
        pt.unlink()
        print("deleted", pt)


if __name__ == "__main__":
    main()
