# AirfRANS: GNN field prediction vs. MLP drag baseline

Surrogate models for 2D RANS simulations of airfoils from the
[AirfRANS](https://arxiv.org/abs/2212.07564) dataset (NeurIPS 2022 Datasets & Benchmarks, 1000 simulations, ~180k mesh nodes each).

* **GNN** (`train_gnn.py`): message-passing network on k-NN graphs of the CFD mesh, predicting
  velocity `(u_x, u_y)`, pressure `p/rho` and turbulent viscosity `nut` at every node.
  Drag and lift are then obtained by integrating the *predicted* surface pressure and wall shear.
* **MLP** (`train_mlp.py`): `(U_inf, angle of attack, airfoil shape descriptor) -> (Cd, Cl)` directly.

Both are evaluated on the four AirfRANS tasks:

| Task | Train | Test | Tests |
|---|---|---|---|
| `full` | 800 sims | 200 sims | accuracy with all data |
| `scarce` | 200 sims | same 200 as `full` | low-data regime |
| `reynolds` | mid-range Reynolds numbers | Reynolds outside that range | extrapolation in Reynolds |
| `aoa` | mid-range angles of attack | angles outside that range | extrapolation in AoA |

10% of each training set is held out for validation / model selection.

**Results:** see [`results/report/results.md`](results/report/results.md).

## Method notes

* **Reference Cd/Cl.** The PyG release of AirfRANS contains no force coefficients, so they are
  computed from the full-resolution CFD fields: pressure integrated over the airfoil plus wall
  shear `nu * u_t / d` from the first off-wall node (the mesh is wall-resolved, y+ ~ 1).
  Surface points are ordered into a closed loop by splitting upper/lower sides with the sign of
  the surface normal, which is robust for thin, highly cambered airfoils.
* **Graphs.** Each mesh is stored with 32k nodes (all surface nodes and their first off-wall
  neighbours always kept, so forces can be recomputed exactly). During training a random
  16k-node subset is drawn each time and connected as a k-NN graph (k = 12) with scale-free
  edge features (unit direction, log distance).
* **Targets** are non-dimensionalised by the inlet speed (`u/U`, `Cp`, `nut/nu`) and standardised.

## Files

| File | Role |
|---|---|
| `common.py` | paths, force-coefficient integration, geometry descriptor, graph dataset |
| `prepare_data.py` | download AirfRANS (4.5 GB) and convert to compact per-simulation files |
| `train_mlp.py`, `train_gnn.py` | models; test predictions go to `results/` |
| `compare.py` | MLP vs GNN for one task |
| `analyze.py` | GNN drag error split into pressure / viscous parts for one task |
| `report.py` | all tasks: tables + figures in `results/report/` |
| `hpc/` | PBS job scripts for the Cranfield Crescent2 cluster |

Data location is set by the `AIRFRANS_DATA` environment variable (default `~/data/airfrans`).

## Reproducing

### On a PBS cluster (Crescent2)

```bash
bash hpc/setup_env.sh                      # login node: conda env + data download
qsub hpc/prepare.pbs                       # preprocessing (CPU)
qsub hpc/gpu_check.pbs                     # optional: check the GPU works
for t in full scarce reynolds aoa; do qsub -v TASK=$t hpc/train.pbs; done
python report.py                           # after all jobs have finished
```

PyTorch is pinned to 2.5.1 + CUDA 12.1 because the GPU nodes run NVIDIA driver 530.

### Locally (CPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python prepare_data.py --delete-raw   # ~13 GB disk at peak, ~1.5 GB after
python train_mlp.py --task full
python train_gnn.py --task full --epochs 40
python report.py
```

## Data and citation

The AirfRANS dataset is released under the [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/)
license and is **not** included in this repository. If you use it, cite:

> F. Bonnet, J. A. Mazari, P. Cinnella, P. Gallinari. *AirfRANS: High Fidelity Computational Fluid
> Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions.* NeurIPS 2022
> Datasets and Benchmarks Track.
