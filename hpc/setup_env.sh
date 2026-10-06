#!/bin/bash
# One-time setup on the Crescent2 LOGIN node:  bash hpc/setup_env.sh
# Installs Miniforge (a personal conda) in your beegfs home, creates the "airfrans" env,
# then downloads the AirfRANS data (login nodes have internet; compute nodes may not).
set -euo pipefail

BASE=/mnt/beegfs/home/$USER/cnn
CONDA_DIR=$BASE/miniforge3
export AIRFRANS_DATA=$BASE/data
cd "$(dirname "$0")/.."

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
    rm /tmp/miniforge.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^airfrans "; then
    conda create -y -n airfrans python=3.11
fi
conda activate airfrans
# PyTorch built for CUDA 12.1: Crescent2 GPU nodes run NVIDIA driver 530 (max CUDA 12.1);
# newer CUDA builds fail there with "CUDA-capable device(s) is/are busy or unavailable".
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "torch_geometric==2.6.1"
pip install numpy scipy pandas matplotlib tqdm

python -c "import torch, torch_geometric; print('torch', torch.__version__, '| PyG', torch_geometric.__version__)"

# Download + extract the raw data (4.5 GB zip -> ~8 GB AirfRANS.pt) into $AIRFRANS_DATA/raw
python prepare_data.py --download-only
echo "Setup done. Next:  qsub hpc/prepare.pbs"
