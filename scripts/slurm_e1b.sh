#!/bin/bash
#SBATCH --job-name=dfl_e1b
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/e1b_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run () { echo "=== RUN: $* ==="; python main.py training=mnistcnn seed=42 federation.rounds=100 "$@"; }

# (1) complete the topology-correlated women row (missing heat-kernel)
run dataset=fmnist_topo  network=women client=heatkern_t40 client.num_eig=6 client.t=5
# (2) strong heterogeneity alpha=0.1 (Dirichlet) on the medium graph
run dataset=fmnist_niid01 network=women client=nofed
run dataset=fmnist_niid01 network=women client=broadcast
run dataset=fmnist_niid01 network=women client=random
run dataset=fmnist_niid01 network=women client=spect_eig3_cos client.num_eig=6
run dataset=fmnist_niid01 network=women client=heatkern_t40 client.num_eig=6 client.t=5
echo "=== E1b DONE ==="
