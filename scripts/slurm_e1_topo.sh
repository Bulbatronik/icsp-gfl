#!/bin/bash
#SBATCH --job-name=dfl_e1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=scripts/logs/e1_%j.out

# E1: topology-correlated heterogeneity (topo partition) + strong Dirichlet (alpha=0.1)
# Answers R2-2 / R1-2 / AE: does topology reflect learning relevance under heterogeneity?
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
which python

run () {  # args: dataset network num_eig t client extra...
  echo "=== RUN: $* ==="
  python main.py training=mnistcnn seed=42 federation.rounds=100 "$@"
}

# --- Topology-correlated partition (topo) ---
for NET in "small 3 5" "women 6 5"; do
  set -- $NET; NETNAME=$1; KEIG=$2; TT=$3
  run dataset=fmnist_topo network=$NETNAME client=nofed
  run dataset=fmnist_topo network=$NETNAME client=broadcast
  run dataset=fmnist_topo network=$NETNAME client=random
  run dataset=fmnist_topo network=$NETNAME client=gradients
  run dataset=fmnist_topo network=$NETNAME client=spect_eig3_cos client.num_eig=$KEIG
  run dataset=fmnist_topo network=$NETNAME client=heatkern_t40 client.num_eig=$KEIG client.t=$TT
done

# --- Strong Dirichlet heterogeneity (alpha=0.1) on the medium graph ---
run dataset=fmnist_niid01 network=women client=nofed
run dataset=fmnist_niid01 network=women client=broadcast
run dataset=fmnist_niid01 network=women client=random
run dataset=fmnist_niid01 network=women client=spect_eig3_cos client.num_eig=6
run dataset=fmnist_niid01 network=women client=heatkern_t40 client.num_eig=6 client.t=5

echo "=== E1 DONE ==="
