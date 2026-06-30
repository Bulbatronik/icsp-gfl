#!/bin/bash
#SBATCH --job-name=heat005
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=scripts/logs/heat005_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
# Re-run only the heat-kernel curve for the alpha=0.1 headline at t=0.05 (the Table I large-graph
# value) instead of t=1, so the heat-kernel diffusion time is consistent across all experiments.
# Large graph (miserable), 100 rounds, 4 seeds, k=12.
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid01 network=miserable client=heatkern_t40 client.num_eig=12 client.t=0.05 "$@"; }
for S in 42 43 44 45; do
  run seed=$S
done
echo DONE
