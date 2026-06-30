#!/bin/bash
#SBATCH --job-name=dfl_hl100b
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/hl100b_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
# Job B (parallel to A): data-aware KLD + reference bounds, alpha=0.1, 100 rounds, 5 seeds.
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid01 network=miserable "$@"; }
for S in 42 43 44 45 46; do
  run seed=$S client=kld
  run seed=$S client=broadcast
  run seed=$S client=nofed
done
echo DONE
