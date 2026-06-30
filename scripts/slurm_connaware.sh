#!/bin/bash
#SBATCH --job-name=dfl_conn
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/conn_%j.out
# Decisive test: does connectivity-aware (effective-resistance) selection beat
# random and the original similarity method on the LARGE graph (miserable, 77
# nodes) under strong heterogeneity, where the mixing-rate gap is real?
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=60 dataset=fmnist_niid01 network=miserable "$@"; }
for S in 42 43; do
  run seed=$S client=random
  run seed=$S client=connaware
  run seed=$S client=spect_eig3_cos client.num_eig=12
done
run seed=42 client=broadcast
run seed=42 client=nofed
echo DONE
