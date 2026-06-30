#!/bin/bash
#SBATCH --job-name=dfl_blklo
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=scripts/logs/blklo_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=80 dataset=fmnist_block network=women "$@"; }
for S in 42 43; do
  run seed=$S client=random        client.selection_ratio=0.25
  run seed=$S client=spect_eig3_cos client.num_eig=6 client.selection_ratio=0.25
done
run seed=42 client=broadcast
run seed=42 client=nofed
echo DONE
