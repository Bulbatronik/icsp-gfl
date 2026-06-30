#!/bin/bash
#SBATCH --job-name=dfl_lg
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=scripts/logs/lg_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=60 dataset=fmnist_niid01 network=sparse100 "$@"; }
run seed=42 client=nofed
run seed=42 client=broadcast
for S in 42 43 44; do
  run seed=$S client=random
  run seed=$S client=spect_eig3_cos client.num_eig=12
done
echo DONE
