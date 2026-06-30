#!/bin/bash
#SBATCH --job-name=dfl_heat
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/heat_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=60 dataset=fmnist_niid01 network=miserable "$@"; }
# heat-kernel t-sweep on the large graph (calibrate informative t)
for T in 0.5 1 2 5; do
  run seed=42 client=heatkern_t40 client.num_eig=12 client.t=$T
done
# extra seeds for the headline (spectral vs random) on the large graph
for S in 44 45; do
  run seed=$S client=random
  run seed=$S client=spect_eig3_cos client.num_eig=12
done
echo DONE
