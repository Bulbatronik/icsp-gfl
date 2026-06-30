#!/bin/bash
#SBATCH --job-name=dfl_sens3
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:30:00
#SBATCH --output=scripts/logs/sens3_%j.out
# 3-seed sensitivity on the cheap small graph (Dirichlet alpha=0.3) for error bars.
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid network=small "$@"; }
for S in 42 43 44; do
  for B in 0.25 0.5 0.75; do
    run seed=$S client=spect_eig3_cos client.num_eig=3 client.selection_ratio=$B
    run seed=$S client=random client.selection_ratio=$B
  done
  for K in 2 3 6; do
    run seed=$S client=spect_eig3_cos client.num_eig=$K client.selection_ratio=0.5
  done
done
echo DONE
