#!/bin/bash
#SBATCH --job-name=kld_tabC
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/kld_tabC_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
export RESULTS_DIR=results_tableI
# Data-similarity (KLD) baseline for Table I: CIFAR10, alpha=0.3, 100 rounds, 1 seed, all graphs.
base(){ python main.py training=cifar10cnn federation.rounds=100 dataset=cifar10_niid "$@"; }
for S in 42; do
  base network=small     seed=$S client=kld
  base network=women     seed=$S client=kld
  base network=miserable seed=$S client=kld
done
echo DONE
