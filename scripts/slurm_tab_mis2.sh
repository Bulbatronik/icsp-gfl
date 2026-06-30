#!/bin/bash
#SBATCH --job-name=tab_mis2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/tab_mis2_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
export RESULTS_DIR=results_tableI
# Table I FMNIST alpha=0.3, miserable (77), spectral + heat methods (k=12, t=1), 3 seeds.
base(){ python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid network=miserable "$@"; }
for S in 42 43 44; do
  base seed=$S client=spect_eig3_cos client.num_eig=12
  base seed=$S client=heatkern_t40 client.num_eig=12 client.t=1
done
echo DONE
