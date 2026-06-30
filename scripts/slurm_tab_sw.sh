#!/bin/bash
#SBATCH --job-name=tab_sw
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/tab_sw_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
export RESULTS_DIR=results_tableI   # isolated from existing results/
# Table I reproduction (FMNIST, Dirichlet alpha=0.3, 100 rounds), small + women graphs.
# Hyperparams: spectral/heat num_eig = 3 (small) / 6 (women); heat t=1.
base(){ python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid "$@"; }
for S in 42 43 44; do
  for cfg in "small 3" "women 6"; do
    set -- $cfg; NET=$1; K=$2
    base network=$NET seed=$S client=broadcast
    base network=$NET seed=$S client=nofed
    base network=$NET seed=$S client=random
    base network=$NET seed=$S client=gradients
    base network=$NET seed=$S client=spect_eig3_cos client.num_eig=$K
    base network=$NET seed=$S client=heatkern_t40 client.num_eig=$K client.t=1
  done
done
echo DONE
