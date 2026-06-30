#!/bin/bash
#SBATCH --job-name=dfl_gsim
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=scripts/logs/gsim_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
# Gradient-similarity baseline on the large graph at alpha=0.1, 100 rounds, 4 seeds,
# to add to Fig. 3a alongside spectral/heat/random/data-sim.
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid01 network=miserable "$@"; }
for S in 42 43 44 45; do
  run seed=$S client=gradients
done
echo DONE
