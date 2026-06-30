#!/bin/bash
#SBATCH --job-name=dfl_kld
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/logs/kld_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=60 dataset=fmnist_niid01 network=miserable "$@"; }
# Data-aware KLD selection on the large graph at alpha=0.1 (matches Fig. 3a regime),
# four seeds for a direct comparison against spectral/heat/random.
for S in 42 43 44 45; do
  run seed=$S client=kld
done
echo DONE
