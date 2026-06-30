#!/bin/bash
#SBATCH --job-name=dfl_hl100
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=scripts/logs/hl100_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
# Rigorous headline: large graph (Les Miserables), alpha=0.1, 100 rounds (matches Table I),
# 5 seeds for every method so spectral/heat/kld/random all carry error bars.
run(){ echo "=== $* ==="; python main.py training=mnistcnn federation.rounds=100 dataset=fmnist_niid01 network=miserable "$@"; }
# Job A: the three core comparison methods, 5 seeds each (~22 min/run).
for S in 42 43 44 45 46; do
  run seed=$S client=random
  run seed=$S client=spect_eig3_cos client.num_eig=12
  run seed=$S client=heatkern_t40 client.num_eig=12 client.t=1
done
echo DONE
