#!/bin/bash
#SBATCH --job-name=dfl_e2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --output=scripts/logs/e2_%j.out

# E2: sensitivity / ablation over beta (sampling ratio), k (embedding dim), t (diffusion time).
# Run on the cheap small (8-node) graph under the Dirichlet (alpha=0.3) regime of Table I,
# where spectral selection is known to help. Answers R1: ablation + sensitivity.
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run () { echo "=== RUN: $* ==="; python main.py training=mnistcnn seed=42 federation.rounds=100 \
         dataset=fmnist_niid network=small "$@"; }

# beta sweep (spectral, k=3) + random reference at each beta
for B in 0.25 0.5 0.75; do
  run client=spect_eig3_cos client.num_eig=3 client.selection_ratio=$B
  run client=random          client.selection_ratio=$B
done

# k (embedding dimension) sweep, spectral, beta=0.5  (k=7 -> near-full spectrum on 8 nodes)
for K in 2 3 5 7; do
  run client=spect_eig3_cos client.num_eig=$K client.selection_ratio=0.5
done

# t (diffusion time) sweep, heat kernel, k=3, beta=0.5  (probes the informative-t window)
for T in 0.1 0.5 1 5 20; do
  run client=heatkern_t40 client.num_eig=3 client.t=$T client.selection_ratio=0.5
done

echo "=== E2 DONE ==="
