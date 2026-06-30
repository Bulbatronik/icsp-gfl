#!/bin/bash
#SBATCH --job-name=dfl_blkchk
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=scripts/logs/blkchk_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
run(){ echo "=== $* ==="; python main.py training=mnistcnn seed=42 federation.rounds=40 dataset=fmnist_block network=women "$@"; }
run client=nofed
run client=broadcast
run client=random
run client=spect_eig3_cos client.num_eig=6
run client=heatkern_t40 client.num_eig=6 client.t=2
echo DONE
