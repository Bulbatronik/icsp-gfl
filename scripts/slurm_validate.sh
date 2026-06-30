#!/bin/bash
#SBATCH --job-name=dfl_val
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=scripts/logs/val_%j.out
set -e
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ./.venv
export WANDB_MODE=offline
python -c "import torch;print('CUDA avail:',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
python main.py training=mnistcnn seed=42 federation.rounds=3 dataset=fmnist_topo network=small client=heatkern_t40 client.num_eig=3 client.t=5
python main.py training=mnistcnn seed=42 federation.rounds=3 dataset=fmnist_topo network=women client=spect_eig3_cos client.num_eig=6
echo "=== VALIDATE DONE ==="
