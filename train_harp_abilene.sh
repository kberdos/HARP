#!/bin/bash
#SBATCH -J harp_abilene
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/harp_abilene_%j.out

set -euo pipefail

cd /users/rkapoor8/CS2680/HARP
mkdir -p logs

source harp_env/bin/activate

echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

python3 run_harp.py \
  --topo dynamic_abilene \
  --mode train \
  --epochs 2 \
  --lr 0.00005 \
  --batch_size 1 \
  --num_paths_per_pair 4 \
  --num_transformer_layers 2 \
  --num_gnn_layers 3 \
  --num_mlp1_hidden_layers 1 \
  --num_mlp2_hidden_layers 1 \
  --num_for_loops 3 \
  --framework harp \
  --pred 0 \
  --dynamic 1 \
  --dynamic_samples_dir dynamic_abilene_samples \
  --dynamic_train_start_idx 0 \
  --dynamic_train_end_idx 80 \
  --dynamic_val_start_idx 80 \
  --dynamic_val_end_idx 100 \
  --use_dynamic_opt 1