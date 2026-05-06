#!/bin/bash
#SBATCH -J harp_test_abilene
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/harp_test_abilene_%j.out

set -euo pipefail

cd /users/rkapoor8/CS2680/HARP
mkdir -p logs

source harp_env/bin/activate

echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

python3 run_harp.py \
  --topo abilene \
  --mode test \
  --num_paths_per_pair 8 \
  --num_for_loops 3 \
  --test_cluster 0 \
  --test_start_idx 14112 \
  --test_end_idx 16128 \
  --framework harp \
  --pred 0 \
  --dynamic 0