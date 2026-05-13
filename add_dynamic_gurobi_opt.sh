#!/bin/bash
#SBATCH -J dyn_gurobi_h6_1000
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/dyn_gurobi_h6_1000_%j.out

set -euo pipefail

cd /users/rkapoor8/CS2680/HARP
mkdir -p logs

source harp_env/bin/activate

echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
python -c "import gurobipy as gp; print('gurobi ok', gp.gurobi.version())"

python add_dynamic_gurobi_opt.py \
  --samples_dir dynamic_abilene_h1_1000_samples \
  --k_paths 4 \
  --start_idx 0 \
  --end_idx 1000 \
  --overwrite 1 \
  --time_limit 30