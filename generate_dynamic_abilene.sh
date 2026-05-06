#!/bin/bash
#SBATCH -J dyn_abilene_h6_1000
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/dyn_abilene_h6_1000_%j.out

set -euo pipefail

cd /oscar/home/rkapoor8/CS2680/HARP

mkdir -p logs
mkdir -p dynamic_abilene_h6_1000_samples

source harp_env/bin/activate

echo "Using python:"
which python
python --version

echo "Working dir:"
pwd

echo "Combining Abilene traffic matrices into one time-series file..."

python - <<'PY'
import re
import pickle
import numpy as np
from pathlib import Path

traffic_dir = Path("traffic_matrices/abilene")
out_dir = Path("dynamic_abilene_h6_1000_samples")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "abilene_tm_series.pkl"

files = list(traffic_dir.glob("t*.pkl"))
files = sorted(files, key=lambda p: int(re.findall(r"\d+", p.stem)[0]))

if len(files) == 0:
    raise ValueError(f"No traffic matrix files found in {traffic_dir}")

series = []

for p in files:
    with open(p, "rb") as f:
        x = pickle.load(f)

    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 1:
        raise ValueError(f"{p} has unexpected shape {x.shape}; expected one vector like (132,)")

    series.append(x)

series = np.stack(series, axis=0)

with open(out_path, "wb") as f:
    pickle.dump(series, f)

print(f"Saved combined traffic series to {out_path}")
print(f"Combined shape: {series.shape}")
print(f"First file: {files[0]}")
print(f"Last file: {files[-1]}")
PY

echo "Generating dynamic Abilene h6 1000-sample dataset..."

python generate_dynamic_abilene.py \
  --topology_json topologies/abilene/t1.json \
  --pairs_pkl pairs/abilene/t1.pkl \
  --traffic_pkl dynamic_abilene_h6_1000_samples/abilene_tm_series.pkl \
  --out_dir dynamic_abilene_h6_1000_samples \
  --history_len 6 \
  --k_paths 4 \
  --num_samples 1000 \
  --failure_probability 0.05 \
  --max_failed_edges 1 \
  --min_failure_duration 2 \
  --max_failure_duration 4 \
  --max_resample_attempts 50 \
  --seed 0

echo "Done generating dynamic_abilene_h6_1000_samples"