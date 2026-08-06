#!/bin/bash
#SBATCH -J dyn_abilene_h6_learned_failure
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/dyn_abilene_h6_learned_failure_%j.out
#SBATCH -e logs/dyn_abilene_h6_learned_failure_%j.err

set -euo pipefail

cd /oscar/home/rkapoor8/CS2680/HARP
mkdir -p logs
source harp_env/bin/activate

echo "Using python:"
which python
python --version

echo "Working dir:"
pwd
echo "Date:"
date

TOPOLOGY_JSON="topologies/abilene/t1.json"
PAIRS_PKL="pairs/abilene/t1.pkl"
TRAFFIC_DIR="traffic_matrices/abilene"
GENERATOR="generate_dynamic_failures.py"

OUT_DIR="dynamic_abilene_h6_realistic_learned_failure"
TM_SERIES="$OUT_DIR/abilene_tm_series.pkl"

HISTORY_LEN=6
K_PATHS=4
NUM_SAMPLES=1000
SEED=0

if [[ ! -f "$GENERATOR" ]]; then
  echo "ERROR: $GENERATOR not found in $(pwd)."
  exit 1
fi

if [[ ! -f "$TOPOLOGY_JSON" ]]; then
  echo "ERROR: topology file not found: $TOPOLOGY_JSON"
  exit 1
fi

if [[ ! -f "$PAIRS_PKL" ]]; then
  echo "ERROR: pairs file not found: $PAIRS_PKL"
  exit 1
fi

if [[ ! -d "$TRAFFIC_DIR" ]]; then
  echo "ERROR: traffic dir not found: $TRAFFIC_DIR"
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "Combining Abilene traffic matrices into one time-series file..."

TRAFFIC_DIR_ENV="$TRAFFIC_DIR" OUT_DIR_ENV="$OUT_DIR" python - <<'PY'
import os
import re
import pickle
import numpy as np
from pathlib import Path

traffic_dir = Path(os.environ["TRAFFIC_DIR_ENV"])
out_dir = Path(os.environ["OUT_DIR_ENV"])
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
        raise ValueError(f"{p} has unexpected shape {x.shape}; expected one flat traffic vector")
    series.append(x)

series = np.stack(series, axis=0)

with open(out_path, "wb") as f:
    pickle.dump(series, f)

print(f"Saved combined traffic series to {out_path}")
print(f"Combined shape: {series.shape}")
print(f"First file: {files[0]}")
print(f"Last file: {files[-1]}")
PY

echo "Generating dynamic Abilene h6 dataset with learned-failure labels..."
echo "IMPORTANT: no candidate states/probabilities should be saved in the dataset."

python "$GENERATOR" \
  --topology_json "$TOPOLOGY_JSON" \
  --pairs_pkl "$PAIRS_PKL" \
  --traffic_pkl "$TM_SERIES" \
  --out_dir "$OUT_DIR" \
  --history_len "$HISTORY_LEN" \
  --k_paths "$K_PATHS" \
  --num_samples "$NUM_SAMPLES" \
  --failure_model realistic \
  --failure_probability 0.020 \
  --partial_capacity_levels "0.75,0.5,0.25" \
  --partial_event_weight 0.40 \
  --full_event_weight 0.22 \
  --correlated_event_weight 0.13 \
  --maintenance_event_weight 0.10 \
  --flaky_burst_event_weight 0.15 \
  --max_failed_edges 1 \
  --max_edges_per_event 2 \
  --max_correlated_edges 2 \
  --correlated_full_probability 0.25 \
  --maintenance_full_probability 0.35 \
  --maintenance_node_event_probability 0.30 \
  --max_maintenance_edges 2 \
  --short_duration_min 1 \
  --short_duration_max 2 \
  --medium_duration_min 3 \
  --medium_duration_max 12 \
  --long_duration_min 12 \
  --long_duration_max 48 \
  --short_duration_weight 0.50 \
  --medium_duration_weight 0.35 \
  --long_duration_weight 0.15 \
  --maintenance_duration_min 12 \
  --maintenance_duration_max 72 \
  --flaky_edge_fraction 0.15 \
  --flaky_edge_multiplier 5.0 \
  --flaky_full_probability 0.35 \
  --edge_risk_sigma 1.0 \
  --max_resample_attempts 50 \
  --num_failure_candidates 0 \
  --seed "$SEED"

echo "Quick audit of generated Abilene samples..."

OUT_DIR_ENV="$OUT_DIR" python - <<'PY'
import os
import torch
from pathlib import Path
from collections import Counter

out_dir = Path(os.environ["OUT_DIR_ENV"])
files = sorted(out_dir.glob("sample_*.pt"))

print("out_dir:", out_dir)
print("num samples:", len(files))

if not files:
    raise RuntimeError("No sample_*.pt files found")

required = {
    "node_features",
    "edge_index",
    "capacities",
    "padded_edge_ids_per_path",
    "paths_to_edges",
    "tm",
    "tm_pred",
    "future_capacities",
    "future_capacity_multipliers",
    "future_edge_state_labels",
    "metadata",
}

forbidden = {
    "failure_candidate_probs",
    "failure_candidate_capacities",
    "failure_candidate_capacity_multipliers",
    "failure_candidate_metadata",
}

s = torch.load(files[0], map_location="cpu")

print()
print("first sample:", files[0].name)
print("keys:", list(s.keys()))

missing = required - set(s.keys())
if missing:
    raise RuntimeError(f"First sample missing required keys: {sorted(missing)}")

leaked = forbidden & set(s.keys())
if leaked:
    raise RuntimeError(f"Forbidden candidate keys found in first sample: {sorted(leaked)}")

print("tm shape:", tuple(s["tm"].shape))
print("tm_pred == tm:", torch.allclose(s["tm"], s["tm_pred"]))

for key in ["future_capacities", "future_capacity_multipliers", "future_edge_state_labels"]:
    print(key, tuple(s[key].shape))

T = s["node_features"].shape[1]
P = s["tm"].shape[2]
E_final = s["capacities"][-1].shape[-1]

assert len(s["edge_index"]) == T, "edge_index length != T"
assert len(s["capacities"]) == T, "capacities length != T"
assert len(s["padded_edge_ids_per_path"]) == T, "padded paths length != T"
assert len(s["paths_to_edges"]) == T, "paths_to_edges length != T"
assert s["paths_to_edges"][-1].shape == (P, E_final), "paths_to_edges final shape mismatch"
assert s["future_capacities"].shape == (E_final,), "future_capacities shape mismatch"
assert s["future_capacity_multipliers"].shape == (E_final,), "future_capacity_multipliers shape mismatch"
assert s["future_edge_state_labels"].shape == (E_final,), "future_edge_state_labels shape mismatch"

print("T:", T, "P:", P, "E_final:", E_final)
print("future multipliers seen in first sample:", sorted(set(round(float(x), 4) for x in s["future_capacity_multipliers"].flatten())))
print("future labels seen in first sample:", sorted(set(int(x) for x in s["future_edge_state_labels"].flatten())))

meta = s.get("metadata", {})
print("metadata keys:", list(meta.keys()))
print("future_timestep:", meta.get("future_timestep"))
print("future label assumption:", meta.get("future_label_assumption"))
print("failure model note:", meta.get("failure_model_note"))
print("evaluation note:", meta.get("evaluation_note"))

print()
print("Checking sampled files for required keys, shape alignment, and forbidden candidate keys...")

indices = sorted(set([0, 1, 2, len(files)//2, len(files)-3, len(files)-2, len(files)-1]))
label_counts = Counter()

for i in indices:
    p = files[i]
    x = torch.load(p, map_location="cpu")

    missing = required - set(x.keys())
    if missing:
        raise RuntimeError(f"{p.name} missing required keys: {sorted(missing)}")

    leaked = forbidden & set(x.keys())
    if leaked:
        raise RuntimeError(f"{p.name} has forbidden candidate keys: {sorted(leaked)}")

    T_i = x["node_features"].shape[1]
    P_i = x["tm"].shape[2]
    E_i = x["capacities"][-1].shape[-1]

    assert len(x["edge_index"]) == T_i, f"{p.name}: edge_index length mismatch"
    assert len(x["capacities"]) == T_i, f"{p.name}: capacities length mismatch"
    assert len(x["padded_edge_ids_per_path"]) == T_i, f"{p.name}: padded paths length mismatch"
    assert len(x["paths_to_edges"]) == T_i, f"{p.name}: paths_to_edges length mismatch"
    assert x["paths_to_edges"][-1].shape == (P_i, E_i), f"{p.name}: paths_to_edges final shape mismatch"
    assert x["future_capacities"].shape == (E_i,), f"{p.name}: future_capacities mismatch"
    assert x["future_capacity_multipliers"].shape == (E_i,), f"{p.name}: future_capacity_multipliers mismatch"
    assert x["future_edge_state_labels"].shape == (E_i,), f"{p.name}: future_edge_state_labels mismatch"

    label_counts.update(int(v) for v in x["future_edge_state_labels"].flatten().tolist())
    print(f"  {p.name}: clean | T={T_i} P={P_i} E_final={E_i} labels={sorted(set(int(v) for v in x['future_edge_state_labels'].flatten()))}")

print()
print("Sampled future label counts:")
for cls in range(5):
    print(f"  class {cls}: {label_counts.get(cls, 0)}")

print()
print("PASS: Abilene dataset format is clean.")
print("PASS: no failure_candidate_* keys found in checked samples.")
print("PASS: realized future labels are present and aligned.")
PY

echo "Done generating $OUT_DIR"
