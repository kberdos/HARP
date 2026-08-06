#!/bin/bash
#SBATCH -J dyn_kdl_full_k4_shared_full
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o logs/dyn_kdl_full_k4_shared_full_%j.out
#SBATCH -e logs/dyn_kdl_full_k4_shared_full_%j.err

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

echo "Host:"
hostname

echo "Quota before run:"
checkquota || true

SCRATCH_ROOT="/oscar/scratch/rkapoor8/HARP"
mkdir -p "$SCRATCH_ROOT"

TOPOLOGY_JSON="topologies/kdl/t1.json"
FULL_PAIRS_PKL="pairs/kdl/t1.pkl"
TRAFFIC_DIR="traffic_matrices/kdl"
GENERATOR="generate_dynamic_failures.py"

SMOKE_DIR="$SCRATCH_ROOT/dynamic_kdl_full_h6_k4_realistic_learned_failure_shared_smoke"
OUT_DIR="$SCRATCH_ROOT/dynamic_kdl_full_h6_k4_realistic_learned_failure_shared_full"

SMOKE_CACHE="$SMOKE_DIR/static_cache.pt"
SMOKE_TM_SERIES="$SMOKE_DIR/kdl_full_tm_series.pkl"

STATIC_CACHE="$OUT_DIR/static_cache.pt"
TM_SERIES="$OUT_DIR/kdl_full_tm_series.pkl"

HISTORY_LEN=6
K_PATHS=4

# KDL has 278 traffic snapshots.
# With h=6 and true t+1 labels, we will generate 271 samples to match prior split logic.
NUM_SAMPLES=271

SEED=0

echo "Checking required files..."

if [[ ! -f "$GENERATOR" ]]; then
  echo "ERROR: $GENERATOR not found in $(pwd)."
  exit 1
fi

if [[ ! -f "$TOPOLOGY_JSON" ]]; then
  echo "ERROR: topology file not found: $TOPOLOGY_JSON"
  exit 1
fi

if [[ ! -f "$FULL_PAIRS_PKL" ]]; then
  echo "ERROR: full pairs file not found: $FULL_PAIRS_PKL"
  exit 1
fi

if [[ ! -d "$TRAFFIC_DIR" ]]; then
  echo "ERROR: traffic dir not found: $TRAFFIC_DIR"
  exit 1
fi

if [[ ! -f "$SMOKE_CACHE" ]]; then
  echo "ERROR: smoke static cache not found:"
  echo "$SMOKE_CACHE"
  echo "Run the one-sample shared-cache smoke script first."
  exit 1
fi

if [[ ! -f "$SMOKE_TM_SERIES" ]]; then
  echo "ERROR: smoke traffic series not found:"
  echo "$SMOKE_TM_SERIES"
  echo "Run the one-sample shared-cache smoke script first."
  exit 1
fi

if [[ "$OUT_DIR" != *"dynamic_kdl_full"* || "$OUT_DIR" != *"shared_full"* ]]; then
  echo "ERROR: OUT_DIR does not look like the intended full-KDL shared full directory:"
  echo "$OUT_DIR"
  exit 1
fi

echo "Smoke directory:"
echo "$SMOKE_DIR"

echo "Full output directory:"
echo "$OUT_DIR"

echo "------------------------------------------------------------"
echo "Step 0: Compile generator"
echo "------------------------------------------------------------"
python -m py_compile "$GENERATOR"

echo "------------------------------------------------------------"
echo "Step 1: Prepare full output directory and reuse shared cache"
echo "------------------------------------------------------------"

mkdir -p "$OUT_DIR"

# Do NOT delete static_cache.pt. Only remove old sample files/summaries if this full dir existed before.
rm -f "$OUT_DIR"/sample_*.pt
rm -f "$OUT_DIR"/failure_timeline_summary.json

# Reuse the expensive static cache from the successful smoke run.
# Hard-link first to avoid duplicate storage; fall back to copy if hard-link is unavailable.
if [[ ! -f "$STATIC_CACHE" ]]; then
  ln "$SMOKE_CACHE" "$STATIC_CACHE" 2>/dev/null || cp "$SMOKE_CACHE" "$STATIC_CACHE"
fi

# Reuse the full traffic series too.
if [[ ! -f "$TM_SERIES" ]]; then
  ln "$SMOKE_TM_SERIES" "$TM_SERIES" 2>/dev/null || cp "$SMOKE_TM_SERIES" "$TM_SERIES"
fi

echo "Static cache:"
ls -lh "$STATIC_CACHE"

echo "Traffic series:"
ls -lh "$TM_SERIES"

echo "------------------------------------------------------------"
echo "Step 2: Generate full KDL K=4 shared-cache dataset"
echo "------------------------------------------------------------"

python "$GENERATOR" \
  --topology_json "$TOPOLOGY_JSON" \
  --pairs_pkl "$FULL_PAIRS_PKL" \
  --traffic_pkl "$TM_SERIES" \
  --out_dir "$OUT_DIR" \
  --history_len "$HISTORY_LEN" \
  --k_paths "$K_PATHS" \
  --num_samples "$NUM_SAMPLES" \
  --shared_static_cache 1 \
  --static_cache_name "static_cache.pt" \
  --reuse_static_cache 1 \
  --current_failure_capacity_floor 1e-4 \
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
  --seed "$SEED"

echo "------------------------------------------------------------"
echo "Step 3: Quick audit of full-KDL shared-cache full dataset"
echo "------------------------------------------------------------"

OUT_DIR_ENV="$OUT_DIR" NUM_SAMPLES_ENV="$NUM_SAMPLES" python - <<'PY'
import os
import torch
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_ENV"])
expected_samples = int(os.environ["NUM_SAMPLES_ENV"])

files = sorted(out_dir.glob("sample_*.pt"))
cache_path = out_dir / "static_cache.pt"
tm_path = out_dir / "kdl_full_tm_series.pkl"

print("out_dir:", out_dir)
print("num sample files:", len(files))
print("expected samples:", expected_samples)
print("static cache exists:", cache_path.exists())
print("traffic series exists:", tm_path.exists())

if not cache_path.exists():
    raise RuntimeError("Missing static_cache.pt")

if not tm_path.exists():
    raise RuntimeError("Missing kdl_full_tm_series.pkl")

if len(files) != expected_samples:
    raise RuntimeError(f"Expected {expected_samples} samples, got {len(files)}")

s = torch.load(files[0], map_location="cpu")

print("first sample:", files[0].name)
print("sample keys:", list(s.keys()))
print("node_features:", tuple(s["node_features"].shape))
print("capacities:", tuple(s["capacities"].shape))
print("tm:", tuple(s["tm"].shape))
print("tm_pred:", tuple(s["tm_pred"].shape))
print("tm_pred == tm:", torch.allclose(s["tm"], s["tm_pred"]))
print("static_cache_path:", s.get("static_cache_path"))

for forbidden in ["edge_index", "paths_to_edges", "padded_edge_ids_per_path"]:
    if forbidden in s:
        raise RuntimeError(f"Shared-cache sample should not contain {forbidden}")

required = ["future_capacities", "future_capacity_multipliers", "future_edge_state_labels"]
for key in required:
    if key not in s:
        raise RuntimeError(f"Missing required learned-failure field: {key}")
    print(key, tuple(s[key].shape))

cache = torch.load(cache_path, map_location="cpu")

print("cache keys:", list(cache.keys()))
print("cache edge_index:", tuple(cache["edge_index"].shape))
print("cache base_capacities:", tuple(cache["base_capacities"].shape))
print("cache padded_edge_ids_per_path:", tuple(cache["padded_edge_ids_per_path"].shape))
print("cache paths_to_edges:", tuple(cache["paths_to_edges"].shape))
print("cache num_pairs:", cache["num_pairs"])
print("cache k_paths:", cache["k_paths"])
print("cache num_paths:", cache["num_paths"])
print("cache num_edges:", cache["num_edges"])
print("cache max_path_length:", cache["max_path_length"])

expected_paths = int(cache["num_pairs"]) * int(cache["k_paths"])

if int(cache["num_paths"]) != expected_paths:
    raise RuntimeError(f"num_paths mismatch: {cache['num_paths']} vs {expected_paths}")

if s["tm"].shape[2] != int(cache["num_paths"]):
    raise RuntimeError("sample tm path dimension does not match cache num_paths")

if s["capacities"].shape[-1] != int(cache["num_edges"]):
    raise RuntimeError("sample capacity edge dimension does not match cache num_edges")

if s["future_capacities"].shape[0] != int(cache["num_edges"]):
    raise RuntimeError("future capacity edge dimension does not match cache num_edges")

sample_size_mb = files[0].stat().st_size / 1024 / 1024
cache_size_mb = cache_path.stat().st_size / 1024 / 1024
dir_size_gb = sum(p.stat().st_size for p in out_dir.glob("*") if p.is_file()) / 1024 / 1024 / 1024

print("sample size MB:", sample_size_mb)
print("static cache size MB:", cache_size_mb)
print("dataset dir size GB:", dir_size_gb)
print("estimated samples-only GB:", sum(p.stat().st_size for p in files) / 1024 / 1024 / 1024)

print("first sample future multipliers seen:", sorted(set(round(float(x), 4) for x in s["future_capacity_multipliers"].flatten())))
print("first sample future labels seen:", sorted(set(int(x) for x in s["future_edge_state_labels"].flatten())))

# Check a few samples across the dataset without loading all 271.
probe_indices = sorted(set([0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]))
for idx in probe_indices:
    x = torch.load(files[idx], map_location="cpu")
    print("probe:", files[idx].name, "tm", tuple(x["tm"].shape), "future labels", sorted(set(int(y) for y in x["future_edge_state_labels"].flatten())))
    for key in required:
        if key not in x:
            raise RuntimeError(f"Missing {key} in {files[idx]}")
    if x["future_capacities"].shape[0] != int(cache["num_edges"]):
        raise RuntimeError(f"Future capacities not aligned in {files[idx]}")
    for forbidden in ["edge_index", "paths_to_edges", "padded_edge_ids_per_path"]:
        if forbidden in x:
            raise RuntimeError(f"{files[idx]} unexpectedly contains {forbidden}")

print("Quick audit passed.")
PY

echo "------------------------------------------------------------"
echo "Done generating full KDL K=4 shared-cache full dataset"
echo "Output written to scratch:"
echo "$OUT_DIR"
echo "Current scratch usage:"
du -sh /oscar/scratch/rkapoor8/HARP || true
echo "Quota after run:"
checkquota || true