#!/bin/bash
#SBATCH -J grate_kdlfull_modes
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o /oscar/home/rkapoor8/CS2680/HARP/logs/grate_kdlfull_modes_%j.out
#SBATCH -e /oscar/home/rkapoor8/CS2680/HARP/logs/grate_kdlfull_modes_%j.err

set -euo pipefail

cd /oscar/home/rkapoor8/CS2680/HARP
mkdir -p logs configs checkpoints /oscar/scratch/rkapoor8/HARP_logs
source harp_env/bin/activate

# Use less-fragmented CUDA allocations when possible.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
# -----------------------------------------------------------------------------
# Run mode:
#   smoke = one-sample backward/training sanity check. Run this first.
#   full  = full KDL training on 0:190, validation 190:230, test 230:271.
#
# Submit smoke:
#   sbatch z_KDL_scripts/train_dynamic_kdl.sh
#
# Submit full after smoke passes:
#   RUN_MODE=full sbatch z_KDL_scripts/train_dynamic_kdl.sh
# -----------------------------------------------------------------------------
RUN_MODE="${RUN_MODE:-smoke}"

TOPO="dynamic_kdl"
DATASET="/oscar/scratch/rkapoor8/HARP/dynamic_kdl_full_h6_k4_realistic_learned_failure_shared_full"

# Architecture / model size. Keep this close to the KDL-50k setup, but K=4 paths.
NUM_PATHS=4
NUM_HEADS=1
NUM_TRANSFORMER_LAYERS=1
NUM_GNN_LAYERS=2
NUM_MLP1_LAYERS=1
NUM_MLP2_LAYERS=1
NUM_FOR_LOOPS=0
PRED=0

BATCH_SIZE=1
LR=0.00005

# Chunking controls for full KDL. These are injected into HARP props by the
# inline Python monkey patch below, so run_harp.py does not need new CLI args.
PATH_CHUNK_THRESHOLD="${PATH_CHUNK_THRESHOLD:-0}"
PATH_CHUNK_SIZE="${PATH_CHUNK_SIZE:-5000}"
FAILURE_AGGREGATION_CHUNK_SIZE="${FAILURE_AGGREGATION_CHUNK_SIZE:-200000}"
PATH_CHUNK_CHECKPOINT="${PATH_CHUNK_CHECKPOINT:-1}"
export PATH_CHUNK_THRESHOLD PATH_CHUNK_SIZE FAILURE_AGGREGATION_CHUNK_SIZE PATH_CHUNK_CHECKPOINT

# Resilience / failure-head objective.
USE_DYNAMIC_OPT=0
SPLIT_LOSS_WEIGHT=0.0
USE_RESILIENCE_OBJECTIVE=1
RESILIENCE_LOSS_WEIGHT=0.5
FAILURE_PREDICTION_LOSS_WEIGHT=0.1
NUM_FAILURE_STATES=5
NUM_FAILURE_SCENARIOS=5
SCENARIO_PROBABILITY_LOSS_WEIGHT=1.0
SCENARIO_CAPACITY_MODE="hard"
DETACH_SCENARIO_PROBABILITIES=1
DETACH_SCENARIO_CAPACITIES=1
SOFT_BACKUP_PENALTY_WEIGHT=0.0
FUTURE_LOCAL_RESCALE=1
DISCONNECTED_PENALTY=100.0

if [[ "$RUN_MODE" == "smoke" ]]; then
  EPOCHS=1
  TRAIN_START=0
  TRAIN_END=1
  VAL_START=1
  VAL_END=2
  TEST_START=2
  TEST_END=3
  EXP_NAME="smoke_h6_kdlfull_grate_topology_modes_K5_lambda0p5_beta0p1_lr5e5_chunk${PATH_CHUNK_SIZE}"
elif [[ "$RUN_MODE" == "full" ]]; then
  EPOCHS="${EPOCHS:-5}"
  TRAIN_START=0
  TRAIN_END=190
  VAL_START=190
  VAL_END=230
  TEST_START=230
  TEST_END=271
  EXP_NAME="h6_kdlfull_grate_topology_modes_K5_lambda0p5_beta0p1_lr5e5_epochs${EPOCHS}_split190_230_271_chunk${PATH_CHUNK_SIZE}"
else
  echo "ERROR: RUN_MODE must be smoke or full. Got: $RUN_MODE"
  exit 1
fi

CHECKPOINT="checkpoints/${EXP_NAME}.pt"
CONFIG_PATH="configs/${EXP_NAME}.txt"
RAW_MODEL="HARP_dynamic_${TOPO}_pred_False_${NUM_PATHS}sp.pkl"

echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Date: $(date)"
echo "RUN_MODE: $RUN_MODE"
echo "Dataset: $DATASET"
echo "Experiment: $EXP_NAME"
echo "Checkpoint target: $CHECKPOINT"
echo "Path chunk threshold: $PATH_CHUNK_THRESHOLD"
echo "Path chunk size: $PATH_CHUNK_SIZE"
echo "Failure aggregation chunk size: $FAILURE_AGGREGATION_CHUNK_SIZE"
echo "Path chunk checkpoint: $PATH_CHUNK_CHECKPOINT"
echo "GPU:"
nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

if [[ ! -d "$DATASET" ]]; then
  echo "ERROR: dataset directory not found: $DATASET"
  exit 1
fi

for f in run_harp.py utils/training_utils.py frameworks/harp_system.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing required file: $f"
    exit 1
  fi
done

if grep -q "generate_heuristic_candidates" utils/training_utils.py; then
  echo "ERROR: utils/training_utils.py still calls generate_heuristic_candidates during training."
  echo "ERROR: Use the clean K-scenario/topology-mode training_utils.py."
  exit 1
fi

if ! grep -qi "scenario" utils/training_utils.py; then
  echo "WARNING: utils/training_utils.py does not obviously mention scenario."
  echo "WARNING: Make sure this is the K-scenario GRATE topology-mode training_utils.py."
fi

python -m py_compile run_harp.py utils/training_utils.py frameworks/harp_system.py

echo "------------------------------------------------------------"
echo "Sanity-checking full-KDL shared-cache dataset"
echo "------------------------------------------------------------"

DATASET_ENV="$DATASET" NUM_PATHS_ENV="$NUM_PATHS" python - <<'PY'
import os
import glob
import torch
from pathlib import Path

root = Path(os.environ["DATASET_ENV"])
num_paths_per_pair = int(os.environ["NUM_PATHS_ENV"])
files = sorted(glob.glob(str(root / "sample_*.pt")))
cache_path = root / "static_cache.pt"

print("Found sample files:", len(files))
print("Static cache exists:", cache_path.exists())

if len(files) != 271:
    print("WARNING: expected 271 samples, found", len(files))
if len(files) == 0:
    raise RuntimeError("No sample files found")
if not cache_path.exists():
    raise RuntimeError("Missing static_cache.pt")

s = torch.load(files[0], map_location="cpu")
cache = torch.load(cache_path, map_location="cpu")

print("First sample:", files[0])
print("Sample keys:", list(s.keys()))
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
for k in required:
    if k not in s:
        raise RuntimeError(f"Missing {k}")
    print(k, "shape:", tuple(s[k].shape))

print("cache edge_index:", tuple(cache["edge_index"].shape))
print("cache padded_edge_ids_per_path:", tuple(cache["padded_edge_ids_per_path"].shape))
print("cache paths_to_edges:", tuple(cache["paths_to_edges"].shape))
print("cache num_pairs:", cache["num_pairs"])
print("cache k_paths:", cache["k_paths"])
print("cache num_paths:", cache["num_paths"])
print("cache num_edges:", cache["num_edges"])
print("cache max_path_length:", cache["max_path_length"])

if int(cache["k_paths"]) != num_paths_per_pair:
    raise RuntimeError(f"Expected k_paths={num_paths_per_pair}, got {cache['k_paths']}")
if s["tm"].shape[2] != int(cache["num_paths"]):
    raise RuntimeError("sample tm path dimension does not match cache num_paths")
if s["capacities"].shape[-1] != int(cache["num_edges"]):
    raise RuntimeError("sample capacity edge dimension does not match cache num_edges")
if s["future_capacities"].shape[0] != int(cache["num_edges"]):
    raise RuntimeError("future capacity edge dimension does not match cache num_edges")

forbidden_candidate_keys = [k for k in s.keys() if k.startswith("failure_candidate_")]
if forbidden_candidate_keys:
    raise RuntimeError(f"Forbidden candidate artifacts found: {forbidden_candidate_keys}")

print("first-sample label classes seen:", sorted(set(int(x) for x in s["future_edge_state_labels"].flatten())))
print("Full-KDL shared-cache sanity check passed.")
PY

cat > "$CONFIG_PATH" <<EOF_CONFIG
EXP_NAME=$EXP_NAME
RUN_MODE=$RUN_MODE
TOPO=$TOPO
DATASET=$DATASET
CHECKPOINT=$CHECKPOINT
EPOCHS=$EPOCHS
BATCH_SIZE=$BATCH_SIZE
LR=$LR
NUM_PATHS=$NUM_PATHS
NUM_HEADS=$NUM_HEADS
NUM_TRANSFORMER_LAYERS=$NUM_TRANSFORMER_LAYERS
NUM_GNN_LAYERS=$NUM_GNN_LAYERS
NUM_MLP1_LAYERS=$NUM_MLP1_LAYERS
NUM_MLP2_LAYERS=$NUM_MLP2_LAYERS
NUM_FOR_LOOPS=$NUM_FOR_LOOPS
PRED=$PRED
TRAIN=$TRAIN_START:$TRAIN_END
VAL=$VAL_START:$VAL_END
TEST=$TEST_START:$TEST_END
PATH_CHUNK_THRESHOLD=$PATH_CHUNK_THRESHOLD
PATH_CHUNK_SIZE=$PATH_CHUNK_SIZE
FAILURE_AGGREGATION_CHUNK_SIZE=$FAILURE_AGGREGATION_CHUNK_SIZE
PATH_CHUNK_CHECKPOINT=$PATH_CHUNK_CHECKPOINT
USE_DYNAMIC_OPT=$USE_DYNAMIC_OPT
SPLIT_LOSS_WEIGHT=$SPLIT_LOSS_WEIGHT
USE_RESILIENCE_OBJECTIVE=$USE_RESILIENCE_OBJECTIVE
RESILIENCE_LOSS_WEIGHT=$RESILIENCE_LOSS_WEIGHT
FAILURE_PREDICTION_LOSS_WEIGHT=$FAILURE_PREDICTION_LOSS_WEIGHT
NUM_FAILURE_STATES=$NUM_FAILURE_STATES
NUM_FAILURE_SCENARIOS=$NUM_FAILURE_SCENARIOS
SCENARIO_PROBABILITY_LOSS_WEIGHT=$SCENARIO_PROBABILITY_LOSS_WEIGHT
SCENARIO_CAPACITY_MODE=$SCENARIO_CAPACITY_MODE
DETACH_SCENARIO_PROBABILITIES=$DETACH_SCENARIO_PROBABILITIES
DETACH_SCENARIO_CAPACITIES=$DETACH_SCENARIO_CAPACITIES
SOFT_BACKUP_PENALTY_WEIGHT=$SOFT_BACKUP_PENALTY_WEIGHT
FUTURE_LOCAL_RESCALE=$FUTURE_LOCAL_RESCALE
DISCONNECTED_PENALTY=$DISCONNECTED_PENALTY
OBJECTIVE=current_mlu + lambda * expected_scenario_mlu + beta * WTA_scenario_loss
NOTE=Full KDL shared-cache dataset. Full topology, all 567762 directed SD pairs, K=4 paths, 2271048 total paths.
EOF_CONFIG

echo "Saved config to $CONFIG_PATH"
echo "Starting full-KDL GRATE topology-mode training..."

python - <<PY
import os
import sys
import runpy
import torch

if torch.cuda.is_available():
    print("Disabling flash/mem-efficient SDPA; forcing math SDPA")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# run_harp.py does not currently expose the chunking flags as CLI args.
# Patch HARP.__init__ so the same HARP model receives path-chunking props.
import frameworks.harp_system as harp_system

_original_harp_init = harp_system.HARP.__init__

def _chunked_harp_init(self, props):
    props.path_chunk_threshold = int(os.environ.get("PATH_CHUNK_THRESHOLD", "0"))
    props.path_chunk_size = int(os.environ.get("PATH_CHUNK_SIZE", "5000"))
    props.failure_aggregation_chunk_size = int(os.environ.get("FAILURE_AGGREGATION_CHUNK_SIZE", "200000"))
    props.path_chunk_checkpoint = bool(int(os.environ.get("PATH_CHUNK_CHECKPOINT", "1")))
    print("Injected HARP path chunking props:")
    print("  path_chunk_threshold:", props.path_chunk_threshold)
    print("  path_chunk_size:", props.path_chunk_size)
    print("  failure_aggregation_chunk_size:", props.failure_aggregation_chunk_size)
    print("  path_chunk_checkpoint:", props.path_chunk_checkpoint)
    return _original_harp_init(self, props)

harp_system.HARP.__init__ = _chunked_harp_init

sys.argv = [
    "run_harp.py",
    "--topo", "$TOPO",
    "--epochs", "$EPOCHS",
    "--batch_size", "$BATCH_SIZE",
    "--lr", "$LR",

    "--num_paths_per_pair", "$NUM_PATHS",
    "--num_heads", "$NUM_HEADS",
    "--num_transformer_layers", "$NUM_TRANSFORMER_LAYERS",
    "--num_gnn_layers", "$NUM_GNN_LAYERS",
    "--num_mlp1_hidden_layers", "$NUM_MLP1_LAYERS",
    "--num_mlp2_hidden_layers", "$NUM_MLP2_LAYERS",
    "--num_for_loops", "$NUM_FOR_LOOPS",
    "--pred", "$PRED",

    "--dynamic_samples_dir", "$DATASET",
    "--dynamic_train_start_idx", "$TRAIN_START",
    "--dynamic_train_end_idx", "$TRAIN_END",
    "--dynamic_val_start_idx", "$VAL_START",
    "--dynamic_val_end_idx", "$VAL_END",
    "--dynamic_test_start_idx", "$TEST_START",
    "--dynamic_test_end_idx", "$TEST_END",

    "--use_dynamic_opt", "$USE_DYNAMIC_OPT",
    "--split_loss_weight", "$SPLIT_LOSS_WEIGHT",

    "--use_resilience_objective", "$USE_RESILIENCE_OBJECTIVE",
    "--resilience_loss_weight", "$RESILIENCE_LOSS_WEIGHT",
    "--failure_prediction_loss_weight", "$FAILURE_PREDICTION_LOSS_WEIGHT",
    "--num_failure_states", "$NUM_FAILURE_STATES",
    "--num_failure_scenarios", "$NUM_FAILURE_SCENARIOS",
    "--scenario_probability_loss_weight", "$SCENARIO_PROBABILITY_LOSS_WEIGHT",
    "--scenario_capacity_mode", "$SCENARIO_CAPACITY_MODE",
    "--detach_scenario_probabilities", "$DETACH_SCENARIO_PROBABILITIES",
    "--detach_scenario_capacities", "$DETACH_SCENARIO_CAPACITIES",
    "--soft_backup_penalty_weight", "$SOFT_BACKUP_PENALTY_WEIGHT",
    "--future_local_rescale", "$FUTURE_LOCAL_RESCALE",
    "--disconnected_penalty", "$DISCONNECTED_PENALTY",
]

print("run_harp argv:")
print(" ".join(sys.argv))
runpy.run_path("run_harp.py", run_name="__main__")
PY

if [[ -f "$RAW_MODEL" ]]; then
  cp "$RAW_MODEL" "$CHECKPOINT"
  echo "Saved checkpoint to $CHECKPOINT"
elif [[ -f "$CHECKPOINT" ]]; then
  echo "Checkpoint already exists at $CHECKPOINT"
else
  echo "WARNING: Could not find raw model $RAW_MODEL or checkpoint $CHECKPOINT"
  echo "Recent model/checkpoint files:"
  find . -maxdepth 2 -type f \( -name '*dynamic*4sp*.pkl' -o -name '*.pt' \) -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20
fi

echo "Done with $EXP_NAME"
echo "Final GPU state:"
nvidia-smi || true