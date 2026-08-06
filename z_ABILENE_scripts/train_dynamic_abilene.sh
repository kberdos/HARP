#!/bin/bash
#SBATCH -J grate_abilene_modes
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/grate_abilene_modes_%j.out
#SBATCH -e logs/grate_abilene_modes_%j.err

set -euo pipefail

cd /oscar/home/rkapoor8/CS2680/HARP
mkdir -p logs configs checkpoints
source harp_env/bin/activate

echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Date: $(date)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY

TOPO="dynamic_abilene"
DATASET="dynamic_abilene_h6_realistic_learned_failure"
EXP_NAME="h6_abilene_grate_topology_modes_K5_lambda0p5_beta0p1_rau1_lr1e4_epochs5_split700_850_1000"
CHECKPOINT="checkpoints/${EXP_NAME}.pt"
CONFIG_PATH="configs/${EXP_NAME}.txt"

EPOCHS=5
BATCH_SIZE=1
LR=0.0001
NUM_PATHS=4
NUM_TRANSFORMER_LAYERS=2
NUM_GNN_LAYERS=3
NUM_MLP1_LAYERS=1
NUM_MLP2_LAYERS=1
NUM_FOR_LOOPS=1
PRED=0

TRAIN_START=0
TRAIN_END=700
VAL_START=705
VAL_END=850
TEST_START=855
TEST_END=1000

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

RAW_MODEL="HARP_dynamic_${TOPO}_pred_False_${NUM_PATHS}sp.pkl"

if [[ ! -d "$DATASET" ]]; then
  echo "ERROR: dataset directory not found: $DATASET"
  exit 1
fi

if [[ ! -f "run_harp.py" ]]; then
  echo "ERROR: run_harp.py not found."
  exit 1
fi

if [[ ! -f "utils/training_utils.py" ]]; then
  echo "ERROR: utils/training_utils.py not found."
  exit 1
fi

if grep -q "generate_heuristic_candidates" utils/training_utils.py; then
  echo "ERROR: utils/training_utils.py still calls generate_heuristic_candidates during training."
  echo "ERROR: Copy the K-scenario GRATE topology-mode training_utils.py before running."
  exit 1
fi

if ! grep -qi "K-scenario" utils/training_utils.py; then
  echo "WARNING: utils/training_utils.py does not mention K-scenario."
  echo "WARNING: Make sure you copied the K-scenario GRATE topology-mode training_utils.py."
fi

python -m py_compile run_harp.py utils/training_utils.py frameworks/harp_system.py

echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET"

echo "Sanity-checking learned-failure dataset..."
DATASET_ENV="$DATASET" python - <<'PY'
import os
import glob
import torch
from pathlib import Path

root = Path(os.environ["DATASET_ENV"])
files = sorted(glob.glob(str(root / "sample_*.pt")))
print("Found", len(files), "samples")
if len(files) == 0:
    raise RuntimeError("No sample files found")

s = torch.load(files[0], map_location="cpu")
print("First sample:", files[0])
print("Keys:", list(s.keys()))

required = ["future_capacities", "future_capacity_multipliers", "future_edge_state_labels"]
for k in required:
    if k not in s:
        raise RuntimeError(f"Missing {k}")
    print(k, "shape:", tuple(s[k].shape))

forbidden = [
    "failure_candidate_probs",
    "failure_candidate_capacities",
    "failure_candidate_capacity_multipliers",
    "failure_candidate_metadata",
]
leaked = [k for k in forbidden if k in s]
if leaked:
    raise RuntimeError(f"Forbidden candidate artifacts found in dataset: {leaked}")

print("label classes seen:", sorted(set(int(x) for x in s["future_edge_state_labels"].flatten())))
print("Learned-failure sample sanity check passed.")
PY

cat > "$CONFIG_PATH" <<EOF_CONFIG
EXP_NAME=$EXP_NAME
TOPO=$TOPO
DATASET=$DATASET
CHECKPOINT=$CHECKPOINT
EPOCHS=$EPOCHS
BATCH_SIZE=$BATCH_SIZE
LR=$LR
NUM_PATHS=$NUM_PATHS
NUM_TRANSFORMER_LAYERS=$NUM_TRANSFORMER_LAYERS
NUM_GNN_LAYERS=$NUM_GNN_LAYERS
NUM_MLP1_LAYERS=$NUM_MLP1_LAYERS
NUM_MLP2_LAYERS=$NUM_MLP2_LAYERS
NUM_FOR_LOOPS=$NUM_FOR_LOOPS
PRED=$PRED
TRAIN=$TRAIN_START:$TRAIN_END
VAL=$VAL_START:$VAL_END
TEST=$TEST_START:$TEST_END
USE_DYNAMIC_OPT=$USE_DYNAMIC_OPT
USE_RESILIENCE_OBJECTIVE=$USE_RESILIENCE_OBJECTIVE
RESILIENCE_LOSS_WEIGHT=$RESILIENCE_LOSS_WEIGHT
FAILURE_PREDICTION_LOSS_WEIGHT=$FAILURE_PREDICTION_LOSS_WEIGHT
NUM_FAILURE_SCENARIOS=$NUM_FAILURE_SCENARIOS
SCENARIO_PROBABILITY_LOSS_WEIGHT=$SCENARIO_PROBABILITY_LOSS_WEIGHT
SCENARIO_CAPACITY_MODE=$SCENARIO_CAPACITY_MODE
DETACH_SCENARIO_PROBABILITIES=$DETACH_SCENARIO_PROBABILITIES
DETACH_SCENARIO_CAPACITIES=$DETACH_SCENARIO_CAPACITIES
SOFT_BACKUP_PENALTY_WEIGHT=$SOFT_BACKUP_PENALTY_WEIGHT
FUTURE_LOCAL_RESCALE=$FUTURE_LOCAL_RESCALE
DISCONNECTED_PENALTY=$DISCONNECTED_PENALTY
OBJECTIVE=current_mlu + lambda * sum_k P_GRATE(scenario_k) * MLU(scenario_k) + beta * WTA_scenario_loss
EOF_CONFIG

echo "Saved config to $CONFIG_PATH"
echo "Starting training..."

python run_harp.py \
  --topo "$TOPO" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --num_paths_per_pair "$NUM_PATHS" \
  --num_transformer_layers "$NUM_TRANSFORMER_LAYERS" \
  --num_gnn_layers "$NUM_GNN_LAYERS" \
  --num_mlp1_hidden_layers "$NUM_MLP1_LAYERS" \
  --num_mlp2_hidden_layers "$NUM_MLP2_LAYERS" \
  --num_for_loops "$NUM_FOR_LOOPS" \
  --pred "$PRED" \
  --dynamic_samples_dir "$DATASET" \
  --dynamic_train_start_idx "$TRAIN_START" \
  --dynamic_train_end_idx "$TRAIN_END" \
  --dynamic_val_start_idx "$VAL_START" \
  --dynamic_val_end_idx "$VAL_END" \
  --dynamic_test_start_idx "$TEST_START" \
  --dynamic_test_end_idx "$TEST_END" \
  --use_dynamic_opt "$USE_DYNAMIC_OPT" \
  --split_loss_weight "$SPLIT_LOSS_WEIGHT" \
  --use_resilience_objective "$USE_RESILIENCE_OBJECTIVE" \
  --resilience_loss_weight "$RESILIENCE_LOSS_WEIGHT" \
  --failure_prediction_loss_weight "$FAILURE_PREDICTION_LOSS_WEIGHT" \
  --num_failure_states "$NUM_FAILURE_STATES" \
  --num_failure_scenarios "$NUM_FAILURE_SCENARIOS" \
  --scenario_probability_loss_weight "$SCENARIO_PROBABILITY_LOSS_WEIGHT" \
  --scenario_capacity_mode "$SCENARIO_CAPACITY_MODE" \
  --detach_scenario_probabilities "$DETACH_SCENARIO_PROBABILITIES" \
  --detach_scenario_capacities "$DETACH_SCENARIO_CAPACITIES" \
  --soft_backup_penalty_weight "$SOFT_BACKUP_PENALTY_WEIGHT" \
  --future_local_rescale "$FUTURE_LOCAL_RESCALE" \
  --disconnected_penalty "$DISCONNECTED_PENALTY"

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
