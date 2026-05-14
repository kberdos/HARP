#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

if [ -d "harp_env" ]; then
  source harp_env/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-8}"

EPOCHS="${EPOCHS:-20}"
TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-800}"
VAL_START="${VAL_START:-800}"
VAL_END="${VAL_END:-1000}"

CURRENT_WEIGHT="${CURRENT_WEIGHT:-1.0}"
RESILIENCE_WEIGHT="${RESILIENCE_WEIGHT:-0.25}"
WORST_CASE_WEIGHT="${WORST_CASE_WEIGHT:-0.5}"
FAILURE_CAPACITY_FRACTION="${FAILURE_CAPACITY_FRACTION:-0.25}"
RISK_PRIOR="${RISK_PRIOR:-0.001}"
RISK_RECENCY_POWER="${RISK_RECENCY_POWER:-2.0}"
SCENARIO_TOP_K="${SCENARIO_TOP_K:-0}"

echo "Running local future-failure resilient temporal HARP training"
echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('mps:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"

echo "Epochs: ${EPOCHS}"
echo "Train slice: [${TRAIN_START}, ${TRAIN_END})"
echo "Val slice: [${VAL_START}, ${VAL_END})"
echo "Resilience weight: ${RESILIENCE_WEIGHT}"
echo "Failure capacity fraction: ${FAILURE_CAPACITY_FRACTION}"

TRAIN_CMD=(
python3 run_resilient_harp_dynamic.py
  --mode train
  --model_type temporal
  --samples_dir dynamic_abilene_h6_1000_samples
  --topo dynamic_abilene
  --epochs "${EPOCHS}"
  --lr 0.00005
  --train_start_idx "${TRAIN_START}"
  --train_end_idx "${TRAIN_END}"
  --val_start_idx "${VAL_START}"
  --val_end_idx "${VAL_END}"
  --num_paths_per_pair 4
  --num_transformer_layers 2
  --num_gnn_layers 3
  --num_mlp1_hidden_layers 1
  --num_mlp2_hidden_layers 1
  --num_for_loops 3
  --current_weight "${CURRENT_WEIGHT}"
  --resilience_weight "${RESILIENCE_WEIGHT}"
  --worst_case_weight "${WORST_CASE_WEIGHT}"
  --failure_capacity_fraction "${FAILURE_CAPACITY_FRACTION}"
  --risk_prior "${RISK_PRIOR}"
  --risk_recency_power "${RISK_RECENCY_POWER}"
  --scenario_top_k "${SCENARIO_TOP_K}"
)

if [ "${LOG_TO_FILE:-0}" = "1" ]; then
  "${TRAIN_CMD[@]}" 2>&1 | tee logs/train_resilient_dynamic_abilene_local_m3.log
else
  "${TRAIN_CMD[@]}"
fi
