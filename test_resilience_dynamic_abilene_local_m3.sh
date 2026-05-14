#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

if [ -d "harp_env" ]; then
  source harp_env/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi

MODEL_TYPE="${MODEL_TYPE:-temporal}"
MODEL_PATH="${MODEL_PATH:-}"
TEST_START="${TEST_START:-800}"
TEST_END="${TEST_END:-1000}"

CURRENT_WEIGHT="${CURRENT_WEIGHT:-1.0}"
RESILIENCE_WEIGHT="${RESILIENCE_WEIGHT:-0.25}"
WORST_CASE_WEIGHT="${WORST_CASE_WEIGHT:-0.5}"
FAILURE_CAPACITY_FRACTION="${FAILURE_CAPACITY_FRACTION:-0.25}"
RISK_PRIOR="${RISK_PRIOR:-0.001}"
RISK_RECENCY_POWER="${RISK_RECENCY_POWER:-2.0}"
SCENARIO_TOP_K="${SCENARIO_TOP_K:-0}"

echo "Running local future-failure resilience test"
echo "Working dir: $(pwd)"
echo "Model type: ${MODEL_TYPE}"
echo "Test slice: [${TEST_START}, ${TEST_END})"

TEST_CMD=(
python3 run_resilient_harp_dynamic.py
  --mode test
  --model_type "${MODEL_TYPE}"
  --samples_dir dynamic_abilene_h6_1000_samples
  --topo dynamic_abilene
  --test_cluster 0
  --test_start_idx "${TEST_START}"
  --test_end_idx "${TEST_END}"
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

if [ -n "${MODEL_PATH}" ]; then
  TEST_CMD+=(--model_path "${MODEL_PATH}")
fi

if [ "${LOG_TO_FILE:-0}" = "1" ]; then
  "${TEST_CMD[@]}" 2>&1 | tee logs/test_resilience_dynamic_abilene_local_m3.log
else
  "${TEST_CMD[@]}"
fi
