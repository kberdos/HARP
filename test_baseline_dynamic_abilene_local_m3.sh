#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

if [ -d "harp_env" ]; then
  source harp_env/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi

TEST_START="${TEST_START:-800}"
TEST_END="${TEST_END:-1000}"

echo "Running local snapshot-only HARP baseline test"
echo "Working dir: $(pwd)"
echo "Test slice: [${TEST_START}, ${TEST_END})"

TEST_CMD=(
python3 run_baseline_harp_dynamic.py
  --mode test
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
)

if [ "${LOG_TO_FILE:-0}" = "1" ]; then
  "${TEST_CMD[@]}" 2>&1 | tee logs/test_baseline_dynamic_abilene_local_m3.log
else
  "${TEST_CMD[@]}"
fi
