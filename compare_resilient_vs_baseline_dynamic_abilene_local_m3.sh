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

RESILIENT_MODEL_PATH="${RESILIENT_MODEL_PATH:-HARP_resilient_temporal_dynamic_abilene_pred_False_4sp.pkl}"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-HARP_baseline_dynamic_abilene_pred_False_4sp.pkl}"

CURRENT_WEIGHT="${CURRENT_WEIGHT:-1.0}"
RESILIENCE_WEIGHT="${RESILIENCE_WEIGHT:-0.25}"
WORST_CASE_WEIGHT="${WORST_CASE_WEIGHT:-0.5}"
FAILURE_CAPACITY_FRACTION="${FAILURE_CAPACITY_FRACTION:-0.25}"
RISK_PRIOR="${RISK_PRIOR:-1.0}"
RISK_RECENCY_POWER="${RISK_RECENCY_POWER:-2.0}"
SCENARIO_TOP_K="${SCENARIO_TOP_K:-0}"

COMPARE_ROOT="${COMPARE_ROOT:-results/dynamic_abilene/4sp/0/resilience_compare}"
RESILIENT_RESULTS_DIR="${COMPARE_ROOT}/resilient_temporal"
BASELINE_RESULTS_DIR="${COMPARE_ROOT}/snapshot_baseline"

echo "Comparing resilient temporal HARP vs snapshot-only HARP baseline"
echo "Working dir: $(pwd)"
echo "Test slice: [${TEST_START}, ${TEST_END})"
echo "Resilient model: ${RESILIENT_MODEL_PATH}"
echo "Baseline model: ${BASELINE_MODEL_PATH}"
echo "Compare root: ${COMPARE_ROOT}"

if [ ! -f "${RESILIENT_MODEL_PATH}" ]; then
  echo "Missing resilient checkpoint: ${RESILIENT_MODEL_PATH}"
  exit 1
fi

if [ ! -f "${BASELINE_MODEL_PATH}" ]; then
  echo "Missing baseline checkpoint: ${BASELINE_MODEL_PATH}"
  exit 1
fi

COMMON_ARGS=(
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

echo
echo "==> Testing resilient temporal model"
python3 run_resilient_harp_dynamic.py \
  --mode test \
  --model_type temporal \
  --model_path "${RESILIENT_MODEL_PATH}" \
  --results_dir "${RESILIENT_RESULTS_DIR}" \
  "${COMMON_ARGS[@]}"

echo
echo "==> Testing snapshot-only baseline model"
python3 run_resilient_harp_dynamic.py \
  --mode test \
  --model_type baseline \
  --model_path "${BASELINE_MODEL_PATH}" \
  --results_dir "${BASELINE_RESULTS_DIR}" \
  "${COMMON_ARGS[@]}"

echo
echo "==> Comparison summary"
COMPARE_ROOT="${COMPARE_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

compare_root = Path(os.environ["COMPARE_ROOT"])
models = {
    "resilient_temporal": compare_root / "resilient_temporal",
    "snapshot_baseline": compare_root / "snapshot_baseline",
}
metrics = ["combined", "current", "expected_failure", "worst_failure"]


def read_average(stats_path):
    for line in stats_path.read_text().splitlines():
        if line.startswith("Average:"):
            return float(line.split(":", 1)[1].strip())
    raise ValueError(f"No Average line in {stats_path}")


rows = {}
for model_name, directory in models.items():
    rows[model_name] = {}
    for metric in metrics:
        matches = sorted(directory.glob(f"*_{metric}_stats.txt"))
        if not matches:
            raise FileNotFoundError(f"Missing {metric} stats in {directory}")
        rows[model_name][metric] = read_average(matches[0])

print("metric              resilient_temporal    snapshot_baseline    improvement")
print("------------------  ------------------    -----------------    -----------")
for metric in metrics:
    resilient = rows["resilient_temporal"][metric]
    baseline = rows["snapshot_baseline"][metric]
    improvement = (baseline - resilient) / baseline * 100.0
    print(f"{metric:<18}  {resilient:>18.6f}    {baseline:>17.6f}    {improvement:>9.2f}%")

print()
print(f"Detailed files are under: {compare_root}")
PY
