#!/bin/bash
#SBATCH -J h6_res_sweep
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/h6_resilience_sweep_%j.log
#SBATCH -e logs/h6_resilience_sweep_%j.log

set -euo pipefail

cd /oscar/home/rkapoor8/CS2680/HARP

mkdir -p logs checkpoints configs results

source harp_env/bin/activate

echo "============================================================"
echo "H6 resilience hyperparameter sweep + validation selection + final test"
echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-manual}"
echo "============================================================"

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

# ------------------------------------------------------------
# Shared experiment config
# ------------------------------------------------------------

SWEEP_NAME="h6_resilience_sweep_select_test_split700_850_1000"

DATASET="dynamic_abilene_h6_realistic_candidates"

EPOCHS=5
BATCH_SIZE=1
LR=0.0001

NUM_PATHS=4
NUM_TRANSFORMER_LAYERS=2
NUM_GNN_LAYERS=3
NUM_MLP1_LAYERS=1
NUM_MLP2_LAYERS=1
NUM_FOR_LOOPS=1

# No-overlap split:
# Train samples: 0-699
# Gap:           700-704
# Val samples:   705-849
# Gap:           850-854
# Test samples:  855-999
TRAIN_START=0
TRAIN_END=700

VAL_START=705
VAL_END=850

TEST_START=855
TEST_END=1000

# ------------------------------------------------------------
# Selection policy
# ------------------------------------------------------------

# For now, automatically select the best config among local_rescale=1 runs.
# Set this to 0 if you want to select/test the harsher no-local-rescale setting.
SELECT_LOCAL_RESCALE=1

# Only select configs whose validation raw/current MLU is not more than 10%
# worse than the lambda=0 current-only baseline for the same local_rescale group.
CURRENT_MLU_TOLERANCE=1.10

# ------------------------------------------------------------
# Fixed loss config
# ------------------------------------------------------------

USE_DYNAMIC_OPT=0
SPLIT_LOSS_WEIGHT=0
SOFT_BACKUP_PENALTY_WEIGHT=0.0
DISCONNECTED_PENALTY=100.0

MODEL_FILE="HARP_dynamic_dynamic_abilene_pred_False_4sp.pkl"

# This is the metrics CSV written by the updated training_utils.py.
GLOBAL_METRICS_FILE="results/dynamic_metrics_dynamic_abilene_${NUM_PATHS}sp.csv"

SWEEP_SUMMARY="results/${SWEEP_NAME}_summary.csv"
RANKED_ALL="results/${SWEEP_NAME}_ranked_all.csv"
RANKED_SELECTED_GROUP="results/${SWEEP_NAME}_ranked_selected_local_rescale${SELECT_LOCAL_RESCALE}.csv"
BEST_ENV="results/${SWEEP_NAME}_best.env"

# ------------------------------------------------------------
# Sweep configs
#
# Format:
#   RUN_TAG USE_RESILIENCE_OBJECTIVE RESILIENCE_LOSS_WEIGHT CANDIDATE_LOCAL_RESCALE
#
# Note:
#   lambda0p0_rescale1 is the current-only baseline.
#   We keep USE_RESILIENCE_OBJECTIVE=1 even when lambda=0.0 so that
#   expected_failure_mlu and resilience_score are still computed for evaluation.
# ------------------------------------------------------------

CONFIGS=(
  "lambda0p0_rescale1            1 0.0 1"
  "lambda0p25_rescale1           1 0.25 1"
  "lambda0p5_rescale1            1 0.5 1"
  "lambda1p0_rescale1            1 1.0 1"
  "lambda2p0_rescale1            1 2.0 1"
  "lambda0p0_rescale0            1 0.0 0"
  "lambda0p5_rescale0            1 0.5 0"
  "lambda1p0_rescale0            1 1.0 0"
  "lambda2p0_rescale0            1 2.0 0"
)

# ------------------------------------------------------------
# Dataset sanity checks
# ------------------------------------------------------------

echo "Dataset: ${DATASET}"

if [[ ! -d "${DATASET}" ]]; then
  echo "ERROR: dataset directory ${DATASET} does not exist."
  exit 1
fi

NUM_SAMPLES=$(find "${DATASET}" -name "sample_*.pt" | wc -l)
echo "Found ${NUM_SAMPLES} samples"

if [[ "${NUM_SAMPLES}" -lt 1000 ]]; then
  echo "WARNING: expected around 1000 samples, found ${NUM_SAMPLES}"
fi

python - <<PY
import torch
from pathlib import Path

dataset = Path("${DATASET}")
files = sorted(dataset.glob("sample_*.pt"))

if not files:
    raise RuntimeError(f"No sample_*.pt files found in {dataset}")

s = torch.load(files[0], map_location="cpu")

required = [
    "failure_candidate_probs",
    "failure_candidate_capacities",
]

print("First sample:", files[0])
print("Keys:", list(s.keys()))

for k in required:
    if k not in s:
        raise RuntimeError(f"Missing required key: {k}")

print("failure_candidate_probs shape:", tuple(s["failure_candidate_probs"].shape))
print("failure_candidate_capacities shape:", tuple(s["failure_candidate_capacities"].shape))
print("prob sum:", float(s["failure_candidate_probs"].sum()))

if abs(float(s["failure_candidate_probs"].sum()) - 1.0) > 1e-4:
    raise RuntimeError("failure_candidate_probs do not sum to 1")

print("Candidate sanity check passed.")
PY

# ------------------------------------------------------------
# Sweep summary header
# ------------------------------------------------------------

cat > "${SWEEP_SUMMARY}" <<EOF
exp_name,run_tag,use_resilience_objective,resilience_loss_weight,candidate_local_rescale,epochs,lr,num_for_loops,train_start,train_end,val_start,val_end,test_start,test_end,checkpoint,metrics_csv,config_txt
EOF

echo "Sweep summary will be written to ${SWEEP_SUMMARY}"

# ------------------------------------------------------------
# Run sweep: train + validation only
# ------------------------------------------------------------

for CONFIG in "${CONFIGS[@]}"; do
  read -r RUN_TAG USE_RESILIENCE_OBJECTIVE RESILIENCE_LOSS_WEIGHT CANDIDATE_LOCAL_RESCALE <<< "${CONFIG}"

  EXP_NAME="h6_${RUN_TAG}_rau${NUM_FOR_LOOPS}_lr1e4_epochs${EPOCHS}_split700_850_1000"

  CHECKPOINT_OUT="checkpoints/${EXP_NAME}.pt"
  CONFIG_OUT="configs/${EXP_NAME}.txt"
  RUN_METRICS_OUT="results/${EXP_NAME}_metrics.csv"

  echo ""
  echo "============================================================"
  echo "Starting run: ${EXP_NAME}"
  echo "RUN_TAG: ${RUN_TAG}"
  echo "USE_RESILIENCE_OBJECTIVE: ${USE_RESILIENCE_OBJECTIVE}"
  echo "RESILIENCE_LOSS_WEIGHT: ${RESILIENCE_LOSS_WEIGHT}"
  echo "CANDIDATE_LOCAL_RESCALE: ${CANDIDATE_LOCAL_RESCALE}"
  echo "Date: $(date)"
  echo "============================================================"

  rm -f "${MODEL_FILE}"
  rm -f "${GLOBAL_METRICS_FILE}"

  cat > "${CONFIG_OUT}" <<EOF
EXP_NAME=${EXP_NAME}
SWEEP_NAME=${SWEEP_NAME}
RUN_TAG=${RUN_TAG}
DATASET=${DATASET}
EPOCHS=${EPOCHS}
LR=${LR}
BATCH_SIZE=${BATCH_SIZE}
NUM_PATHS=${NUM_PATHS}
NUM_TRANSFORMER_LAYERS=${NUM_TRANSFORMER_LAYERS}
NUM_GNN_LAYERS=${NUM_GNN_LAYERS}
NUM_MLP1_LAYERS=${NUM_MLP1_LAYERS}
NUM_MLP2_LAYERS=${NUM_MLP2_LAYERS}
NUM_FOR_LOOPS=${NUM_FOR_LOOPS}
USE_DYNAMIC_OPT=${USE_DYNAMIC_OPT}
SPLIT_LOSS_WEIGHT=${SPLIT_LOSS_WEIGHT}
USE_RESILIENCE_OBJECTIVE=${USE_RESILIENCE_OBJECTIVE}
RESILIENCE_LOSS_WEIGHT=${RESILIENCE_LOSS_WEIGHT}
SOFT_BACKUP_PENALTY_WEIGHT=${SOFT_BACKUP_PENALTY_WEIGHT}
CANDIDATE_LOCAL_RESCALE=${CANDIDATE_LOCAL_RESCALE}
DISCONNECTED_PENALTY=${DISCONNECTED_PENALTY}
TRAIN_START=${TRAIN_START}
TRAIN_END=${TRAIN_END}
VAL_START=${VAL_START}
VAL_END=${VAL_END}
TEST_START=${TEST_START}
TEST_END=${TEST_END}
SELECT_LOCAL_RESCALE=${SELECT_LOCAL_RESCALE}
CURRENT_MLU_TOLERANCE=${CURRENT_MLU_TOLERANCE}
GIT_BRANCH=$(git branch --show-current || true)
GIT_COMMIT=$(git rev-parse --short HEAD || true)
DATE=$(date)
SLURM_JOB_ID=${SLURM_JOB_ID:-manual}
EOF

  echo "Saved config to ${CONFIG_OUT}"
  echo "Training + validating ${EXP_NAME}..."

  python3 run_harp.py \
    --topo dynamic_abilene \
    --mode train \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_paths_per_pair "${NUM_PATHS}" \
    --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}" \
    --num_gnn_layers "${NUM_GNN_LAYERS}" \
    --num_mlp1_hidden_layers "${NUM_MLP1_LAYERS}" \
    --num_mlp2_hidden_layers "${NUM_MLP2_LAYERS}" \
    --num_for_loops "${NUM_FOR_LOOPS}" \
    --framework harp \
    --pred 0 \
    --dynamic 1 \
    --dynamic_samples_dir "${DATASET}" \
    --dynamic_train_start_idx "${TRAIN_START}" \
    --dynamic_train_end_idx "${TRAIN_END}" \
    --dynamic_val_start_idx "${VAL_START}" \
    --dynamic_val_end_idx "${VAL_END}" \
    --use_dynamic_opt "${USE_DYNAMIC_OPT}" \
    --split_loss_weight "${SPLIT_LOSS_WEIGHT}" \
    --use_resilience_objective "${USE_RESILIENCE_OBJECTIVE}" \
    --resilience_loss_weight "${RESILIENCE_LOSS_WEIGHT}" \
    --soft_backup_penalty_weight "${SOFT_BACKUP_PENALTY_WEIGHT}" \
    --candidate_local_rescale "${CANDIDATE_LOCAL_RESCALE}" \
    --disconnected_penalty "${DISCONNECTED_PENALTY}"

  if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "ERROR: expected model file ${MODEL_FILE} not found after ${EXP_NAME}."
    exit 1
  fi

  cp "${MODEL_FILE}" "${CHECKPOINT_OUT}"
  echo "Saved checkpoint to ${CHECKPOINT_OUT}"

  if [[ -f "${GLOBAL_METRICS_FILE}" ]]; then
    cp "${GLOBAL_METRICS_FILE}" "${RUN_METRICS_OUT}"
    echo "Saved metrics CSV to ${RUN_METRICS_OUT}"
  else
    echo "ERROR: expected metrics file ${GLOBAL_METRICS_FILE} not found."
    exit 1
  fi

  echo "${EXP_NAME},${RUN_TAG},${USE_RESILIENCE_OBJECTIVE},${RESILIENCE_LOSS_WEIGHT},${CANDIDATE_LOCAL_RESCALE},${EPOCHS},${LR},${NUM_FOR_LOOPS},${TRAIN_START},${TRAIN_END},${VAL_START},${VAL_END},${TEST_START},${TEST_END},${CHECKPOINT_OUT},${RUN_METRICS_OUT},${CONFIG_OUT}" >> "${SWEEP_SUMMARY}"

  echo "Finished run: ${EXP_NAME}"
  echo "Date: $(date)"
done

# ------------------------------------------------------------
# Rank validation metrics and choose best checkpoint
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Ranking validation metrics and selecting best checkpoint"
echo "Selection local_rescale group: ${SELECT_LOCAL_RESCALE}"
echo "Current MLU tolerance: ${CURRENT_MLU_TOLERANCE}"
echo "============================================================"

python - <<PY
import math
from pathlib import Path

import pandas as pd

sweep_summary = Path("${SWEEP_SUMMARY}")
ranked_all_path = Path("${RANKED_ALL}")
ranked_selected_path = Path("${RANKED_SELECTED_GROUP}")
best_env_path = Path("${BEST_ENV}")

select_local_rescale = int("${SELECT_LOCAL_RESCALE}")
current_mlu_tolerance = float("${CURRENT_MLU_TOLERANCE}")

summary = pd.read_csv(sweep_summary)

rows = []

for _, row in summary.iterrows():
    metrics_path = Path(str(row["metrics_csv"]))
    if not metrics_path.exists():
        print(f"[WARN] Missing metrics file: {metrics_path}")
        continue

    df = pd.read_csv(metrics_path)

    val = df[df["stage"] == "validation"].copy()
    if len(val) == 0:
        print(f"[WARN] No validation rows in {metrics_path}")
        continue

    val["epoch"] = pd.to_numeric(val["epoch"])
    final = val[val["epoch"] == val["epoch"].max()].iloc[0]

    rows.append({
        "exp_name": row["exp_name"],
        "run_tag": row["run_tag"],
        "use_resilience_objective": int(row["use_resilience_objective"]),
        "resilience_loss_weight": float(row["resilience_loss_weight"]),
        "candidate_local_rescale": int(row["candidate_local_rescale"]),
        "epoch": int(final["epoch"]),
        "val_loss": float(final["loss_avg"]),
        "val_raw_mlu": float(final["raw_mlu_avg"]),
        "val_expected_failure_mlu": float(final["expected_failure_mlu_avg"]),
        "val_expected_failure_mlu_p95": float(final["expected_failure_mlu_p95"]),
        "val_resilience_score": float(final["resilience_score_avg"]),
        "val_expected_disconnected": float(final["expected_disconnected_avg"]),
        "checkpoint": row["checkpoint"],
        "metrics_csv": row["metrics_csv"],
        "config_txt": row["config_txt"],
    })

if not rows:
    raise RuntimeError("No validation metrics were found. Cannot select best config.")

ranked = pd.DataFrame(rows)

# Do not rank by val_loss across different lambda values because lambda changes
# the loss scale. Rank by lambda-independent evaluation metrics instead.
ranked_all = ranked.sort_values(
    ["candidate_local_rescale", "val_expected_failure_mlu", "val_expected_failure_mlu_p95", "val_raw_mlu"],
    ascending=[False, True, True, True],
)
ranked_all.to_csv(ranked_all_path, index=False)

group = ranked[ranked["candidate_local_rescale"] == select_local_rescale].copy()

if group.empty:
    raise RuntimeError(f"No configs found for candidate_local_rescale={select_local_rescale}")

# Find current-only baseline in the selected local_rescale group.
baseline = group[group["resilience_loss_weight"] == 0.0].copy()

if len(baseline) > 0:
    baseline_raw = float(baseline.iloc[0]["val_raw_mlu"])
    raw_limit = baseline_raw * current_mlu_tolerance
    eligible = group[group["val_raw_mlu"] <= raw_limit].copy()

    print(f"Baseline validation raw MLU: {baseline_raw:.6f}")
    print(f"Raw MLU selection limit: {raw_limit:.6f}")

    if eligible.empty:
        print("[WARN] No configs satisfied the current MLU guardrail. Falling back to all configs in selected group.")
        eligible = group
else:
    print("[WARN] No lambda=0.0 baseline found in selected group. Ranking all configs in selected group.")
    baseline_raw = float("nan")
    raw_limit = float("nan")
    eligible = group

ranked_selected = group.sort_values(
    ["val_expected_failure_mlu", "val_expected_failure_mlu_p95", "val_raw_mlu"],
    ascending=[True, True, True],
)

ranked_selected["baseline_val_raw_mlu"] = baseline_raw
ranked_selected["current_mlu_limit"] = raw_limit
ranked_selected["eligible_under_current_mlu_guardrail"] = ranked_selected["val_raw_mlu"] <= raw_limit if not math.isnan(raw_limit) else True

ranked_selected.to_csv(ranked_selected_path, index=False)

eligible_ranked = eligible.sort_values(
    ["val_expected_failure_mlu", "val_expected_failure_mlu_p95", "val_raw_mlu"],
    ascending=[True, True, True],
)

best = eligible_ranked.iloc[0]

print("")
print("=== Ranked selected group by validation expected failure MLU ===")
print(ranked_selected.to_string(index=False))

print("")
print("=== Best selected config ===")
print(best.to_string())

with open(best_env_path, "w") as f:
    f.write(f"export BEST_EXP_NAME='{best['exp_name']}'\n")
    f.write(f"export BEST_RUN_TAG='{best['run_tag']}'\n")
    f.write(f"export BEST_USE_RESILIENCE_OBJECTIVE='{int(best['use_resilience_objective'])}'\n")
    f.write(f"export BEST_RESILIENCE_LOSS_WEIGHT='{float(best['resilience_loss_weight'])}'\n")
    f.write(f"export BEST_CANDIDATE_LOCAL_RESCALE='{int(best['candidate_local_rescale'])}'\n")
    f.write(f"export BEST_CHECKPOINT='{best['checkpoint']}'\n")
    f.write(f"export BEST_METRICS_CSV='{best['metrics_csv']}'\n")
    f.write(f"export BEST_CONFIG_TXT='{best['config_txt']}'\n")
    f.write(f"export BEST_VAL_RAW_MLU='{float(best['val_raw_mlu'])}'\n")
    f.write(f"export BEST_VAL_EXPECTED_FAILURE_MLU='{float(best['val_expected_failure_mlu'])}'\n")
    f.write(f"export BEST_VAL_EXPECTED_FAILURE_MLU_P95='{float(best['val_expected_failure_mlu_p95'])}'\n")
    f.write(f"export BEST_VAL_RESILIENCE_SCORE='{float(best['val_resilience_score'])}'\n")

print("")
print(f"Saved ranked-all CSV: {ranked_all_path}")
print(f"Saved selected-group ranking CSV: {ranked_selected_path}")
print(f"Saved best config env: {best_env_path}")
PY

source "${BEST_ENV}"

echo ""
echo "============================================================"
echo "Best config selected from validation"
echo "BEST_EXP_NAME: ${BEST_EXP_NAME}"
echo "BEST_RUN_TAG: ${BEST_RUN_TAG}"
echo "BEST_CHECKPOINT: ${BEST_CHECKPOINT}"
echo "BEST_RESILIENCE_LOSS_WEIGHT: ${BEST_RESILIENCE_LOSS_WEIGHT}"
echo "BEST_CANDIDATE_LOCAL_RESCALE: ${BEST_CANDIDATE_LOCAL_RESCALE}"
echo "BEST_VAL_RAW_MLU: ${BEST_VAL_RAW_MLU}"
echo "BEST_VAL_EXPECTED_FAILURE_MLU: ${BEST_VAL_EXPECTED_FAILURE_MLU}"
echo "BEST_VAL_EXPECTED_FAILURE_MLU_P95: ${BEST_VAL_EXPECTED_FAILURE_MLU_P95}"
echo "BEST_VAL_RESILIENCE_SCORE: ${BEST_VAL_RESILIENCE_SCORE}"
echo "============================================================"

# ------------------------------------------------------------
# Final test on the selected best checkpoint
# ------------------------------------------------------------

echo ""
echo "Starting final held-out test on selected best checkpoint."
echo "Test samples: ${TEST_START}:${TEST_END}"

if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
  echo "ERROR: selected checkpoint does not exist: ${BEST_CHECKPOINT}"
  exit 1
fi

# run_harp.py test mode expects this filename.
cp "${BEST_CHECKPOINT}" "${MODEL_FILE}"

# Make final combined train/val/test metrics CSV.
# Start with the selected train/validation metrics, then test will append a test row.
rm -f "${GLOBAL_METRICS_FILE}"
cp "${BEST_METRICS_CSV}" "${GLOBAL_METRICS_FILE}"

python3 run_harp.py \
  --topo dynamic_abilene \
  --mode test \
  --batch_size 1 \
  --num_paths_per_pair "${NUM_PATHS}" \
  --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}" \
  --num_gnn_layers "${NUM_GNN_LAYERS}" \
  --num_mlp1_hidden_layers "${NUM_MLP1_LAYERS}" \
  --num_mlp2_hidden_layers "${NUM_MLP2_LAYERS}" \
  --num_for_loops "${NUM_FOR_LOOPS}" \
  --framework harp \
  --pred 0 \
  --dynamic 1 \
  --dynamic_samples_dir "${DATASET}" \
  --dynamic_test_start_idx "${TEST_START}" \
  --dynamic_test_end_idx "${TEST_END}" \
  --use_dynamic_opt "${USE_DYNAMIC_OPT}" \
  --split_loss_weight "${SPLIT_LOSS_WEIGHT}" \
  --use_resilience_objective "${BEST_USE_RESILIENCE_OBJECTIVE}" \
  --resilience_loss_weight "${BEST_RESILIENCE_LOSS_WEIGHT}" \
  --soft_backup_penalty_weight "${SOFT_BACKUP_PENALTY_WEIGHT}" \
  --candidate_local_rescale "${BEST_CANDIDATE_LOCAL_RESCALE}" \
  --disconnected_penalty "${DISCONNECTED_PENALTY}" \
  --test_cluster 0 \
  --failure_id 0

FINAL_METRICS_OUT="results/${BEST_EXP_NAME}_selected_train_val_test_metrics.csv"
cp "${GLOBAL_METRICS_FILE}" "${FINAL_METRICS_OUT}"

TEST_STATS_FILE="results/dynamic_abilene/${NUM_PATHS}sp/0/harp_dynamic_stats_failure_id_0.txt"
TEST_VALUES_FILE="results/dynamic_abilene/${NUM_PATHS}sp/0/harp_dynamic_values_failure_id_0.txt"

if [[ -f "${TEST_STATS_FILE}" ]]; then
  cp "${TEST_STATS_FILE}" "results/${BEST_EXP_NAME}_test_stats.txt"
fi

if [[ -f "${TEST_VALUES_FILE}" ]]; then
  cp "${TEST_VALUES_FILE}" "results/${BEST_EXP_NAME}_test_values.csv"
fi

echo ""
echo "============================================================"
echo "Sweep + validation selection + final test complete."
echo ""
echo "Best experiment:"
echo "  ${BEST_EXP_NAME}"
echo ""
echo "Ranking files:"
echo "  ${RANKED_ALL}"
echo "  ${RANKED_SELECTED_GROUP}"
echo ""
echo "Best config file:"
echo "  ${BEST_ENV}"
echo ""
echo "Final metrics:"
echo "  ${FINAL_METRICS_OUT}"
echo "  results/${BEST_EXP_NAME}_test_stats.txt"
echo "  results/${BEST_EXP_NAME}_test_values.csv"
echo ""
echo "One combined SLURM log:"
echo "  logs/h6_resilience_sweep_${SLURM_JOB_ID}.log"
echo "============================================================"