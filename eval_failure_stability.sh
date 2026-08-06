#!/bin/bash
#SBATCH -J eval_failure_stability
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/eval_failure_stability_%j.out
#SBATCH -e logs/eval_failure_stability_%j.err

set -euo pipefail

# ------------------------------------------------------------
# General paths
# ------------------------------------------------------------
HARP_ROOT="${HARP_ROOT:-/oscar/home/rkapoor8/CS2680/HARP}"

cd "${HARP_ROOT}"

mkdir -p logs results configs checkpoints

source harp_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Running on host: $(hostname)"
echo "Working dir: $(pwd)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current || true)"
echo "Commit: $(git rev-parse --short HEAD || true)"

python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"

# ------------------------------------------------------------
# Required / commonly changed evaluation config
# Override these from the command line when submitting.
# ------------------------------------------------------------

# TOPO="${TOPO:-dynamic_geant}"
# DATASET="${DATASET:-dynamic_geant_h6_realistic_learned_failure}"
# CHECKPOINT="${CHECKPOINT:-checkpoints/h6_geant_grate_topology_modes_K5_lambda0p5_beta0p1_rau1_lr1e4_epochs5_split700_850_1000.pt}"

TOPO="${TOPO:-dynamic_abilene}"
DATASET="${DATASET:-dynamic_abilene_h6_realistic_learned_failure}"
CHECKPOINT="${CHECKPOINT:-checkpoints/h6_abilene_grate_topology_modes_K5_lambda0p5_beta0p1_rau1_lr1e4_epochs5_split700_850_1000.pt}"


START_IDX="${START_IDX:-855}"
END_IDX="${END_IDX:-1000}"

# ------------------------------------------------------------
# Evaluation probability source
# ------------------------------------------------------------
# Fair post-hoc scoring uses one shared heuristic candidate distribution
# for every method. GRATE's failure logits may guide its own training/routing,
# but they do not define GRATE's evaluation distribution.
PROB_SOURCE="${PROB_SOURCE:-heuristic}"

# ------------------------------------------------------------
# On-the-fly heuristic candidate generation / scoring config
# The clean dataset stores ONLY realized future labels, not candidates.
# Candidate scenarios are generated at evaluation time by heuristic_candidates.py.
# ------------------------------------------------------------
NUM_FAILURE_CANDIDATES="${NUM_FAILURE_CANDIDATES:-8}"
SKIP_DISCONNECTED_CANDIDATES="${SKIP_DISCONNECTED_CANDIDATES:-0}"
DISCONNECTED_MLU_PENALTY="${DISCONNECTED_MLU_PENALTY:-10.0}"
DISCONNECTED_PENALTY_MODE="${DISCONNECTED_PENALTY_MODE:-fraction}"  # none | fraction | any
FAIL_IF_PROBS_PRESENT="${FAIL_IF_PROBS_PRESENT:-1}"

CANDIDATE_SCORE_FLOOR="${CANDIDATE_SCORE_FLOOR:-0.000001}"
CANDIDATE_NO_CHANGE_WEIGHT="${CANDIDATE_NO_CHANGE_WEIGHT:-1.0}"
CANDIDATE_EDGE_RISK_WEIGHT="${CANDIDATE_EDGE_RISK_WEIGHT:-1.0}"
CANDIDATE_RECENT_HISTORY_WEIGHT="${CANDIDATE_RECENT_HISTORY_WEIGHT:-2.0}"
CANDIDATE_CURRENT_IMPAIRMENT_WEIGHT="${CANDIDATE_CURRENT_IMPAIRMENT_WEIGHT:-3.0}"
CANDIDATE_FLAKY_BONUS_WEIGHT="${CANDIDATE_FLAKY_BONUS_WEIGHT:-1.0}"
CANDIDATE_WORSEN_WEIGHT="${CANDIDATE_WORSEN_WEIGHT:-1.25}"
CANDIDATE_FULL_FAILURE_WEIGHT="${CANDIDATE_FULL_FAILURE_WEIGHT:-0.70}"
CANDIDATE_CORRELATED_WEIGHT="${CANDIDATE_CORRELATED_WEIGHT:-0.80}"
CANDIDATE_RECOVERY_WEIGHT="${CANDIDATE_RECOVERY_WEIGHT:-0.40}"

# ------------------------------------------------------------
# Architecture config
# Must match the checkpoint.
# ------------------------------------------------------------
NUM_PATHS="${NUM_PATHS:-4}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-2}"
NUM_GNN_LAYERS="${NUM_GNN_LAYERS:-3}"
NUM_MLP1_LAYERS="${NUM_MLP1_LAYERS:-1}"
NUM_MLP2_LAYERS="${NUM_MLP2_LAYERS:-1}"
NUM_FOR_LOOPS="${NUM_FOR_LOOPS:-1}"
PRED="${PRED:-0}"
USE_DYNAMIC_OPT="${USE_DYNAMIC_OPT:-0}"
NUM_FAILURE_STATES="${NUM_FAILURE_STATES:-5}"

# ------------------------------------------------------------
# CUDA attention backend config
#
# KDL-50k can trigger:
#   RuntimeError: CUDA error: invalid configuration argument
#
# because fused SDPA does not like the huge path batch.
#
# FORCE_MATH_SDPA:
#   auto = enable only for KDL-looking runs
#   1    = force math SDPA
#   0    = leave PyTorch defaults
# ------------------------------------------------------------
FORCE_MATH_SDPA="${FORCE_MATH_SDPA:-auto}"

if [[ "${FORCE_MATH_SDPA}" == "auto" ]]; then
  if [[ "${TOPO}" == *"kdl"* || "${DATASET}" == *"kdl"* || "${CHECKPOINT}" == *"kdl"* ]]; then
    FORCE_MATH_SDPA_RESOLVED=1
  else
    FORCE_MATH_SDPA_RESOLVED=0
  fi
else
  FORCE_MATH_SDPA_RESOLVED="${FORCE_MATH_SDPA}"
fi

# ------------------------------------------------------------
# Output config
# ------------------------------------------------------------
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
CHECKPOINT_NAME="${CHECKPOINT_NAME%.*}"

OUT_DIR="${OUT_DIR:-results/${TOPO}/${NUM_PATHS}sp/failure_stability_clean}"
mkdir -p "${OUT_DIR}"

OUT_CSV="${OUT_CSV:-${OUT_DIR}/${CHECKPOINT_NAME}_samples${START_IDX}_${END_IDX}_realized_future_and_candidates_${NUM_FAILURE_CANDIDATES}cand_${DISCONNECTED_PENALTY_MODE}_skipdisc${SKIP_DISCONNECTED_CANDIDATES}.csv}"

echo "------------------------------------------------------------"
echo "Clean learned-failure stability eval config"
echo "------------------------------------------------------------"
echo "TOPO=${TOPO}"
echo "DATASET=${DATASET}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "START_IDX=${START_IDX}"
echo "END_IDX=${END_IDX}"
echo "OUT_CSV=${OUT_CSV}"
echo
echo "Evaluation mode:"
echo "  clean learned-failure samples"
echo "  realized-future MLU from sample[future_capacities]"
echo "  on-the-fly heuristic candidate expected MLU"
echo
echo "Probability source config:"
echo "  PROB_SOURCE=${PROB_SOURCE}"
echo
echo "Candidate generation/scoring config:"
echo "  NUM_FAILURE_CANDIDATES=${NUM_FAILURE_CANDIDATES}"
echo "  SKIP_DISCONNECTED_CANDIDATES=${SKIP_DISCONNECTED_CANDIDATES}"
echo "  DISCONNECTED_MLU_PENALTY=${DISCONNECTED_MLU_PENALTY}"
echo "  DISCONNECTED_PENALTY_MODE=${DISCONNECTED_PENALTY_MODE}"
echo "  FAIL_IF_PROBS_PRESENT=${FAIL_IF_PROBS_PRESENT}"
echo "  CANDIDATE_SCORE_FLOOR=${CANDIDATE_SCORE_FLOOR}"
echo "  CANDIDATE_NO_CHANGE_WEIGHT=${CANDIDATE_NO_CHANGE_WEIGHT}"
echo "  CANDIDATE_EDGE_RISK_WEIGHT=${CANDIDATE_EDGE_RISK_WEIGHT}"
echo "  CANDIDATE_RECENT_HISTORY_WEIGHT=${CANDIDATE_RECENT_HISTORY_WEIGHT}"
echo "  CANDIDATE_CURRENT_IMPAIRMENT_WEIGHT=${CANDIDATE_CURRENT_IMPAIRMENT_WEIGHT}"
echo "  CANDIDATE_FLAKY_BONUS_WEIGHT=${CANDIDATE_FLAKY_BONUS_WEIGHT}"
echo "  CANDIDATE_WORSEN_WEIGHT=${CANDIDATE_WORSEN_WEIGHT}"
echo "  CANDIDATE_FULL_FAILURE_WEIGHT=${CANDIDATE_FULL_FAILURE_WEIGHT}"
echo "  CANDIDATE_CORRELATED_WEIGHT=${CANDIDATE_CORRELATED_WEIGHT}"
echo "  CANDIDATE_RECOVERY_WEIGHT=${CANDIDATE_RECOVERY_WEIGHT}"
echo
echo "Architecture config:"
echo "  NUM_PATHS=${NUM_PATHS}"
echo "  NUM_TRANSFORMER_LAYERS=${NUM_TRANSFORMER_LAYERS}"
echo "  NUM_GNN_LAYERS=${NUM_GNN_LAYERS}"
echo "  NUM_MLP1_LAYERS=${NUM_MLP1_LAYERS}"
echo "  NUM_MLP2_LAYERS=${NUM_MLP2_LAYERS}"
echo "  NUM_FOR_LOOPS=${NUM_FOR_LOOPS}"
echo "  PRED=${PRED}"
echo "  USE_DYNAMIC_OPT=${USE_DYNAMIC_OPT}"
echo "  NUM_FAILURE_STATES=${NUM_FAILURE_STATES}"
echo
echo "CUDA attention config:"
echo "  FORCE_MATH_SDPA=${FORCE_MATH_SDPA}"
echo "  FORCE_MATH_SDPA_RESOLVED=${FORCE_MATH_SDPA_RESOLVED}"
echo "------------------------------------------------------------"

if [[ "${PROB_SOURCE}" != "heuristic" ]]; then
  echo "ERROR: this clean/fair evaluator only supports PROB_SOURCE=heuristic. Got: ${PROB_SOURCE}"
  echo "GRATE logits can guide GRATE routing/training, but final scoring must use the same heuristic distribution for every model."
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  echo "Available checkpoints:"
  find checkpoints -type f \( -name '*.pkl' -o -name '*.pt' -o -name '*.pth' \) | sort || true
  exit 1
fi

if [[ ! -d "${DATASET}" ]]; then
  echo "ERROR: dataset directory not found: ${DATASET}"
  exit 1
fi

if [[ ! -f "eval_failure_stability.py" ]]; then
  echo "ERROR: eval_failure_stability.py not found in $(pwd)"
  exit 1
fi

if [[ ! -f "heuristic_candidates.py" ]]; then
  echo "ERROR: heuristic_candidates.py not found in $(pwd)"
  echo "Copy heuristic_candidates.py into HARP before running this script."
  exit 1
fi

# ------------------------------------------------------------
# Fast dataset sanity check: clean samples should contain realized future labels
# and should NOT contain saved candidate artifacts.
# ------------------------------------------------------------
python - <<PY
import torch
from pathlib import Path

DATASET = Path("${DATASET}")
files = sorted(DATASET.glob("sample_*.pt"))
if not files:
    raise RuntimeError(f"No sample_*.pt files found in {DATASET}")

s = torch.load(files[0], map_location="cpu")
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
missing = required - set(s.keys())
leaked = forbidden & set(s.keys())
if missing:
    raise RuntimeError(f"First sample missing required keys: {sorted(missing)}")
if leaked:
    raise RuntimeError(f"First sample contains forbidden candidate keys: {sorted(leaked)}")

T = s["node_features"].shape[1]
P = s["tm"].shape[2]
E_final = s["capacities"][-1].shape[-1]

assert len(s["edge_index"]) == T
assert len(s["capacities"]) == T
assert len(s["padded_edge_ids_per_path"]) == T
assert len(s["paths_to_edges"]) == T
assert s["paths_to_edges"][-1].shape == (P, E_final)
assert s["future_capacities"].shape == (E_final,)
assert s["future_capacity_multipliers"].shape == (E_final,)
assert s["future_edge_state_labels"].shape == (E_final,)

print("Dataset sanity check passed.")
print("  first sample:", files[0])
print("  num samples:", len(files))
print("  T:", T, "P:", P, "E_final:", E_final)
print("  future labels seen in first sample:", sorted(set(int(x) for x in s["future_edge_state_labels"].flatten())))
PY

# ------------------------------------------------------------
# Build eval args once, then run eval through a Python wrapper.
# The wrapper optionally disables fused CUDA SDPA before eval_failure_stability.py
# imports/runs the model.
# ------------------------------------------------------------
EVAL_ARGS=(
  --topo "${TOPO}"
  --checkpoint "${CHECKPOINT}"
  --dynamic_samples_dir "${DATASET}"
  --start_idx "${START_IDX}"
  --end_idx "${END_IDX}"
  --out_csv "${OUT_CSV}"

  --prob_source "${PROB_SOURCE}"
  --num_failure_candidates "${NUM_FAILURE_CANDIDATES}"
  --skip_disconnected_candidates "${SKIP_DISCONNECTED_CANDIDATES}"
  --disconnected_mlu_penalty "${DISCONNECTED_MLU_PENALTY}"
  --disconnected_penalty_mode "${DISCONNECTED_PENALTY_MODE}"
  --fail_if_probs_present "${FAIL_IF_PROBS_PRESENT}"

  --num_paths_per_pair "${NUM_PATHS}"
  --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
  --num_gnn_layers "${NUM_GNN_LAYERS}"
  --num_mlp1_hidden_layers "${NUM_MLP1_LAYERS}"
  --num_mlp2_hidden_layers "${NUM_MLP2_LAYERS}"
  --num_for_loops "${NUM_FOR_LOOPS}"
  --pred "${PRED}"
  --use_dynamic_opt "${USE_DYNAMIC_OPT}"
  --num_failure_states "${NUM_FAILURE_STATES}"

  --candidate_score_floor "${CANDIDATE_SCORE_FLOOR}"
  --candidate_no_change_weight "${CANDIDATE_NO_CHANGE_WEIGHT}"
  --candidate_edge_risk_weight "${CANDIDATE_EDGE_RISK_WEIGHT}"
  --candidate_recent_history_weight "${CANDIDATE_RECENT_HISTORY_WEIGHT}"
  --candidate_current_impairment_weight "${CANDIDATE_CURRENT_IMPAIRMENT_WEIGHT}"
  --candidate_flaky_bonus_weight "${CANDIDATE_FLAKY_BONUS_WEIGHT}"
  --candidate_worsen_weight "${CANDIDATE_WORSEN_WEIGHT}"
  --candidate_full_failure_weight "${CANDIDATE_FULL_FAILURE_WEIGHT}"
  --candidate_correlated_weight "${CANDIDATE_CORRELATED_WEIGHT}"
  --candidate_recovery_weight "${CANDIDATE_RECOVERY_WEIGHT}"
)

FORCE_MATH_SDPA_RESOLVED_ENV="${FORCE_MATH_SDPA_RESOLVED}" python3 - "${EVAL_ARGS[@]}" <<'PY'
import os
import sys
import runpy
import torch

force_math = os.environ.get("FORCE_MATH_SDPA_RESOLVED_ENV", "0") == "1"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("force math SDPA:", force_math)

if force_math and torch.cuda.is_available():
    print("Disabling flash/mem-efficient SDPA; forcing math SDPA")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

sys.argv = ["eval_failure_stability.py"] + sys.argv[1:]

print("Eval argv:")
print(" ".join(sys.argv))

runpy.run_path("eval_failure_stability.py", run_name="__main__")
PY

echo "Done writing ${OUT_CSV}"