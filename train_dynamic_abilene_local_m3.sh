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

echo "Running local dynamic Abilene training"
echo "Working dir: $(pwd)"
echo "Python: $(which python3)"
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('mps:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"

if [ ! -d dynamic_abilene_h6_1000_samples ]; then
  echo "Missing dynamic_abilene_h6_1000_samples. Generate or copy the samples first."
  exit 1
fi

EPOCHS="${EPOCHS:-20}"
TRAIN_START="${TRAIN_START:-0}"
TRAIN_END="${TRAIN_END:-800}"
VAL_START="${VAL_START:-800}"
VAL_END="${VAL_END:-1000}"

echo "Epochs: ${EPOCHS}"
echo "Train slice: [${TRAIN_START}, ${TRAIN_END})"
echo "Val slice: [${VAL_START}, ${VAL_END})"

TRAIN_CMD=(
python3 run_harp.py
  --topo dynamic_abilene \
  --mode train \
  --epochs "${EPOCHS}" \
  --lr 0.00005 \
  --batch_size 1 \
  --num_paths_per_pair 4 \
  --num_transformer_layers 2 \
  --num_gnn_layers 3 \
  --num_mlp1_hidden_layers 1 \
  --num_mlp2_hidden_layers 1 \
  --num_for_loops 3 \
  --framework harp \
  --pred 0 \
  --dynamic 1 \
  --dynamic_samples_dir dynamic_abilene_h6_1000_samples \
  --dynamic_train_start_idx "${TRAIN_START}" \
  --dynamic_train_end_idx "${TRAIN_END}" \
  --dynamic_val_start_idx "${VAL_START}" \
  --dynamic_val_end_idx "${VAL_END}" \
  --use_dynamic_opt 1
)

if [ "${LOG_TO_FILE:-0}" = "1" ]; then
  "${TRAIN_CMD[@]}" 2>&1 | tee logs/train_dynamic_abilene_local_m3.log
else
  "${TRAIN_CMD[@]}"
fi
