#!/bin/bash
#SBATCH -J smoke_kdl50k_forward
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/smoke_kdl50k_forward_%j.out
#SBATCH -e logs/smoke_kdl50k_forward_%j.err

set -euo pipefail

cd /oscar/home/rkapoor8/CS2680/HARP
mkdir -p logs
source harp_env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import torch
from pathlib import Path

from utils.args_parser import parse_args
from frameworks.harp_system import HARP
from utils.training_utils import (
    move_dynamic_sample_to_device,
    run_model_on_dynamic_sample,
    unpack_model_output,
)

DATASET = "dynamic_kdl_hybrid_top5src_global50k_h6_k1_realistic_learned_failure"
fp = Path(DATASET) / "sample_000000.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("sample:", fp)

props = parse_args([
    "--topo", "dynamic_kdl",
    "--mode", "train",
    "--batch_size", "1",
    "--num_paths_per_pair", "1",
    "--num_transformer_layers", "1",
    "--num_gnn_layers", "2",
    "--num_mlp1_hidden_layers", "1",
    "--num_mlp2_hidden_layers", "1",
    "--num_for_loops", "0",
    "--pred", "0",
    "--dynamic", "1",
])

props.device = device
props.dtype = torch.float32
props.return_splits = True
props.return_failure_logits = True
props.num_failure_states = 5
props.num_failure_scenarios = 5

print("loading sample...")
sample = torch.load(fp, map_location="cpu")

print("sample shapes:")
print("node_features:", tuple(sample["node_features"].shape))
print("tm:", tuple(sample["tm"].shape))
print("paths_to_edges:", [tuple(x.shape) for x in sample["paths_to_edges"]])
print("padded:", [tuple(x.shape) for x in sample["padded_edge_ids_per_path"]])
print("future labels:", tuple(sample["future_edge_state_labels"].shape))

print("moving sample to device...")
sample = move_dynamic_sample_to_device(sample, device, props.dtype)

if device.type == "cuda":
    torch.cuda.empty_cache()
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as e:
        print("reset_peak_memory_stats warning:", e)

print("building model...")
model = HARP(props).to(device=device, dtype=props.dtype)
model.eval()

print("running forward...")
with torch.no_grad():
    out = run_model_on_dynamic_sample(model, props, sample)
    predicted, splits, failure_out = unpack_model_output(out)

print("FORWARD PASSED")
print("predicted:", tuple(predicted.shape))
print("splits:", tuple(splits.shape))

if isinstance(failure_out, dict):
    print("scenario_logits:", tuple(failure_out["scenario_logits"].shape))
    print("scenario_edge_logits:", tuple(failure_out["scenario_edge_logits"].shape))
else:
    print("failure_logits:", tuple(failure_out.shape))

if device.type == "cuda":
    print("peak cuda allocated GB:", torch.cuda.max_memory_allocated(device) / 1024**3)
    print("peak cuda reserved GB:", torch.cuda.max_memory_reserved(device) / 1024**3)
PY
