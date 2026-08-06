"""
eval_grate_scenario_coverage.py

Validation-only diagnostic for GRATE topology-mode coverage.

Goal:
  Compare the external Monte Carlo candidate topology distribution against
  GRATE's learned K scenario modes.

For each sample:
  1. Generate external MC candidate topologies with heuristic_candidates.py.
  2. Run GRATE and read failure_output["scenario_edge_logits"].
  3. For each MC candidate, find the closest GRATE scenario.
  4. Count the candidate as covered if the best distance is below threshold.
  5. Report both:
       - conditional covered mass: mass among the selected/renormalized candidates
       - absolute covered mass: raw MC rollout mass, when mc_count/mc_rollouts exists

Important:
  This script is for validation/model-selection only. Do not use final test
  samples to choose K. Do not backpropagate from this diagnostic.
"""

import argparse
import csv
import glob
import math
import re
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils.args_parser import parse_args
from frameworks.harp_system import HARP
from utils.training_utils import (
    move_dynamic_sample_to_device,
    run_model_on_dynamic_sample,
    unpack_model_output,
)

try:
    from utils.training_utils import hydrate_dynamic_sample_static_cache
except ImportError:
    def hydrate_dynamic_sample_static_cache(sample, sample_path=None):
        return sample

from heuristic_candidates import generate_heuristic_candidates


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def sample_number(path):
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else -1


def summarize(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    values_sorted = sorted(values)

    if len(values_sorted) == 0:
        return {
            "avg": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    def pct(p):
        idx = int(len(values_sorted) * p)
        idx = min(idx, len(values_sorted) - 1)
        return values_sorted[idx]

    return {
        "avg": statistics.mean(values_sorted),
        "median": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "min": values_sorted[0],
        "max": values_sorted[-1],
    }


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _forbidden_candidate_keys(sample):
    bad_keys = {
        "failure_candidate_probs",
        "failure_candidate_capacities",
        "failure_candidate_capacity_multipliers",
        "failure_candidate_metadata",
    }
    return sorted(bad_keys & set(sample.keys()))


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_model(checkpoint_path, props):
    obj = torch.load(checkpoint_path, map_location=props.device)

    if hasattr(obj, "eval") and hasattr(obj, "parameters"):
        model = obj
        model = model.to(device=props.device, dtype=props.dtype)
        model.eval()
        return model

    model = HARP(props).to(device=props.device, dtype=props.dtype)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    elif isinstance(obj, dict):
        model.load_state_dict(obj)
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Candidate labels/probability mass helpers
# -----------------------------------------------------------------------------


def default_multiplier_levels(num_classes: int) -> List[float]:
    """
    Label convention from the GRATE failure/degradation generator:
      0 -> 1.00 healthy
      1 -> 0.75 capacity
      2 -> 0.50 capacity
      3 -> 0.25 capacity
      4 -> 0.00 full failure
    """
    if int(num_classes) == 5:
        return [1.0, 0.75, 0.50, 0.25, 0.0]
    if int(num_classes) == 2:
        return [1.0, 0.0]
    return torch.linspace(1.0, 0.0, steps=int(num_classes)).tolist()


def candidate_multipliers_to_labels(candidate_multipliers: torch.Tensor, num_classes: int):
    levels = torch.tensor(
        default_multiplier_levels(num_classes),
        device=candidate_multipliers.device,
        dtype=candidate_multipliers.dtype,
    )
    distances = (candidate_multipliers.unsqueeze(-1) - levels.view(1, 1, -1)).abs()
    return distances.argmin(dim=-1).long()


def get_candidate_name(record: Dict, index: int) -> str:
    for key in ["name", "candidate_name", "type", "candidate_type"]:
        if key in record and record[key] is not None:
            return str(record[key])
    return f"candidate_{index}"


def compute_normalized_and_raw_candidate_probs(candidate_probs, candidate_records):
    """
    Returns:
      normalized_probs: [C] sums to 1 over the selected candidate set.
      raw_probs:        [C] sums to selected MC rollout coverage when mc_count exists.
      selected_coverage:
          If records have mc_count/mc_rollouts, this is selected_counts / rollouts.
          Otherwise fallback is 1.0.
      has_raw_mc_mass: bool
    """
    if not torch.is_tensor(candidate_probs):
        normalized_probs = torch.tensor(candidate_probs, dtype=torch.float64)
    else:
        normalized_probs = candidate_probs.detach().cpu().to(dtype=torch.float64)

    normalized_probs = normalized_probs.reshape(-1)
    normalized_probs = normalized_probs / normalized_probs.sum().clamp_min(1e-12)

    raw_probs = []
    has_counts = True
    total_rollouts = None

    for rec in candidate_records:
        if "mc_count" not in rec:
            has_counts = False
            break

        count = float(rec.get("mc_count", 0.0))

        if "mc_rollouts" in rec:
            rec_rollouts = float(rec.get("mc_rollouts", 0.0))
            if rec_rollouts > 0:
                total_rollouts = rec_rollouts
        elif "num_mc_rollouts" in rec:
            rec_rollouts = float(rec.get("num_mc_rollouts", 0.0))
            if rec_rollouts > 0:
                total_rollouts = rec_rollouts

        raw_probs.append(count)

    if has_counts and total_rollouts is not None and total_rollouts > 0:
        raw_probs = torch.tensor(raw_probs, dtype=torch.float64) / float(total_rollouts)
        selected_coverage = float(raw_probs.sum().item())
        return normalized_probs, raw_probs, selected_coverage, True

    # Fallback for older heuristic candidate generators.
    # In this case we cannot know how much full MC mass the selected candidates cover.
    raw_probs = normalized_probs.clone()
    return normalized_probs, raw_probs, 1.0, False


# -----------------------------------------------------------------------------
# Distance metrics between MC candidates and GRATE K scenarios
# -----------------------------------------------------------------------------


def build_candidate_edge_weights(
    candidate_labels: torch.Tensor,
    state_class_weights: List[float],
    distance_focus: str,
):
    """
    candidate_labels: [C, E]

    distance_focus:
      - state_weighted:
          all edges count, but degraded/failed labels receive higher weight.
      - changed_edges:
          only edges whose candidate label is not healthy count. For the clean
          no-change candidate, fall back to all edges so it can still be covered.
      - all_edges:
          all edges have weight 1.
    """
    C, E = candidate_labels.shape
    device = candidate_labels.device

    if distance_focus == "all_edges":
        return torch.ones(C, E, device=device, dtype=torch.float32)

    weights_by_class = torch.tensor(
        state_class_weights,
        device=device,
        dtype=torch.float32,
    )

    if int(weights_by_class.numel()) <= int(candidate_labels.max().item()):
        raise ValueError(
            f"state_class_weights has length {weights_by_class.numel()}, but labels use "
            f"class {int(candidate_labels.max().item())}."
        )

    weights = weights_by_class[candidate_labels]

    if distance_focus == "state_weighted":
        return weights

    if distance_focus == "changed_edges":
        changed_mask = candidate_labels.ne(0)
        changed_weights = weights * changed_mask.to(dtype=weights.dtype)

        # If a candidate is clean/no-change, there are no changed edges. In that
        # case, evaluate all edges with weight 1 so the healthy scenario can be covered.
        no_changed = changed_weights.sum(dim=1) <= 1e-12
        if no_changed.any():
            changed_weights[no_changed] = 1.0

        return changed_weights

    raise ValueError(
        f"Unknown distance_focus={distance_focus}. Use one of: "
        "state_weighted, changed_edges, all_edges."
    )


def compute_candidate_to_grate_distances(
    scenario_edge_logits: torch.Tensor,
    candidate_multipliers: torch.Tensor,
    state_class_weights: List[float],
    distance_focus: str,
):
    """
    Args:
      scenario_edge_logits: [K, E, S]
      candidate_multipliers: [C, E]

    Returns dict with per-candidate best distances and matched scenario ids.
    """
    if scenario_edge_logits.dim() != 3:
        raise ValueError(
            f"Expected scenario_edge_logits [K,E,S], got {tuple(scenario_edge_logits.shape)}"
        )

    K, E, S = scenario_edge_logits.shape

    candidate_multipliers = candidate_multipliers.to(
        device=scenario_edge_logits.device,
        dtype=scenario_edge_logits.dtype,
    )
    if candidate_multipliers.dim() != 2:
        raise ValueError(
            f"Expected candidate_multipliers [C,E], got {tuple(candidate_multipliers.shape)}"
        )
    if int(candidate_multipliers.shape[1]) != int(E):
        raise ValueError(
            f"Candidate edge dimension {candidate_multipliers.shape[1]} does not match "
            f"GRATE scenario edge dimension {E}."
        )

    candidate_labels = candidate_multipliers_to_labels(candidate_multipliers, int(S))  # [C,E]
    C = candidate_labels.shape[0]

    edge_weights = build_candidate_edge_weights(
        candidate_labels=candidate_labels,
        state_class_weights=state_class_weights,
        distance_focus=distance_focus,
    )  # [C,E]

    denom = edge_weights.sum(dim=1).clamp_min(1e-12)  # [C]

    # Soft distance: weighted negative log probability assigned by scenario k
    # to candidate c's edge-state labels.
    edge_log_probs = torch.log_softmax(scenario_edge_logits.float(), dim=-1)  # [K,E,S]
    expanded_log_probs = edge_log_probs.unsqueeze(0).expand(C, K, E, S)
    expanded_labels = candidate_labels.view(C, 1, E, 1).expand(C, K, E, 1)

    matched_log_probs = expanded_log_probs.gather(
        dim=-1,
        index=expanded_labels,
    ).squeeze(-1)  # [C,K,E]

    per_edge_nll = -matched_log_probs
    weighted_nll = per_edge_nll * edge_weights.view(C, 1, E)
    soft_nll_by_candidate_scenario = weighted_nll.sum(dim=-1) / denom.view(C, 1)  # [C,K]

    best_soft_nll, best_soft_k = soft_nll_by_candidate_scenario.min(dim=1)  # [C]

    # Hard distance: weighted mismatch rate after argmax discretization.
    scenario_argmax = scenario_edge_logits.argmax(dim=-1)  # [K,E]
    mismatches = scenario_argmax.unsqueeze(0).ne(candidate_labels.unsqueeze(1))  # [C,K,E]
    weighted_mismatches = mismatches.to(dtype=torch.float32) * edge_weights.view(C, 1, E)
    hard_mismatch_by_candidate_scenario = weighted_mismatches.sum(dim=-1) / denom.view(C, 1)

    best_hard_mismatch, best_hard_k = hard_mismatch_by_candidate_scenario.min(dim=1)

    return {
        "candidate_labels": candidate_labels.detach().cpu(),
        "best_soft_nll": best_soft_nll.detach().cpu(),
        "best_soft_k": best_soft_k.detach().cpu(),
        "best_hard_mismatch": best_hard_mismatch.detach().cpu(),
        "best_hard_k": best_hard_k.detach().cpu(),
        "soft_nll_by_candidate_scenario": soft_nll_by_candidate_scenario.detach().cpu(),
        "hard_mismatch_by_candidate_scenario": hard_mismatch_by_candidate_scenario.detach().cpu(),
    }


# -----------------------------------------------------------------------------
# Per-sample coverage diagnostic
# -----------------------------------------------------------------------------


def build_candidate_generator_args(args):
    """
    Pass all script args through to heuristic_candidates.py. The candidate
    generator uses getattr(...), so extra fields are harmless and keep this file
    compatible with both the older deterministic heuristic and the newer MC sampler.
    """
    return SimpleNamespace(**vars(args))


def compute_scenario_coverage_for_sample(model, props, sample_cpu, args):
    if args.fail_if_probs_present:
        leaked = _forbidden_candidate_keys(sample_cpu)
        if leaked:
            raise RuntimeError(
                "This diagnostic expects clean learned-failure samples with no saved candidate artifacts. "
                f"Forbidden keys found: {leaked}"
            )

    # 1. Generate the external candidate topology distribution on the fly.
    candidate_args = build_candidate_generator_args(args)
    (
        _candidate_capacities,
        candidate_multipliers,
        candidate_records,
        candidate_probs,
        raw_scores,
    ) = generate_heuristic_candidates(sample_cpu, candidate_args)

    if len(candidate_records) != int(candidate_multipliers.shape[0]):
        raise RuntimeError(
            f"candidate_records length {len(candidate_records)} does not match "
            f"candidate_multipliers rows {candidate_multipliers.shape[0]}."
        )

    normalized_candidate_probs, raw_candidate_probs, selected_coverage, has_raw_mc_mass = (
        compute_normalized_and_raw_candidate_probs(candidate_probs, candidate_records)
    )

    # 2. Run GRATE and get the K scenario logits.
    sample = move_dynamic_sample_to_device(sample_cpu, props.device, props.dtype)

    with torch.no_grad():
        model_output = run_model_on_dynamic_sample(model, props, sample)
        _predicted, _pred_splits, failure_output = unpack_model_output(model_output)

    if not isinstance(failure_output, dict):
        raise RuntimeError(
            "This diagnostic requires the new GRATE K-topology-mode failure head. "
            "Expected failure_output to be a dict with scenario_logits and scenario_edge_logits."
        )

    if "scenario_logits" not in failure_output or "scenario_edge_logits" not in failure_output:
        raise RuntimeError(
            "failure_output dict is missing scenario_logits or scenario_edge_logits."
        )

    scenario_logits = failure_output["scenario_logits"]
    scenario_edge_logits = failure_output["scenario_edge_logits"]

    if scenario_edge_logits.dim() != 4:
        raise RuntimeError(
            f"Expected scenario_edge_logits [B,K,E,S], got {tuple(scenario_edge_logits.shape)}"
        )
    if scenario_edge_logits.shape[0] != 1:
        raise RuntimeError(
            f"This diagnostic currently expects batch size 1, got {tuple(scenario_edge_logits.shape)}"
        )

    scenario_edge_logits_0 = scenario_edge_logits[0]
    scenario_probs = torch.softmax(scenario_logits[0].float(), dim=-1).detach().cpu()

    # 3. Compare every external candidate to every learned GRATE scenario.
    state_class_weights = parse_float_list(args.state_class_weights)
    distances = compute_candidate_to_grate_distances(
        scenario_edge_logits=scenario_edge_logits_0,
        candidate_multipliers=candidate_multipliers,
        state_class_weights=state_class_weights,
        distance_focus=args.distance_focus,
    )

    best_soft_nll = distances["best_soft_nll"].to(dtype=torch.float64)
    best_soft_k = distances["best_soft_k"].long()
    best_hard_mismatch = distances["best_hard_mismatch"].to(dtype=torch.float64)
    best_hard_k = distances["best_hard_k"].long()

    covered_soft = best_soft_nll <= float(args.soft_nll_threshold)
    covered_hard = best_hard_mismatch <= float(args.hard_mismatch_threshold)

    normalized_candidate_probs = normalized_candidate_probs.to(dtype=torch.float64)
    raw_candidate_probs = raw_candidate_probs.to(dtype=torch.float64)

    soft_conditional_covered_mass = float(normalized_candidate_probs[covered_soft].sum().item())
    soft_absolute_covered_mass = float(raw_candidate_probs[covered_soft].sum().item())

    hard_conditional_covered_mass = float(normalized_candidate_probs[covered_hard].sum().item())
    hard_absolute_covered_mass = float(raw_candidate_probs[covered_hard].sum().item())

    soft_expected_best_nll = float((normalized_candidate_probs * best_soft_nll).sum().item())
    hard_expected_best_mismatch = float((normalized_candidate_probs * best_hard_mismatch).sum().item())

    if candidate_multipliers.numel() > 0:
        candidate_labels = distances["candidate_labels"]
        changed_edge_counts = candidate_labels.ne(0).sum(dim=1).to(dtype=torch.float64)
        expected_changed_edges = float((normalized_candidate_probs * changed_edge_counts).sum().item())
    else:
        expected_changed_edges = float("nan")

    top_idx = int(normalized_candidate_probs.argmax().item())
    top_record = candidate_records[top_idx]
    top_candidate_name = get_candidate_name(top_record, top_idx)

    best_soft_k_top = int(best_soft_k[top_idx].item())
    best_hard_k_top = int(best_hard_k[top_idx].item())

    top_raw_score = float(raw_scores[top_idx]) if raw_scores is not None and len(raw_scores) > top_idx else float("nan")

    return {
        "num_candidates": int(candidate_multipliers.shape[0]),
        "num_grate_scenarios": int(scenario_edge_logits_0.shape[0]),
        "num_edges": int(scenario_edge_logits_0.shape[1]),
        "num_failure_states": int(scenario_edge_logits_0.shape[2]),

        # Candidate distribution mass accounting.
        "candidate_probability_sum_normalized": float(normalized_candidate_probs.sum().item()),
        "candidate_probability_sum_raw": float(raw_candidate_probs.sum().item()),
        "selected_mc_coverage": float(selected_coverage),
        "has_raw_mc_mass": int(bool(has_raw_mc_mass)),

        # Main coverage metrics.
        "soft_conditional_covered_mass": soft_conditional_covered_mass,
        "soft_absolute_covered_mass": soft_absolute_covered_mass,
        "hard_conditional_covered_mass": hard_conditional_covered_mass,
        "hard_absolute_covered_mass": hard_absolute_covered_mass,

        # Distance diagnostics.
        "soft_expected_best_nll": soft_expected_best_nll,
        "hard_expected_best_mismatch": hard_expected_best_mismatch,
        "soft_best_nll_min": float(best_soft_nll.min().item()),
        "soft_best_nll_max": float(best_soft_nll.max().item()),
        "hard_best_mismatch_min": float(best_hard_mismatch.min().item()),
        "hard_best_mismatch_max": float(best_hard_mismatch.max().item()),
        "expected_changed_edges_normalized": expected_changed_edges,

        # Top candidate debugging.
        "top_candidate_name": top_candidate_name,
        "top_candidate_probability_normalized": float(normalized_candidate_probs[top_idx].item()),
        "top_candidate_probability_raw": float(raw_candidate_probs[top_idx].item()),
        "top_candidate_score": top_raw_score,
        "top_candidate_best_soft_nll": float(best_soft_nll[top_idx].item()),
        "top_candidate_best_soft_k": best_soft_k_top,
        "top_candidate_best_soft_k_probability": float(scenario_probs[best_soft_k_top].item()),
        "top_candidate_soft_covered": int(bool(covered_soft[top_idx].item())),
        "top_candidate_best_hard_mismatch": float(best_hard_mismatch[top_idx].item()),
        "top_candidate_best_hard_k": best_hard_k_top,
        "top_candidate_best_hard_k_probability": float(scenario_probs[best_hard_k_top].item()),
        "top_candidate_hard_covered": int(bool(covered_hard[top_idx].item())),
    }


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    # Dataset/checkpoint.
    parser.add_argument("--topo", type=str, default="dynamic_abilene")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dynamic_samples_dir", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=700)
    parser.add_argument("--end_idx", type=int, default=850)
    parser.add_argument("--out_csv", type=str, default="grate_scenario_coverage.csv")
    parser.add_argument("--fail_if_probs_present", type=int, default=1)

    # HARP/GRATE architecture args. These must match the checkpoint.
    parser.add_argument("--num_paths_per_pair", type=int, default=4)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_mlp1_hidden_layers", type=int, default=1)
    parser.add_argument("--num_mlp2_hidden_layers", type=int, default=1)
    parser.add_argument("--num_for_loops", type=int, default=1)
    parser.add_argument("--pred", type=int, default=0)
    parser.add_argument("--use_dynamic_opt", type=int, default=0)
    parser.add_argument("--num_failure_states", type=int, default=5)
    parser.add_argument("--num_failure_scenarios", type=int, default=5)

    # Optional memory controls for large topologies. These are manually attached
    # to props because older args_parser.py may not expose them.
    parser.add_argument("--path_chunk_threshold", type=int, default=100000)
    parser.add_argument("--path_chunk_size", type=int, default=50000)
    parser.add_argument("--failure_aggregation_chunk_size", type=int, default=1000000)
    parser.add_argument("--path_chunk_checkpoint", type=int, default=1)

    # Candidate generation controls. The MC heuristic_candidates.py reads these
    # through getattr(...). Older versions will simply ignore unknown fields.
    parser.add_argument("--num_failure_candidates", type=int, default=8)
    parser.add_argument("--candidate_mc_rollouts", type=int, default=512)
    parser.add_argument("--candidate_mc_target_coverage", type=float, default=0.95)
    parser.add_argument("--candidate_mc_prior_strength", type=float, default=20.0)
    parser.add_argument("--candidate_mc_max_impaired_edges", type=int, default=2)
    parser.add_argument("--candidate_mc_correlated_probability", type=float, default=0.10)
    parser.add_argument("--candidate_mc_seed", type=int, default=0)

    # Backward-compatible old heuristic weights. Harmless for the MC sampler.
    parser.add_argument("--candidate_score_floor", type=float, default=1e-6)
    parser.add_argument("--candidate_no_change_weight", type=float, default=1.0)
    parser.add_argument("--candidate_edge_risk_weight", type=float, default=1.0)
    parser.add_argument("--candidate_recent_history_weight", type=float, default=2.0)
    parser.add_argument("--candidate_current_impairment_weight", type=float, default=3.0)
    parser.add_argument("--candidate_flaky_bonus_weight", type=float, default=1.0)
    parser.add_argument("--candidate_worsen_weight", type=float, default=1.25)
    parser.add_argument("--candidate_full_failure_weight", type=float, default=0.70)
    parser.add_argument("--candidate_correlated_weight", type=float, default=0.80)
    parser.add_argument("--candidate_recovery_weight", type=float, default=0.40)

    # Coverage distance controls.
    parser.add_argument(
        "--distance_focus",
        type=str,
        default="state_weighted",
        choices=["state_weighted", "changed_edges", "all_edges"],
        help=(
            "state_weighted = all edges count, but degraded/failed edges get more weight; "
            "changed_edges = focus only on impaired edges, except clean candidate falls back to all edges; "
            "all_edges = plain unweighted edge average."
        ),
    )
    parser.add_argument(
        "--state_class_weights",
        type=str,
        default="1,4,4,4,8",
        help="Comma-separated weights for classes healthy,0.75,0.5,0.25,failed.",
    )
    parser.add_argument("--soft_nll_threshold", type=float, default=0.35)
    parser.add_argument("--hard_mismatch_threshold", type=float, default=0.05)

    args = parser.parse_args()
    args.fail_if_probs_present = bool(args.fail_if_probs_present)
    args.path_chunk_checkpoint = bool(args.path_chunk_checkpoint)

    if args.end_idx <= args.start_idx:
        raise ValueError(f"end_idx must be greater than start_idx, got {args.start_idx}:{args.end_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    props = parse_args([
        "--topo", args.topo,
        "--mode", "test",
        "--batch_size", "1",
        "--num_paths_per_pair", str(args.num_paths_per_pair),
        "--num_transformer_layers", str(args.num_transformer_layers),
        "--num_gnn_layers", str(args.num_gnn_layers),
        "--num_mlp1_hidden_layers", str(args.num_mlp1_hidden_layers),
        "--num_mlp2_hidden_layers", str(args.num_mlp2_hidden_layers),
        "--num_for_loops", str(args.num_for_loops),
        "--framework", "harp",
        "--pred", str(args.pred),
        "--dynamic", "1",
        "--dynamic_samples_dir", args.dynamic_samples_dir,
        "--use_dynamic_opt", str(args.use_dynamic_opt),
    ])

    props.device = device
    props.dtype = torch.float32
    props.return_splits = True
    props.return_failure_logits = True
    props.num_failure_states = int(args.num_failure_states)
    props.num_failure_scenarios = int(args.num_failure_scenarios)

    # Manual large-topology path-chunk settings.
    props.path_chunk_threshold = int(args.path_chunk_threshold)
    props.path_chunk_size = int(args.path_chunk_size)
    props.failure_aggregation_chunk_size = int(args.failure_aggregation_chunk_size)
    props.path_chunk_checkpoint = bool(args.path_chunk_checkpoint)

    print("Device:", device)
    print("Topology:", args.topo)
    print("Checkpoint:", args.checkpoint)
    print("Samples dir:", args.dynamic_samples_dir)
    print("Range:", args.start_idx, args.end_idx)
    print("Diagnostic: GRATE learned scenario coverage over external MC candidates")
    print("Validation/model-selection only: do not tune K on held-out final test data")
    print("Num MC candidates:", args.num_failure_candidates)
    print("MC rollouts:", args.candidate_mc_rollouts)
    print("MC target coverage arg:", args.candidate_mc_target_coverage)
    print("Distance focus:", args.distance_focus)
    print("State class weights:", args.state_class_weights)
    print("Soft NLL threshold:", args.soft_nll_threshold)
    print("Hard mismatch threshold:", args.hard_mismatch_threshold)
    print("Architecture:")
    print("  num_paths_per_pair:", args.num_paths_per_pair)
    print("  num_transformer_layers:", args.num_transformer_layers)
    print("  num_gnn_layers:", args.num_gnn_layers)
    print("  num_mlp1_hidden_layers:", args.num_mlp1_hidden_layers)
    print("  num_mlp2_hidden_layers:", args.num_mlp2_hidden_layers)
    print("  num_for_loops:", args.num_for_loops)
    print("  pred:", args.pred)
    print("  use_dynamic_opt:", args.use_dynamic_opt)
    print("  num_failure_states:", args.num_failure_states)
    print("  num_failure_scenarios:", args.num_failure_scenarios)
    print("Path chunking:")
    print("  path_chunk_threshold:", props.path_chunk_threshold)
    print("  path_chunk_size:", props.path_chunk_size)
    print("  failure_aggregation_chunk_size:", props.failure_aggregation_chunk_size)
    print("  path_chunk_checkpoint:", props.path_chunk_checkpoint)

    model = load_model(args.checkpoint, props)

    files = sorted(glob.glob(f"{args.dynamic_samples_dir}/sample_*.pt"), key=sample_number)
    files = files[args.start_idx:args.end_idx]
    if len(files) == 0:
        raise RuntimeError("No sample files found for requested range.")

    rows = []
    skipped = 0

    for local_i, fp in enumerate(files):
        global_i = args.start_idx + local_i
        try:
            sample = torch.load(fp, map_location="cpu")
            sample = hydrate_dynamic_sample_static_cache(sample, sample_path=fp)

            result = compute_scenario_coverage_for_sample(
                model=model,
                props=props,
                sample_cpu=sample,
                args=args,
            )

            row = {"idx": global_i, "file": fp, **result}
            rows.append(row)

            if local_i % 10 == 0:
                print(
                    f"[{local_i + 1}/{len(files)}] "
                    f"idx={global_i} "
                    f"soft_cond={result['soft_conditional_covered_mass']:.4f} "
                    f"soft_abs={result['soft_absolute_covered_mass']:.4f} "
                    f"hard_cond={result['hard_conditional_covered_mass']:.4f} "
                    f"hard_abs={result['hard_absolute_covered_mass']:.4f} "
                    f"selected_cov={result['selected_mc_coverage']:.4f} "
                    f"top={result['top_candidate_name']} "
                    f"top_soft_nll={result['top_candidate_best_soft_nll']:.4f} "
                    f"top_hard={result['top_candidate_best_hard_mismatch']:.4f}"
                )

        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping {fp}: {type(exc).__name__}: {exc}", flush=True)
            if skipped <= 3:
                import traceback
                traceback.print_exc(limit=3)

    if len(rows) == 0:
        raise RuntimeError("All samples skipped.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved CSV:", out_path)
    print("Samples evaluated:", len(rows))
    print("Samples skipped:", skipped)

    print("\n==== SCENARIO COVERAGE SUMMARY ====")
    skip_summary_keys = {"idx", "file", "top_candidate_name"}
    numeric_keys = [
        key for key in rows[0].keys()
        if key not in skip_summary_keys and isinstance(rows[0][key], (int, float))
    ]

    for key in numeric_keys:
        s = summarize([row[key] for row in rows])
        print(
            f"{key}: "
            f"avg={s['avg']:.6f} "
            f"median={s['median']:.6f} "
            f"p90={s['p90']:.6f} "
            f"p95={s['p95']:.6f} "
            f"min={s['min']:.6f} "
            f"max={s['max']:.6f}"
        )

    print("===================================")

    print("\nMain numbers to look at:")
    for key in [
        "selected_mc_coverage",
        "soft_conditional_covered_mass",
        "soft_absolute_covered_mass",
        "hard_conditional_covered_mass",
        "hard_absolute_covered_mass",
        "soft_expected_best_nll",
        "hard_expected_best_mismatch",
    ]:
        s = summarize([row[key] for row in rows])
        print(f"  {key}: avg={s['avg']:.6f}, median={s['median']:.6f}")


if __name__ == "__main__":
    main()