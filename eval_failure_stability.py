import argparse
import csv
import glob
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    hydrate_dynamic_sample_static_cache = None

from heuristic_candidates import (
    generate_heuristic_candidates,
    get_clean_capacities_from_sample,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _as_float(x, default=float("nan")) -> float:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return float(x.detach().cpu().reshape(-1)[0])
        return float(x)
    except Exception:
        return default


def _as_int(x, default=-1) -> int:
    try:
        if torch.is_tensor(x):
            if x.numel() == 0:
                return default
            return int(x.detach().cpu().reshape(-1)[0])
        return int(x)
    except Exception:
        return default


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.mean(vals) if vals else float("nan")


def _safe_div(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def _get_final_time_item(value):
    """
    Handles the dynamic sample formats we use:
      list length T        -> final item
      capacities [1,T,E]   -> final capacities [1,E]
      otherwise            -> unchanged
    """
    if isinstance(value, list):
        return value[-1]

    if torch.is_tensor(value):
        # compact shared-cache capacities: [1, T, E]
        if value.dim() == 3:
            return value[:, -1, :]
        return value

    return value


def _coalesce_sparse(x):
    if torch.is_tensor(x) and x.is_sparse:
        return x.coalesce()
    return x


def sample_number(path) -> int:
    stem = Path(path).stem
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else -1


def summarize(values):
    values = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
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


# -----------------------------------------------------------------------------
# Core MLU utilities
# -----------------------------------------------------------------------------


def compute_mlu_from_splits(split_ratios, tm_final, paths_to_edges, capacities, eps=1e-9):
    """
    split_ratios: [1, P]
    tm_final: [1, P, 1]
    paths_to_edges: sparse [P, E]
    capacities: [1, E] or [E]

    Returns:
        mlu scalar tensor
        util tensor [1, E]
        data_on_links tensor [1, E]
    """
    if split_ratios is None:
        raise RuntimeError("compute_mlu_from_splits requires split_ratios")

    if split_ratios.dim() != 2:
        raise RuntimeError(f"Expected split_ratios [B,P], got {tuple(split_ratios.shape)}")

    if tm_final.dim() == 2:
        tm_final = tm_final.unsqueeze(0)
    if tm_final.dim() != 3:
        raise RuntimeError(f"Expected tm_final [B,P,1], got {tuple(tm_final.shape)}")

    if capacities.dim() == 1:
        capacities = capacities.unsqueeze(0)

    paths_to_edges = paths_to_edges.coalesce().to(device=split_ratios.device, dtype=torch.float32)
    capacities = capacities.to(device=split_ratios.device, dtype=split_ratios.dtype)
    tm_final = tm_final.to(device=split_ratios.device, dtype=split_ratios.dtype)

    if capacities.shape[0] == 1 and split_ratios.shape[0] > 1:
        capacities = capacities.expand(split_ratios.shape[0], -1)

    data_on_tunnels = split_ratios * tm_final.squeeze(-1)  # [B, P]

    data_on_links = torch.sparse.mm(
        paths_to_edges.t(),
        data_on_tunnels.to(dtype=torch.float32).t(),
    ).t()  # [B, E]

    data_on_links = data_on_links.to(dtype=split_ratios.dtype)
    util = data_on_links / capacities.clamp_min(eps)
    util = torch.nan_to_num(util, nan=0.0, posinf=1e6, neginf=0.0)

    return util.max(), util, data_on_links


def compute_candidate_mlu_with_local_rescale(
    split_ratios,
    tm_final,
    paths_to_edges,
    candidate_capacities,
    num_paths_per_pair,
    disconnected_mlu_penalty=10.0,
    disconnected_penalty_mode="fraction",
    eps=1e-9,
):
    """
    Evaluates one fixed routing decision under one candidate capacity state.

    Local rescaling:
      - paths using any zero-capacity edge are removed
      - remaining split mass is renormalized within each SD pair
      - partial degradations stay alive but lower capacity

    disconnected_penalty_mode:
      - "none": no extra penalty
      - "fraction": mlu += penalty * disconnected_demand_fraction
      - "any": if any SD pair disconnects, candidate MLU = penalty
    """
    if split_ratios.dim() != 2 or split_ratios.shape[0] != 1:
        raise ValueError(f"Expected split_ratios [1, P], got {tuple(split_ratios.shape)}")

    paths_to_edges = paths_to_edges.coalesce().to(device=split_ratios.device)
    candidate_capacities = candidate_capacities.to(device=split_ratios.device, dtype=split_ratios.dtype)

    if candidate_capacities.dim() == 1:
        candidate_capacities = candidate_capacities.unsqueeze(0)

    B, P = split_ratios.shape
    K = int(num_paths_per_pair)

    if K <= 0:
        raise ValueError(f"num_paths_per_pair must be positive, got {K}")
    if P % K != 0:
        raise ValueError(f"P={P} is not divisible by K={K}")

    num_pairs = P // K

    inactive_edges = (candidate_capacities <= eps).to(dtype=torch.float32)  # [1,E]

    # sparse [P,E] @ dense [E,1] -> dense [P,1] -> [1,P]
    blocked_counts = torch.sparse.mm(
        paths_to_edges.to(dtype=torch.float32),
        inactive_edges.t(),
    ).t()

    path_alive = (blocked_counts <= 0.0).to(dtype=split_ratios.dtype)

    splits_by_pair = split_ratios.reshape(B, num_pairs, K)
    alive_by_pair = path_alive.reshape(B, num_pairs, K)

    masked_splits = splits_by_pair * alive_by_pair
    surviving_mass = masked_splits.sum(dim=-1, keepdim=True)

    disconnected_pairs_mask = surviving_mass.squeeze(-1) <= eps
    disconnected_pairs = int(disconnected_pairs_mask.sum().detach().cpu())

    effective_splits_by_pair = torch.where(
        surviving_mass > eps,
        masked_splits / surviving_mass.clamp_min(eps),
        torch.zeros_like(masked_splits),
    )
    effective_splits = effective_splits_by_pair.reshape(B, P)

    tm_final = tm_final.to(device=split_ratios.device, dtype=split_ratios.dtype)
    tm_by_pair = tm_final.squeeze(-1).reshape(B, num_pairs, K)[:, :, 0]

    total_demand = tm_by_pair.sum(dim=-1).clamp_min(eps)
    disconnected_demand = (tm_by_pair * disconnected_pairs_mask.to(tm_by_pair.dtype)).sum(dim=-1)
    disconnected_fraction = float((disconnected_demand / total_demand).mean().detach().cpu())

    uses_failed = path_alive.reshape(P) <= 0.0
    affected_pairs = int(uses_failed.reshape(num_pairs, K).any(dim=-1).sum().detach().cpu())
    failed_path_fraction = float(uses_failed.float().mean().detach().cpu())

    mlu, _, _ = compute_mlu_from_splits(
        split_ratios=effective_splits,
        tm_final=tm_final,
        paths_to_edges=paths_to_edges,
        capacities=candidate_capacities,
        eps=eps,
    )

    mlu_value = float(mlu.detach().cpu())

    if disconnected_pairs > 0:
        if disconnected_penalty_mode == "any":
            mlu_value = float(disconnected_mlu_penalty)
        elif disconnected_penalty_mode == "fraction":
            mlu_value = mlu_value + float(disconnected_mlu_penalty) * disconnected_fraction
        elif disconnected_penalty_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown disconnected_penalty_mode={disconnected_penalty_mode}")

    return {
        "candidate_mlu": mlu_value,
        "disconnected_pairs": disconnected_pairs,
        "disconnected_fraction": disconnected_fraction,
        "affected_pairs": affected_pairs,
        "failed_path_fraction": failed_path_fraction,
    }


# -----------------------------------------------------------------------------
# Failure prediction diagnostics
# -----------------------------------------------------------------------------


def _failure_prediction_metrics(failure_logits, sample):
    """
    Computes diagnostic edge-state prediction loss/accuracy.

    Supports:
      logits tensor [B,E,S]
      logits tensor [E,S]
      scenario logits tensor [B,K,E,S]
      dict outputs containing one of:
        scenario_edge_logits
        failure_logits
        edge_logits
        future_edge_logits

    For scenario logits, this uses winner-take-all: choose the scenario whose
    per-edge CE is smallest against the realized future labels.
    """

    labels = sample.get("future_edge_state_labels", None)
    if labels is None or failure_logits is None:
        return float("nan"), float("nan")

    labels = labels.long().reshape(-1)
    num_edges = int(labels.numel())

    def extract_logits(obj):
        """
        Recursively find the actual tensor logits inside tensor/dict/tuple outputs.
        Prefer scenario_edge_logits when present.
        """
        if torch.is_tensor(obj):
            return obj

        if isinstance(obj, dict):
            preferred_keys = [
                "scenario_edge_logits",
                "failure_logits",
                "edge_logits",
                "future_edge_logits",
                "logits",
            ]

            for key in preferred_keys:
                if key in obj:
                    found = extract_logits(obj[key])
                    if found is not None:
                        return found

            # Fallback: recursively search all dict values.
            for value in obj.values():
                found = extract_logits(value)
                if found is not None:
                    return found

            return None

        if isinstance(obj, (list, tuple)):
            for value in obj:
                found = extract_logits(value)
                if found is not None:
                    return found

        return None

    logits = extract_logits(failure_logits)

    if logits is None or not torch.is_tensor(logits):
        return float("nan"), float("nan")

    labels = labels.to(device=logits.device)

    if logits.dim() == 2:
        # [E,S]
        if logits.shape[0] != num_edges:
            return float("nan"), float("nan")

        loss = F.cross_entropy(logits.float(), labels, reduction="mean")
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        return float(loss.detach().cpu()), float(acc.detach().cpu())

    if logits.dim() == 3:
        # [B,E,S]
        if logits.shape[0] != 1 or logits.shape[1] != num_edges:
            return float("nan"), float("nan")

        logits0 = logits[0]
        loss = F.cross_entropy(logits0.float(), labels, reduction="mean")
        pred = logits0.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        return float(loss.detach().cpu()), float(acc.detach().cpu())

    if logits.dim() == 4:
        # [B,K,E,S]
        if logits.shape[0] != 1 or logits.shape[2] != num_edges:
            return float("nan"), float("nan")

        scenario_logits = logits[0]  # [K,E,S]
        K = scenario_logits.shape[0]

        losses = []
        for k in range(K):
            losses.append(
                F.cross_entropy(
                    scenario_logits[k].float(),
                    labels,
                    reduction="mean",
                )
            )

        loss_tensor = torch.stack(losses)
        winner = int(loss_tensor.argmin().detach().cpu())

        best_logits = scenario_logits[winner]
        pred = best_logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()

        return float(loss_tensor[winner].detach().cpu()), float(acc.detach().cpu())

    return float("nan"), float("nan")


# -----------------------------------------------------------------------------
# Per-sample evaluator
# -----------------------------------------------------------------------------


def compute_stability_for_sample(
    model,
    props,
    sample,
    args,
    sample_path=None,
):
    if hydrate_dynamic_sample_static_cache is not None:
        sample = hydrate_dynamic_sample_static_cache(sample, sample_path=sample_path)

    sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

    if args.fail_if_probs_present and "failure_candidate_probs" in sample:
        raise RuntimeError(
            "Sample contains saved failure_candidate_probs, but this evaluator expects "
            "probabilities to be generated externally at evaluation time."
        )

    with torch.no_grad():
        model_output = run_model_on_dynamic_sample(model, props, sample)
        predicted, pred_splits, failure_logits = unpack_model_output(model_output)

    if pred_splits is None:
        raise RuntimeError("Model did not return split ratios. Set props.return_splits=True.")

    if not torch.isfinite(pred_splits).all():
        raise RuntimeError("Non-finite predicted splits.")

    tm_final = sample["tm"][:, -1]
    pte_final = _coalesce_sparse(_get_final_time_item(sample["paths_to_edges"]))
    current_caps = _get_final_time_item(sample["capacities"])
    future_caps = sample.get("future_capacities", None)

    clean_caps = get_clean_capacities_from_sample(sample).to(
        device=pred_splits.device,
        dtype=pred_splits.dtype,
    )

    current_caps = current_caps.to(device=pred_splits.device, dtype=pred_splits.dtype)
    if future_caps is None:
        raise RuntimeError("Sample is missing future_capacities.")
    future_caps = future_caps.to(device=pred_splits.device, dtype=pred_splits.dtype)

    # Clean/perfect topology baseline. This is the main numerator.
    clean_mlu, _, _ = compute_mlu_from_splits(
        split_ratios=pred_splits,
        tm_final=tm_final,
        paths_to_edges=pte_final,
        capacities=clean_caps,
    )
    clean_mlu_value = float(clean_mlu.detach().cpu())

    # Diagnostic only: actual observed topology at time t.
    current_mlu, _, _ = compute_mlu_from_splits(
        split_ratios=pred_splits,
        tm_final=tm_final,
        paths_to_edges=pte_final,
        capacities=current_caps,
    )
    current_mlu_value = float(current_mlu.detach().cpu())

    # Realized next-step future from the generator.
    realized_result = compute_candidate_mlu_with_local_rescale(
        split_ratios=pred_splits,
        tm_final=tm_final,
        paths_to_edges=pte_final,
        candidate_capacities=future_caps,
        num_paths_per_pair=args.num_paths_per_pair,
        disconnected_mlu_penalty=args.disconnected_mlu_penalty,
        disconnected_penalty_mode=args.disconnected_penalty_mode,
    )
    realized_future_mlu = float(realized_result["candidate_mlu"])
    realized_future_disconnected_fraction = float(realized_result["disconnected_fraction"])

    realized_future_stability_ratio = _safe_div(clean_mlu_value, realized_future_mlu)
    bounded_realized_future_stability_score = _safe_div(
        clean_mlu_value,
        max(realized_future_mlu, clean_mlu_value),
    )

    # External MC candidate distribution from heuristic_candidates.py.
    (
        candidate_capacities,
        candidate_multipliers,
        candidate_records,
        candidate_probs,
        raw_scores,
    ) = generate_heuristic_candidates(sample, args)

    if candidate_capacities.dim() != 2:
        raise RuntimeError(
            f"candidate_capacities should be [C,E], got {tuple(candidate_capacities.shape)}"
        )

    candidate_capacities = candidate_capacities.to(device=pred_splits.device, dtype=pred_splits.dtype)
    candidate_probs = candidate_probs.detach().cpu().double()

    if candidate_probs.numel() != candidate_capacities.shape[0]:
        raise RuntimeError(
            f"candidate_probs length {candidate_probs.numel()} does not match "
            f"candidate_capacities rows {candidate_capacities.shape[0]}"
        )

    if len(candidate_records) != candidate_capacities.shape[0]:
        raise RuntimeError(
            f"candidate_records length {len(candidate_records)} does not match "
            f"candidate_capacities rows {candidate_capacities.shape[0]}"
        )

    probability_sum = float(candidate_probs.sum().item())
    if probability_sum <= 0.0:
        raise RuntimeError("Candidate probabilities sum to zero.")

    candidate_probs = candidate_probs / candidate_probs.sum().clamp_min(1e-12)

    candidate_mlus = []
    clipped_candidate_mlus = []
    included_probs = []
    affected_pairs_list = []
    failed_path_fraction_list = []
    disconnected_fraction_list = []

    disconnected_candidate_count = 0
    penalized_disconnected_candidate_count = 0
    skipped_disconnected_candidate_count = 0

    for i in range(candidate_capacities.shape[0]):
        cand_result = compute_candidate_mlu_with_local_rescale(
            split_ratios=pred_splits,
            tm_final=tm_final,
            paths_to_edges=pte_final,
            candidate_capacities=candidate_capacities[i],
            num_paths_per_pair=args.num_paths_per_pair,
            disconnected_mlu_penalty=args.disconnected_mlu_penalty,
            disconnected_penalty_mode=args.disconnected_penalty_mode,
        )

        cand_mlu = float(cand_result["candidate_mlu"])
        cand_prob = float(candidate_probs[i].item())

        is_disconnected = int(cand_result["disconnected_pairs"]) > 0
        if is_disconnected:
            disconnected_candidate_count += 1

        if is_disconnected and args.skip_disconnected_candidates:
            skipped_disconnected_candidate_count += 1
            continue

        if is_disconnected:
            penalized_disconnected_candidate_count += 1

        candidate_mlus.append(cand_mlu)
        clipped_candidate_mlus.append(max(cand_mlu, clean_mlu_value))
        included_probs.append(cand_prob)

        affected_pairs_list.append(float(cand_result["affected_pairs"]))
        failed_path_fraction_list.append(float(cand_result["failed_path_fraction"]))
        disconnected_fraction_list.append(float(cand_result["disconnected_fraction"]))

    probability_mass_used = float(sum(included_probs))
    if len(candidate_mlus) == 0 or probability_mass_used <= 1e-12:
        expected_candidate_mlu = float("nan")
        bounded_expected_candidate_mlu = float("nan")
    else:
        norm_probs = [p / probability_mass_used for p in included_probs]
        expected_candidate_mlu = sum(p * m for p, m in zip(norm_probs, candidate_mlus))
        bounded_expected_candidate_mlu = sum(p * m for p, m in zip(norm_probs, clipped_candidate_mlus))

    candidate_stability_ratio = _safe_div(clean_mlu_value, expected_candidate_mlu)
    bounded_candidate_stability_score = _safe_div(clean_mlu_value, bounded_expected_candidate_mlu)

    top_idx = int(candidate_probs.argmax().item()) if candidate_probs.numel() > 0 else -1
    top_candidate_probability = float(candidate_probs[top_idx].item()) if top_idx >= 0 else float("nan")
    top_candidate_score = float(raw_scores[top_idx]) if top_idx >= 0 and top_idx < len(raw_scores) else float("nan")
    top_candidate_name = (
        str(candidate_records[top_idx].get("name", f"candidate_{top_idx}"))
        if top_idx >= 0 and top_idx < len(candidate_records)
        else ""
    )

    # These MC fields are stored redundantly in each record by your updated
    # heuristic_candidates.py. Use record 0 as the distribution-level metadata.
    meta_rec = candidate_records[0] if candidate_records else {}

    mc_rollouts = _as_int(meta_rec.get("mc_rollouts", -1))
    mc_unique_topologies = _as_int(meta_rec.get("mc_unique_topologies", -1))
    mc_selected_candidates = _as_int(
        meta_rec.get("mc_selected_candidates", candidate_capacities.shape[0])
    )
    mc_selected_total = _as_int(meta_rec.get("mc_selected_total", -1))
    mc_selected_mass = _as_float(meta_rec.get("mc_selected_mass", float("nan")))
    mc_dropped_mass = _as_float(meta_rec.get("mc_dropped_mass", float("nan")))
    mc_target_coverage = _as_float(meta_rec.get("mc_target_coverage", float("nan")))
    mc_target_reached = int(bool(meta_rec.get("mc_target_reached", False)))
    mc_max_candidates = _as_int(meta_rec.get("mc_max_candidates", -1))
    mc_prior_strength = _as_float(meta_rec.get("mc_prior_strength", float("nan")))
    mc_max_impaired_edges = _as_int(meta_rec.get("mc_max_impaired_edges", -1))
    mc_correlated_probability = _as_float(meta_rec.get("mc_correlated_probability", float("nan")))

    top_candidate_sampled_probability = (
        _as_float(candidate_records[top_idx].get("mc_sampled_probability", float("nan")))
        if top_idx >= 0 and top_idx < len(candidate_records)
        else float("nan")
    )
    top_candidate_cumulative_sampled_mass = (
        _as_float(candidate_records[top_idx].get("mc_cumulative_sampled_mass", float("nan")))
        if top_idx >= 0 and top_idx < len(candidate_records)
        else float("nan")
    )

    failure_pred_loss, failure_pred_acc = _failure_prediction_metrics(failure_logits, sample)

    return {
        # Old/compatibility naming.
        "norm_mlu": float("nan"),
        "normal_mlu": clean_mlu_value,

        # Main topology baselines.
        "clean_mlu": clean_mlu_value,
        "current_mlu": current_mlu_value,
        "current_over_clean_mlu_ratio": _safe_div(current_mlu_value, clean_mlu_value),

        # Realized next-step future.
        "realized_future_mlu": realized_future_mlu,
        "realized_future_stability_ratio": realized_future_stability_ratio,
        "bounded_realized_future_stability_score": bounded_realized_future_stability_score,
        "realized_future_disconnected_fraction": realized_future_disconnected_fraction,

        # Monte Carlo candidate distribution.
        "expected_candidate_mlu": expected_candidate_mlu,
        "bounded_expected_candidate_mlu": bounded_expected_candidate_mlu,
        "candidate_stability_ratio": candidate_stability_ratio,
        "bounded_candidate_stability_score": bounded_candidate_stability_score,
        "avg_candidate_mlu_unweighted": _mean(candidate_mlus),
        "avg_clipped_candidate_mlu_unweighted": _mean(clipped_candidate_mlus),
        "max_candidate_mlu": max(candidate_mlus) if candidate_mlus else float("nan"),
        "min_candidate_mlu": min(candidate_mlus) if candidate_mlus else float("nan"),

        # Probability accounting.
        "top_candidate_probability": top_candidate_probability,
        "top_candidate_sampled_probability": top_candidate_sampled_probability,
        "top_candidate_cumulative_sampled_mass": top_candidate_cumulative_sampled_mass,
        "top_candidate_score": top_candidate_score,
        "top_candidate_name": top_candidate_name,
        "probability_mass_used": probability_mass_used,
        "candidate_probability_sum": probability_sum,

        # Candidate counts/filtering.
        "num_candidates": int(candidate_capacities.shape[0]),
        "included_candidate_count": int(len(candidate_mlus)),
        "disconnected_candidate_count": int(disconnected_candidate_count),
        "penalized_disconnected_candidate_count": int(penalized_disconnected_candidate_count),
        "skipped_disconnected_candidate_count": int(skipped_disconnected_candidate_count),

        # Candidate impact diagnostics.
        "avg_affected_pairs": _mean(affected_pairs_list),
        "avg_failed_path_fraction": _mean(failed_path_fraction_list),
        "avg_disconnected_fraction": _mean(disconnected_fraction_list),

        # MC coverage diagnostics.
        "mc_rollouts": mc_rollouts,
        "mc_unique_topologies": mc_unique_topologies,
        "mc_selected_candidates": mc_selected_candidates,
        "mc_selected_total": mc_selected_total,
        "mc_selected_mass": mc_selected_mass,
        "mc_dropped_mass": mc_dropped_mass,
        "mc_target_coverage": mc_target_coverage,
        "mc_target_reached": mc_target_reached,
        "mc_max_candidates": mc_max_candidates,
        "mc_prior_strength": mc_prior_strength,
        "mc_max_impaired_edges": mc_max_impaired_edges,
        "mc_correlated_probability": mc_correlated_probability,

        # GRATE failure/scenario head diagnostics.
        "failure_prediction_loss": failure_pred_loss,
        "failure_prediction_accuracy": failure_pred_acc,
    }


# -----------------------------------------------------------------------------
# Model loading / props
# -----------------------------------------------------------------------------


def _parse_harp_props(args, device):
    """
    Build the HARP props object using the repo's args_parser, then attach eval-only
    and dynamic flags. This mirrors the dynamic training/eval launch style.
    """
    base_argv = [
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
    ]

    try:
        props = parse_args(base_argv)
    except SystemExit:
        # Some older args_parser.py versions do not know the dynamic flags.
        fallback_argv = [
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
        ]
        props = parse_args(fallback_argv)

    props.device = device
    props.dtype = torch.float32

    props.dynamic = True
    props.dynamic_samples_dir = args.dynamic_samples_dir
    props.use_dynamic_opt = bool(args.use_dynamic_opt)

    props.return_splits = True
    props.return_failure_logits = True

    props.num_paths_per_pair = int(args.num_paths_per_pair)
    props.num_failure_states = int(args.num_failure_states)
    props.num_failure_scenarios = int(args.num_failure_scenarios)

    # Optional full-KDL/chunking knobs. Harmless for Abilene/GEANT.
    props.path_chunk_threshold = int(args.path_chunk_threshold)
    props.path_chunk_size = int(args.path_chunk_size)
    props.path_chunk_checkpoint = bool(args.path_chunk_checkpoint)
    props.failure_aggregation_chunk_size = int(args.failure_aggregation_chunk_size)

    return props


def _strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_model(checkpoint_path, props):
    obj = torch.load(checkpoint_path, map_location=props.device)

    if hasattr(obj, "eval") and hasattr(obj, "parameters"):
        model = obj
        model = model.to(device=props.device, dtype=props.dtype)
        model.eval()
        return model

    model = HARP(props).to(device=props.device, dtype=props.dtype)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(obj)}")

    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(_strip_module_prefix(state))

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--topo", type=str, default="dynamic_abilene")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dynamic_samples_dir", type=str, required=True)
    parser.add_argument("--start_idx", type=int, default=800)
    parser.add_argument("--end_idx", type=int, default=1000)
    parser.add_argument("--out_csv", type=str, default="failure_stability_results.csv")

    # Fair final evaluation should use heuristic/external distribution only.
    parser.add_argument("--prob_source", type=str, default="heuristic", choices=["heuristic", "model"])
    parser.add_argument("--model_prob_temperature", type=float, default=1.0)
    parser.add_argument("--model_prob_edge_aggregation", type=str, default="sum", choices=["sum", "mean"])

    # Backward-compatible candidate args.
    # num_failure_candidates is no longer the true selector when using the new
    # MC coverage heuristic, but it is kept so old SLURM scripts do not break.
    parser.add_argument("--num_failure_candidates", type=int, default=8)
    parser.add_argument("--skip_disconnected_candidates", type=int, default=0)
    parser.add_argument("--disconnected_mlu_penalty", type=float, default=10.0)
    parser.add_argument("--disconnected_penalty_mode", type=str, default="fraction", choices=["none", "fraction", "any"])
    parser.add_argument("--fail_if_probs_present", type=int, default=1)

    # Monte Carlo topology coverage args. These are read by heuristic_candidates.py.
    parser.add_argument("--candidate_mc_rollouts", type=int, default=512)
    parser.add_argument("--candidate_mc_target_coverage", type=float, default=0.95)
    parser.add_argument("--candidate_mc_max_candidates", type=int, default=32)
    parser.add_argument("--candidate_mc_prior_strength", type=float, default=20.0)
    parser.add_argument("--candidate_mc_max_impaired_edges", type=int, default=2)
    parser.add_argument("--candidate_mc_correlated_probability", type=float, default=0.10)
    parser.add_argument("--candidate_mc_correlated_max_edges", type=int, default=2)
    parser.add_argument("--candidate_mc_seed", type=int, default=2680)

    # Architecture config: must match checkpoint.
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

    # Optional chunking knobs for large topologies.
    parser.add_argument("--path_chunk_threshold", type=int, default=100000)
    parser.add_argument("--path_chunk_size", type=int, default=50000)
    parser.add_argument("--path_chunk_checkpoint", type=int, default=1)
    parser.add_argument("--failure_aggregation_chunk_size", type=int, default=1000000)

    # Legacy deterministic candidate weights. These do not drive the new MC
    # coverage selector, but keep old scripts compatible.
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

    args = parser.parse_args()

    args.skip_disconnected_candidates = bool(args.skip_disconnected_candidates)
    args.fail_if_probs_present = bool(args.fail_if_probs_present)
    args.path_chunk_checkpoint = bool(args.path_chunk_checkpoint)

    if args.end_idx <= args.start_idx:
        raise ValueError(f"end_idx must be greater than start_idx, got {args.start_idx}:{args.end_idx}")

    if args.prob_source != "heuristic":
        raise RuntimeError(
            "This clean/fair evaluator only supports --prob_source heuristic. "
            "GRATE logits can guide GRATE routing/training, but final scoring should use "
            "the same external distribution for every model."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = _parse_harp_props(args, device)

    print("Device:", device)
    print("Topology:", args.topo)
    print("Checkpoint:", args.checkpoint)
    print("Samples dir:", args.dynamic_samples_dir)
    print("Range:", args.start_idx, args.end_idx)
    print("Evaluation mode: clean learned-failure samples + clean-topology MC candidates")
    print("Baseline topology: clean/nominal/perfect")
    print("Probability source: heuristic/external Monte Carlo only")
    print("Legacy num_failure_candidates arg:", args.num_failure_candidates)
    print("MC rollouts:", args.candidate_mc_rollouts)
    print("MC target coverage:", args.candidate_mc_target_coverage)
    print("MC max candidates:", args.candidate_mc_max_candidates)
    print("Skip disconnected candidates:", args.skip_disconnected_candidates)
    print("Disconnected MLU penalty:", args.disconnected_mlu_penalty)
    print("Disconnected penalty mode:", args.disconnected_penalty_mode)
    print("Fail if saved candidate artifacts are present:", args.fail_if_probs_present)
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

    model = load_model(args.checkpoint, props)

    sample_files = sorted(
        glob.glob(str(Path(args.dynamic_samples_dir) / "sample_*.pt")),
        key=sample_number,
    )
    sample_files = [
        fp for fp in sample_files
        if args.start_idx <= sample_number(fp) < args.end_idx
    ]

    if not sample_files:
        raise RuntimeError(
            f"No sample_*.pt files found in {args.dynamic_samples_dir} "
            f"for range [{args.start_idx}, {args.end_idx})."
        )

    rows = []
    skipped = 0

    for local_i, fp in enumerate(sample_files):
        idx = sample_number(fp)

        try:
            sample = torch.load(fp, map_location="cpu")
            result = compute_stability_for_sample(
                model=model,
                props=props,
                sample=sample,
                args=args,
                sample_path=fp,
            )

            row = {
                "idx": idx,
                "file": fp,
                **result,
            }
            rows.append(row)

            if local_i % 10 == 0 or local_i == len(sample_files) - 1:
                mc_mass = result.get("mc_selected_mass", float("nan"))
                mc_drop = result.get("mc_dropped_mass", float("nan"))
                mc_unique = result.get("mc_unique_topologies", -1)
                mc_reached = result.get("mc_target_reached", 0)

                print(
                    f"[{local_i + 1}/{len(sample_files)}] "
                    f"idx={idx} "
                    f"clean={result['clean_mlu']:.4f} "
                    f"current={result['current_mlu']:.4f} "
                    f"realized={result['realized_future_mlu']:.4f} "
                    f"expected_candidate={result['expected_candidate_mlu']:.4f} "
                    f"candidate_ratio={result['candidate_stability_ratio']:.6f} "
                    f"bounded_score={result['bounded_candidate_stability_score']:.6f} "
                    f"candidates={result['included_candidate_count']}/{result['num_candidates']} "
                    f"mc_mass={mc_mass:.4f} "
                    f"mc_drop={mc_drop:.4f} "
                    f"mc_unique={mc_unique} "
                    f"mc_reached={mc_reached} "
                    f"prob_mass={result['probability_mass_used']:.6f} "
                    f"top={result['top_candidate_name']}:{result['top_candidate_probability']:.4f}"
                )

        except RuntimeError as exc:
            skipped += 1
            print(f"[WARN] idx={idx} failed: {exc}")
            continue

    if not rows:
        raise RuntimeError("No samples were successfully evaluated.")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print()
    print("Saved CSV:", out_path)
    print("Skipped:", skipped)
    print()

    summary_metrics = [
        "norm_mlu",
        "normal_mlu",
        "clean_mlu",
        "current_mlu",
        "current_over_clean_mlu_ratio",
        "realized_future_mlu",
        "realized_future_stability_ratio",
        "bounded_realized_future_stability_score",
        "realized_future_disconnected_fraction",
        "expected_candidate_mlu",
        "bounded_expected_candidate_mlu",
        "candidate_stability_ratio",
        "bounded_candidate_stability_score",
        "avg_candidate_mlu_unweighted",
        "avg_clipped_candidate_mlu_unweighted",
        "max_candidate_mlu",
        "min_candidate_mlu",
        "top_candidate_probability",
        "top_candidate_sampled_probability",
        "top_candidate_cumulative_sampled_mass",
        "top_candidate_score",
        "avg_affected_pairs",
        "avg_failed_path_fraction",
        "avg_disconnected_fraction",
        "probability_mass_used",
        "candidate_probability_sum",
        "num_candidates",
        "included_candidate_count",
        "disconnected_candidate_count",
        "penalized_disconnected_candidate_count",
        "skipped_disconnected_candidate_count",
        "mc_rollouts",
        "mc_unique_topologies",
        "mc_selected_candidates",
        "mc_selected_total",
        "mc_selected_mass",
        "mc_dropped_mass",
        "mc_target_coverage",
        "mc_target_reached",
        "mc_max_candidates",
        "mc_prior_strength",
        "mc_max_impaired_edges",
        "mc_correlated_probability",
        "failure_prediction_loss",
        "failure_prediction_accuracy",
    ]

    for key in summary_metrics:
        vals = [row.get(key, float("nan")) for row in rows]
        s = summarize(vals)
        print(key)
        print(f"  avg: {s['avg']:.6f}")
        print(f"  median: {s['median']:.6f}")
        print(f"  p90: {s['p90']:.6f}")
        print(f"  p95: {s['p95']:.6f}")
        print(f"  min: {s['min']:.6f}")
        print(f"  max: {s['max']:.6f}")
        print()


if __name__ == "__main__":
    main()
