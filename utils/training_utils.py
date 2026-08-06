import csv
import glob
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

def move_to_device(dictionary, device="cpu"):
    for key in dictionary.keys():
        dictionary[key] = dictionary[key].to(device)
    return dictionary


def loss_mlu(y_pred_batch, y_true_batch):
    losses = []
    loss_vals = []
    batch_size = y_pred_batch.shape[0]

    for i in range(batch_size):
        y_pred = y_pred_batch[[i]]
        opt = y_true_batch[[i]]
        max_cong = torch.max(y_pred)

        loss = 1.0 - max_cong if max_cong.item() == 0.0 else max_cong / max_cong.item()
        loss_val = 1.0 if opt == 0.0 else max_cong.item() / opt.item()

        losses.append(loss)
        loss_vals.append(loss_val)

    ret = sum(losses) / len(losses)
    ret_val = sum(loss_vals) / len(loss_vals)
    return ret, ret_val


def unpack_model_output(model_output):
    """
    Dynamic HARP can return either:
        edges_util
        (edges_util, split_ratios)
        (edges_util, split_ratios, failure_logits)

    failure_logits are next-step edge-state predictions aligned with the
    final/current directed edge list: [B, E_final, num_failure_states].
    """
    if isinstance(model_output, tuple):
        if len(model_output) == 2:
            return model_output[0], model_output[1], None
        if len(model_output) == 3:
            return model_output[0], model_output[1], model_output[2]
        raise ValueError(f"Unexpected model output tuple length: {len(model_output)}")
    return model_output, None, None



def _as_batched_capacities(capacities, batch_size, device, dtype):
    """
    Converts capacities into [B, E].
    """
    if capacities.dim() == 1:
        capacities = capacities.unsqueeze(0)

    capacities = capacities.to(device=device, dtype=dtype)

    if capacities.shape[0] == 1 and batch_size > 1:
        capacities = capacities.expand(batch_size, -1)

    return capacities


def _compute_mlu_from_splits_under_capacities(
    split_ratios,
    tm_final,
    paths_to_edges,
    capacities,
    num_paths_per_pair,
    local_rescale=True,
    disconnected_penalty=100.0,
    eps=1e-8,
):
    """
    Evaluate one routing decision under a specified capacity vector.

    Args:
        split_ratios:
            [B, P] model output after softmax over each SD pair's K paths.
        tm_final:
            [B, P, 1] final-timestep traffic repeated once per path slot.
        paths_to_edges:
            sparse [P, E] path-edge incidence for the final/current path set.
        capacities:
            [E] or [B, E] candidate capacities aligned with final/current edge ids.
        num_paths_per_pair:
            K paths per SD pair.
        local_rescale:
            If True, any path using a zero-capacity edge is removed and that
            SD pair's traffic is renormalized over surviving paths. This mimics
            DOTE/TEAL-style local data-plane rescaling rather than recomputing
            paths with a controller.
        disconnected_penalty:
            Extra penalty when every path for an SD pair is blocked.

    Returns:
        mlu:
            scalar tensor, average over batch.
        disconnected_fraction:
            scalar tensor, fraction of final-timestep demand whose SD pair has
            no surviving candidate path.
    """
    if split_ratios is None:
        raise RuntimeError(
            "Resilience objective requires pred_splits, but model did not return splits."
        )

    if not paths_to_edges.is_sparse:
        raise RuntimeError("paths_to_edges must be a sparse tensor")

    paths_to_edges = paths_to_edges.coalesce()
    batch_size, total_paths = split_ratios.shape

    if total_paths % num_paths_per_pair != 0:
        raise RuntimeError(
            f"total_paths={total_paths} is not divisible by num_paths_per_pair={num_paths_per_pair}"
        )

    num_pairs = total_paths // num_paths_per_pair

    capacities = _as_batched_capacities(
        capacities,
        batch_size=batch_size,
        device=split_ratios.device,
        dtype=split_ratios.dtype,
    )

    if capacities.shape[1] != paths_to_edges.shape[1]:
        raise RuntimeError(
            f"Candidate capacity length {capacities.shape[1]} does not match "
            f"paths_to_edges edge dimension {paths_to_edges.shape[1]}"
        )

    tm_final = tm_final.to(device=split_ratios.device, dtype=split_ratios.dtype)

    if tm_final.dim() != 3:
        raise RuntimeError(f"tm_final should be [B, P, 1], got {tuple(tm_final.shape)}")

    if tm_final.shape[1] != total_paths:
        raise RuntimeError(
            f"tm_final path dimension {tm_final.shape[1]} does not match split dimension {total_paths}"
        )

    effective_splits = split_ratios
    disconnected_fraction = torch.tensor(
        0.0,
        device=split_ratios.device,
        dtype=split_ratios.dtype,
    )

    if local_rescale:
        # A path is alive only if every edge on that path has positive capacity.
        inactive_edges = (capacities <= eps).to(dtype=torch.float32)

        # sparse [P, E] @ dense [E, B] -> dense [P, B] -> [B, P]
        blocked_counts = torch.sparse.mm(
            paths_to_edges.to(dtype=torch.float32),
            inactive_edges.t(),
        ).t()

        path_alive = (blocked_counts <= 0.0).to(dtype=split_ratios.dtype)

        splits_by_pair = split_ratios.reshape(
            batch_size,
            num_pairs,
            num_paths_per_pair,
        )
        alive_by_pair = path_alive.reshape(
            batch_size,
            num_pairs,
            num_paths_per_pair,
        )

        masked_splits = splits_by_pair * alive_by_pair
        denom = masked_splits.sum(dim=-1, keepdim=True)

        rescaled_splits = torch.where(
            denom > eps,
            masked_splits / denom.clamp_min(eps),
            torch.zeros_like(masked_splits),
        )

        effective_splits = rescaled_splits.reshape(batch_size, total_paths)

        # If all paths for an SD pair are blocked, the traffic is disconnected.
        disconnected_pairs = (denom.squeeze(-1) <= eps)

        tm_by_pair = tm_final.squeeze(-1).reshape(
            batch_size,
            num_pairs,
            num_paths_per_pair,
        )[:, :, 0]

        total_demand = tm_by_pair.sum(dim=-1).clamp_min(eps)
        disconnected_demand = (tm_by_pair * disconnected_pairs.to(tm_by_pair.dtype)).sum(dim=-1)
        disconnected_fraction = (disconnected_demand / total_demand).mean()

    data_on_tunnels = effective_splits * tm_final.squeeze(-1)

    data_on_links = torch.sparse.mm(
        paths_to_edges.to(dtype=torch.float32).t(),
        data_on_tunnels.to(dtype=torch.float32).t(),
    ).t()

    data_on_links = data_on_links.to(dtype=split_ratios.dtype)

    edges_util = data_on_links / capacities.clamp_min(eps)
    edges_util = torch.nan_to_num(edges_util, nan=0.0, posinf=1e6, neginf=0.0)

    mlu_per_batch = edges_util.max(dim=-1).values

    if disconnected_penalty > 0.0:
        mlu_per_batch = mlu_per_batch + float(disconnected_penalty) * disconnected_fraction

    return mlu_per_batch.mean(), disconnected_fraction


def _future_mlu_from_true_next_state(
    sample,
    pred_splits,
    num_paths_per_pair,
    local_rescale=True,
    disconnected_penalty=100.0,
):
    """
    Computes MLU(r, tau_{t+1}) using the actual next-step capacity state
    saved by the dynamic generator.

    This replaces the old heuristic top-K candidate expectation. The model is
    trained against the realized next failure/degradation state from the global
    generated timeline.
    """
    future_capacities = sample.get("future_capacities", None)

    if future_capacities is None:
        raise RuntimeError(
            "Resilience objective requested, but sample is missing future_capacities. "
            "Regenerate the dataset with next-step future labels."
        )

    if pred_splits is None:
        raise RuntimeError("Future-resilience objective requires split ratios from the model.")

    final_paths_to_edges = sample["paths_to_edges"][-1].coalesce()
    tm_final = sample["tm"][:, -1]

    future_mlu, future_disconnected = _compute_mlu_from_splits_under_capacities(
        split_ratios=pred_splits,
        tm_final=tm_final,
        paths_to_edges=final_paths_to_edges,
        capacities=future_capacities,
        num_paths_per_pair=num_paths_per_pair,
        local_rescale=local_rescale,
        disconnected_penalty=disconnected_penalty,
    )

    return future_mlu, future_disconnected




def _default_multiplier_levels(num_classes: int):
    """
    Label convention shared with generate_dynamic_failures.py:
      0 -> 1.00 healthy
      1 -> 0.75
      2 -> 0.50
      3 -> 0.25
      4 -> 0.00 full failure
    """
    if int(num_classes) == 5:
        return [1.0, 0.75, 0.50, 0.25, 0.0]
    if int(num_classes) == 2:
        return [1.0, 0.0]
    return torch.linspace(1.0, 0.0, steps=int(num_classes)).tolist()


def _failure_output_is_scenario_dict(failure_output):
    return (
        isinstance(failure_output, dict)
        and "scenario_logits" in failure_output
        and "scenario_edge_logits" in failure_output
    )


def _failure_output_all_finite(failure_output):
    if failure_output is None:
        return True
    if torch.is_tensor(failure_output):
        return torch.isfinite(failure_output).all()
    if isinstance(failure_output, dict):
        finite = True
        for value in failure_output.values():
            if torch.is_tensor(value):
                finite = finite and bool(torch.isfinite(value).all())
        return finite
    return True


def _scenario_probabilities(failure_output, detach_probabilities=True):
    if not _failure_output_is_scenario_dict(failure_output):
        raise RuntimeError(
            "K-scenario topology objective requires failure_output dict with "
            "scenario_logits and scenario_edge_logits."
        )
    scenario_logits = failure_output["scenario_logits"]
    probs = torch.softmax(scenario_logits.float(), dim=-1).to(dtype=scenario_logits.dtype)
    if detach_probabilities:
        probs = probs.detach()
    return probs


def _scenario_capacities_from_failure_output(
    failure_output,
    current_capacities,
    capacity_mode="hard",
    detach_capacities=True,
):
    """
    Converts K scenario edge-class logits into K full capacity vectors.

    failure_output:
        scenario_edge_logits [B, K, E, S]

    current_capacities:
        [E] or [B, E]

    capacity_mode:
        hard:
            forward pass uses argmax edge classes for each scenario.
        straight_through:
            forward pass uses hard classes but gradients flow as if softmax
            were used. Usually combine with detach_capacities=False.
        soft:
            uses expected multiplier inside each scenario.

    By default detach_capacities=True so the resilience MLU term trains the
    routing network against the failure head's predicted modes, while the
    failure head itself is trained by the supervised WTA loss instead of
    gaming the MLU objective.
    """
    if not _failure_output_is_scenario_dict(failure_output):
        raise RuntimeError(
            "K-scenario topology objective requires failure_output dict with "
            "scenario_logits and scenario_edge_logits."
        )

    scenario_edge_logits = failure_output["scenario_edge_logits"]
    if scenario_edge_logits.dim() != 4:
        raise RuntimeError(
            f"scenario_edge_logits should be [B,K,E,S], got {tuple(scenario_edge_logits.shape)}"
        )

    B, K, E, S = scenario_edge_logits.shape

    current_capacities = _as_batched_capacities(
        current_capacities,
        batch_size=B,
        device=scenario_edge_logits.device,
        dtype=scenario_edge_logits.dtype,
    )

    if current_capacities.shape[1] != E:
        raise RuntimeError(
            f"current capacity length {current_capacities.shape[1]} does not match "
            f"scenario edge dimension {E}."
        )

    levels = torch.tensor(
        _default_multiplier_levels(S),
        device=scenario_edge_logits.device,
        dtype=scenario_edge_logits.dtype,
    )

    probs = torch.softmax(scenario_edge_logits.float(), dim=-1).to(dtype=scenario_edge_logits.dtype)
    mode = str(capacity_mode).lower()

    if mode in {"soft", "expected", "expected_capacity"}:
        edge_state_weights = probs
    elif mode in {"hard", "argmax"}:
        labels = probs.argmax(dim=-1)
        edge_state_weights = F.one_hot(labels, num_classes=S).to(dtype=probs.dtype)
    elif mode in {"straight_through", "st"}:
        labels = probs.argmax(dim=-1)
        hard = F.one_hot(labels, num_classes=S).to(dtype=probs.dtype)
        edge_state_weights = hard.detach() - probs.detach() + probs
    else:
        raise ValueError(
            f"Unknown scenario_capacity_mode={capacity_mode}. "
            "Use hard, straight_through, or soft."
        )

    multipliers = (edge_state_weights * levels.view(1, 1, 1, S)).sum(dim=-1)

    if detach_capacities:
        multipliers = multipliers.detach()

    scenario_capacities = current_capacities.unsqueeze(1) * multipliers
    return scenario_capacities


def _expected_mlu_from_grate_scenario_modes(
    sample,
    pred_splits,
    failure_output,
    num_paths_per_pair,
    local_rescale=True,
    disconnected_penalty=100.0,
    scenario_capacity_mode="hard",
    detach_scenario_probabilities=True,
    detach_scenario_capacities=True,
):
    """
    Computes GRATE's self-contained K-mode topology resilience objective:

        sum_k p_k * MLU_model(topology_k)

    No heuristic candidate generator is called here. GRATE supplies both:
      1. the K full-topology scenario modes, through scenario_edge_logits
      2. the probabilities p_k, through scenario_logits
    """
    if pred_splits is None:
        raise RuntimeError("K-scenario resilience objective requires split ratios.")

    if not _failure_output_is_scenario_dict(failure_output):
        raise RuntimeError(
            "K-scenario resilience objective requires the model to return a "
            "failure_output dict with scenario_logits and scenario_edge_logits."
        )

    final_paths_to_edges = sample["paths_to_edges"][-1].coalesce()
    tm_final = sample["tm"][:, -1]
    current_capacities = sample["capacities"][-1]

    scenario_capacities = _scenario_capacities_from_failure_output(
        failure_output=failure_output,
        current_capacities=current_capacities,
        capacity_mode=scenario_capacity_mode,
        detach_capacities=detach_scenario_capacities,
    )

    scenario_probs = _scenario_probabilities(
        failure_output,
        detach_probabilities=detach_scenario_probabilities,
    ).to(device=pred_splits.device, dtype=pred_splits.dtype)

    if scenario_probs.shape[0] != 1:
        raise RuntimeError(
            "Dynamic K-scenario objective currently expects batch size 1. "
            f"Got scenario_probs shape {tuple(scenario_probs.shape)}."
        )

    expected_mlu = torch.tensor(0.0, device=pred_splits.device, dtype=pred_splits.dtype)
    expected_disconnected = torch.tensor(0.0, device=pred_splits.device, dtype=pred_splits.dtype)

    K = scenario_capacities.shape[1]
    for k in range(K):
        scenario_mlu, scenario_disconnected = _compute_mlu_from_splits_under_capacities(
            split_ratios=pred_splits,
            tm_final=tm_final,
            paths_to_edges=final_paths_to_edges,
            capacities=scenario_capacities[:, k, :],
            num_paths_per_pair=num_paths_per_pair,
            local_rescale=local_rescale,
            disconnected_penalty=disconnected_penalty,
        )
        expected_mlu = expected_mlu + scenario_probs[0, k] * scenario_mlu
        expected_disconnected = expected_disconnected + scenario_probs[0, k] * scenario_disconnected

    return expected_mlu, expected_disconnected


def _failure_prediction_loss(failure_output, sample, scenario_probability_loss_weight=1.0):
    """
    Supervised next-step failure/capacity-state loss.

    Supports two model output formats:
      Legacy per-edge marginal head:
          failure_output [B, E, S]
          uses normal per-edge CE.

      New K-scenario whole-topology head:
          failure_output["scenario_logits"]      [B, K]
          failure_output["scenario_edge_logits"] [B, K, E, S]
          uses winner-take-all / multiple-choice learning:
            1. compute per-scenario edge CE against the single realized topology
            2. choose the best matching scenario k*
            3. train only that scenario's edge content CE
            4. train scenario_logits to put probability on k*
    """
    labels = sample.get("future_edge_state_labels", None)

    if labels is None:
        raise RuntimeError(
            "Failure prediction requested, but sample is missing "
            "future_edge_state_labels. Regenerate the dataset with next-step labels."
        )

    if failure_output is None:
        raise RuntimeError(
            "Failure prediction loss requested, but the model did not return failure output."
        )

    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    if _failure_output_is_scenario_dict(failure_output):
        scenario_logits = failure_output["scenario_logits"]
        scenario_edge_logits = failure_output["scenario_edge_logits"]

        labels = labels.to(device=scenario_edge_logits.device, dtype=torch.long)

        if scenario_edge_logits.dim() != 4:
            raise RuntimeError(
                f"scenario_edge_logits should be [B,K,E,S], got {tuple(scenario_edge_logits.shape)}"
            )

        B, K, E, S = scenario_edge_logits.shape
        if labels.shape != (B, E):
            raise RuntimeError(
                f"scenario labels shape {tuple(labels.shape)} does not match "
                f"scenario edge shape {(B, E)}"
            )

        expanded_labels = labels.unsqueeze(1).expand(B, K, E)
        per_edge_ce = F.cross_entropy(
            scenario_edge_logits.reshape(B * K * E, S).float(),
            expanded_labels.reshape(B * K * E),
            reduction="none",
        ).reshape(B, K, E)

        scenario_content_ce = per_edge_ce.mean(dim=-1)  # [B, K]

        with torch.no_grad():
            winning_scenarios = scenario_content_ce.argmin(dim=1)  # [B]

        batch_index = torch.arange(B, device=scenario_edge_logits.device)
        content_loss = scenario_content_ce[batch_index, winning_scenarios].mean()
        probability_loss = F.cross_entropy(
            scenario_logits.float(),
            winning_scenarios,
        ).to(dtype=scenario_edge_logits.dtype)

        loss = content_loss.to(dtype=scenario_edge_logits.dtype) + (
            float(scenario_probability_loss_weight) * probability_loss
        )

        winning_logits = scenario_edge_logits[batch_index, winning_scenarios]
        pred_labels = winning_logits.argmax(dim=-1)
        accuracy = (pred_labels == labels).to(dtype=scenario_edge_logits.dtype).mean()

        return loss, accuracy

    # Legacy [B, E, S] per-edge marginal output.
    failure_logits = failure_output
    labels = labels.to(device=failure_logits.device, dtype=torch.long)

    if failure_logits.dim() != 3:
        raise RuntimeError(
            f"legacy failure_logits should be [B, E, C], got {tuple(failure_logits.shape)}"
        )

    if failure_logits.shape[:2] != labels.shape:
        raise RuntimeError(
            f"failure_logits edge shape {tuple(failure_logits.shape[:2])} "
            f"does not match labels shape {tuple(labels.shape)}"
        )

    num_classes = failure_logits.shape[-1]
    loss = F.cross_entropy(
        failure_logits.reshape(-1, num_classes).float(),
        labels.reshape(-1),
    ).to(dtype=failure_logits.dtype)

    pred_labels = failure_logits.argmax(dim=-1)
    accuracy = (pred_labels == labels).to(dtype=failure_logits.dtype).mean()

    return loss, accuracy

def _soft_backup_penalty(pred_splits, num_paths_per_pair):
    """
    Optional regularizer encouraging each SD pair to keep some traffic on backup
    paths. This is intentionally off by default. Lower is better.

    For one SD pair with K paths:
        sum_i split_i^2
    is small when traffic is spread and large when traffic is concentrated.
    """
    if pred_splits is None:
        return torch.tensor(0.0)

    batch_size, total_paths = pred_splits.shape

    if total_paths % num_paths_per_pair != 0:
        raise RuntimeError(
            f"total_paths={total_paths} is not divisible by num_paths_per_pair={num_paths_per_pair}"
        )

    splits = pred_splits.reshape(batch_size, -1, num_paths_per_pair)
    return (splits ** 2).sum(dim=-1).mean()


def dynamic_mlu_loss(
    predicted,
    sample,
    pred_splits=None,
    failure_logits=None,
    use_opt=True,
    split_loss_weight=0.0,
    use_resilience_objective=False,
    resilience_loss_weight=0.0,
    failure_prediction_loss_weight=0.0,
    soft_backup_penalty_weight=0.0,
    local_rescale=True,
    disconnected_penalty=100.0,
    num_paths_per_pair=4,
    num_failure_scenarios=5,
    scenario_probability_loss_weight=1.0,
    scenario_capacity_mode="hard",
    detach_scenario_probabilities=True,
    detach_scenario_capacities=True,
    # Kept for backwards compatibility with older scripts; unused by the
    # K-scenario GRATE objective.
    num_failure_candidates=8,
    model_prob_temperature=1.0,
    model_prob_edge_aggregation="sum",
    detach_candidate_probabilities=True,
):
    """
    Dynamic-sample loss.

    Default clean mode:
        loss = current_mlu

    Solver-normalized legacy mode:
        if use_opt=True and sample["opt"] exists, loss = current_mlu / opt.
        Do not use this for the final no-Gurobi claim.

    K-scenario learned-failure-resilience mode:
        loss = current_mlu
             + lambda * sum_k P_GRATE(scenario_k) * MLU_model(scenario_k)
             + beta * WTA_scenario_prediction_loss
             + mu * optional_soft_backup_penalty

    GRATE predicts K whole-topology scenario modes directly. No heuristic
    candidate generator is used during training/validation/test_dynamic.
    """
    if not torch.isfinite(predicted).all():
        raise RuntimeError("Non-finite predicted tensor reached dynamic_mlu_loss")

    if pred_splits is not None and not torch.isfinite(pred_splits).all():
        raise RuntimeError("Non-finite split tensor reached dynamic_mlu_loss")

    if failure_logits is not None and not _failure_output_all_finite(failure_logits):
        raise RuntimeError("Non-finite failure-output tensor reached dynamic_mlu_loss")

    predicted = predicted.clamp_min(0.0)

    current_mlu = predicted.max()
    total_util = predicted.sum()

    if "tm" in sample and torch.is_tensor(sample["tm"]):
        tm_final_sum = sample["tm"][:, -1].sum()
    else:
        tm_final_sum = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)

    future_mlu = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
    future_disconnected = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
    resilience_score = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
    backup_penalty = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
    failure_pred_loss = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)
    failure_pred_acc = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)

    if use_resilience_objective:
        # For the research objective, avoid Gurobi normalization. The current
        # term should be the raw current MLU from the model.
        base_loss = current_mlu

        future_mlu, future_disconnected = _expected_mlu_from_grate_scenario_modes(
            sample=sample,
            pred_splits=pred_splits,
            failure_output=failure_logits,
            num_paths_per_pair=num_paths_per_pair,
            local_rescale=local_rescale,
            disconnected_penalty=disconnected_penalty,
            scenario_capacity_mode=scenario_capacity_mode,
            detach_scenario_probabilities=detach_scenario_probabilities,
            detach_scenario_capacities=detach_scenario_capacities,
        )

        # Predict next-step link/capacity states whenever labels are present.
        # Set failure_prediction_loss_weight=0 to log accuracy without training
        # on the prediction task, but for GRATE we normally want beta > 0.
        if "future_edge_state_labels" in sample:
            failure_pred_loss, failure_pred_acc = _failure_prediction_loss(
                failure_output=failure_logits,
                sample=sample,
                scenario_probability_loss_weight=scenario_probability_loss_weight,
            )

        if soft_backup_penalty_weight > 0.0:
            backup_penalty = _soft_backup_penalty(
                pred_splits=pred_splits,
                num_paths_per_pair=num_paths_per_pair,
            ).to(device=predicted.device, dtype=predicted.dtype)

        resilience_score = current_mlu / future_mlu.clamp_min(1e-12)

    else:
        opt = sample.get("opt", None)

        if use_opt and opt is not None:
            if not torch.is_tensor(opt):
                opt = torch.tensor(opt, device=predicted.device, dtype=predicted.dtype)
            else:
                opt = opt.to(device=predicted.device, dtype=predicted.dtype)

            opt = opt.reshape(()).clamp_min(1e-12)
            base_loss = current_mlu / opt
        else:
            base_loss = current_mlu

    # Temporary no-Gurobi guardrail against all-zero utilization collapse.
    has_traffic = tm_final_sum > 1e-8

    zero_collapse_penalty = torch.where(
        has_traffic,
        torch.relu(
            torch.tensor(1e-4, device=predicted.device, dtype=predicted.dtype) - total_util
        ) * 1000.0,
        torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype),
    )

    anti_collapse_term = torch.where(
        has_traffic,
        -0.001 * torch.log(total_util.clamp_min(1e-6)),
        torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype),
    )

    split_loss = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)

    if split_loss_weight > 0.0:
        opt_splits = sample.get("opt_splits", None)

        if opt_splits is not None and pred_splits is not None:
            opt_splits = opt_splits.to(device=pred_splits.device, dtype=pred_splits.dtype)

            if opt_splits.dim() == 1:
                opt_splits = opt_splits.unsqueeze(0)

            if opt_splits.shape != pred_splits.shape:
                opt_splits = opt_splits.reshape_as(pred_splits)

            split_loss = torch.mean((pred_splits - opt_splits) ** 2)

    loss = (
        base_loss
        + float(resilience_loss_weight) * future_mlu
        + float(failure_prediction_loss_weight) * failure_pred_loss
        + float(soft_backup_penalty_weight) * backup_penalty
        + anti_collapse_term
        + zero_collapse_penalty
        + float(split_loss_weight) * split_loss
    )

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite dynamic loss computed")

    loss_value = float(loss.detach().cpu())
    current_term_value = float(base_loss.detach().cpu())
    raw_mlu_value = float(current_mlu.detach().cpu())
    total_util_value = float(total_util.detach().cpu())
    tm_sum_value = float(tm_final_sum.detach().cpu())
    split_loss_value = float(split_loss.detach().cpu())
    future_mlu_value = float(future_mlu.detach().cpu())
    resilience_score_value = float(resilience_score.detach().cpu())
    backup_penalty_value = float(backup_penalty.detach().cpu())
    future_disconnected_value = float(future_disconnected.detach().cpu())
    failure_pred_loss_value = float(failure_pred_loss.detach().cpu())
    failure_pred_acc_value = float(failure_pred_acc.detach().cpu())

    return (
        loss,
        loss_value,
        current_term_value,
        raw_mlu_value,
        total_util_value,
        tm_sum_value,
        split_loss_value,
        future_mlu_value,
        resilience_score_value,
        backup_penalty_value,
        future_disconnected_value,
        failure_pred_loss_value,
        failure_pred_acc_value,
    )



# -----------------------------------------------------------------------------
# Shared static-cache support for large dynamic datasets.
# -----------------------------------------------------------------------------
#
# Full KDL with K>1 should not store path/topology tensors inside every
# sample_*.pt. Instead, each small sample can contain:
#
#   static_cache_path: "static_cache.pt"
#
# and the cache contains:
#
#   edge_index
#   padded_edge_ids_per_path
#   paths_to_edges
#
# Samples may store capacities either as the old list length T or as a compact
# tensor [1, T, E]. This helper hydrates the compact format back into the
# old list-style format expected by the current HARP dynamic branch.
# -----------------------------------------------------------------------------


_GLOBAL_STATIC_CACHE_STORE = {}


def _infer_history_len_from_sample(sample):
    if "node_features" in sample and torch.is_tensor(sample["node_features"]):
        # [1, T, N, 2]
        return int(sample["node_features"].shape[1])
    if "tm" in sample and torch.is_tensor(sample["tm"]):
        # [1, T, P, 1]
        return int(sample["tm"].shape[1])
    if (
        "capacities" in sample
        and torch.is_tensor(sample["capacities"])
        and sample["capacities"].dim() == 3
    ):
        # [1, T, E]
        return int(sample["capacities"].shape[1])
    raise RuntimeError("Cannot infer history length T from sample.")


def _resolve_static_cache_path(sample, sample_path=None):
    cache_path = sample.get("static_cache_path", None)

    if cache_path is None and isinstance(sample.get("metadata", None), dict):
        cache_path = sample["metadata"].get("static_cache_path", None)

    if cache_path is None:
        return None

    cache_path = Path(cache_path)

    if not cache_path.is_absolute():
        if sample_path is not None:
            cache_path = Path(sample_path).parent / cache_path

    return cache_path


def _load_static_cache_for_sample(sample, sample_path=None, cache_store=None):
    cache_path = _resolve_static_cache_path(sample, sample_path=sample_path)

    if cache_path is None:
        return None

    cache_path = cache_path.resolve()

    if cache_store is None:
        cache_store = _GLOBAL_STATIC_CACHE_STORE

    key = str(cache_path)

    if key not in cache_store:
        if not cache_path.exists():
            raise FileNotFoundError(f"static_cache_path does not exist: {cache_path}")
        cache_store[key] = torch.load(cache_path, map_location="cpu")

    return cache_store[key]


def _first_present(mapping, names, required=True):
    for name in names:
        if name in mapping:
            return mapping[name]
    if required:
        raise KeyError(f"None of these keys found: {names}")
    return None


def _repeat_static_tensor_as_time_list(value, T):
    if isinstance(value, list):
        return value
    return [value for _ in range(T)]


def _capacities_to_time_list(value, T):
    if isinstance(value, list):
        return value

    if not torch.is_tensor(value):
        raise TypeError(f"capacities must be tensor or list, got {type(value)}")

    # Compact format: [1, T, E]
    if value.dim() == 3:
        if value.shape[1] != T:
            raise RuntimeError(f"capacities has T={value.shape[1]}, expected T={T}")
        return [value[:, t, :] for t in range(T)]

    # Compact format: [T, E]
    if value.dim() == 2 and value.shape[0] == T:
        return [value[t].unsqueeze(0) for t in range(T)]

    # Static format: [1, E]
    if value.dim() == 2:
        return [value for _ in range(T)]

    # Static format: [E]
    if value.dim() == 1:
        return [value.unsqueeze(0) for _ in range(T)]

    raise RuntimeError(f"Unsupported capacities shape: {tuple(value.shape)}")


def hydrate_dynamic_sample_static_cache(sample, sample_path=None, cache_store=None):
    """
    Returns a sample compatible with the existing HARP dynamic branch.

    Old format:
        sample already contains list-valued edge_index/capacities/paths/etc.
        -> returned mostly unchanged.

    New shared-cache format:
        sample contains static_cache_path and compact capacities [1,T,E].
        -> attaches cache tensors and expands compact capacities into lists.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dict, got {type(sample)}")

    cache = _load_static_cache_for_sample(
        sample,
        sample_path=sample_path,
        cache_store=cache_store,
    )

    if cache is None:
        return sample

    sample = dict(sample)
    T = _infer_history_len_from_sample(sample)

    if "edge_index" not in sample:
        edge_index = _first_present(cache, ["edge_index", "edge_indices"])
        sample["edge_index"] = _repeat_static_tensor_as_time_list(edge_index, T)

    if "padded_edge_ids_per_path" not in sample:
        padded = _first_present(
            cache,
            ["padded_edge_ids_per_path", "padded_paths", "padded_edge_ids"],
        )
        sample["padded_edge_ids_per_path"] = _repeat_static_tensor_as_time_list(padded, T)

    if "paths_to_edges" not in sample:
        pte = _first_present(cache, ["paths_to_edges", "pte"])
        sample["paths_to_edges"] = _repeat_static_tensor_as_time_list(pte, T)

    if "capacities" in sample:
        sample["capacities"] = _capacities_to_time_list(sample["capacities"], T)
    else:
        caps = _first_present(
            cache,
            ["capacities", "base_capacities"],
            required=False,
        )
        if caps is None:
            raise KeyError(
                "Sample has no capacities, and static cache has no capacities/base_capacities."
            )
        sample["capacities"] = _capacities_to_time_list(caps, T)

    if "metadata" not in sample or sample["metadata"] is None:
        sample["metadata"] = {}

    if isinstance(sample["metadata"], dict):
        resolved = _resolve_static_cache_path(sample, sample_path=sample_path)
        sample["metadata"]["static_cache_hydrated"] = True
        sample["metadata"]["static_cache_path_resolved"] = (
            str(resolved) if resolved is not None else None
        )

    return sample


class DynamicAbileneDataset(Dataset):
    """
    Loads generated dynamic .pt samples.

    Supports both formats:

    Old full sample format:
        every sample_*.pt contains edge_index, capacities,
        padded_edge_ids_per_path, paths_to_edges.

    New shared-static-cache format:
        every sample_*.pt contains small temporal tensors plus:
            static_cache_path: "static_cache.pt"

        static_cache.pt contains the large repeated path/topology tensors.
    """

    def __init__(self, samples_dir, start_idx=0, end_idx=None):
        self.samples_dir = Path(samples_dir)
        self.files = sorted(glob.glob(str(self.samples_dir / "sample_*.pt")))
        self._static_cache_store = {}

        if end_idx is None:
            end_idx = len(self.files)

        self.files = self.files[start_idx:end_idx]

        if len(self.files) == 0:
            raise ValueError(
                f"No dynamic sample files found in {samples_dir} "
                f"for slice [{start_idx}:{end_idx}]"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        sample = torch.load(fp, map_location="cpu")
        sample = hydrate_dynamic_sample_static_cache(
            sample,
            sample_path=fp,
            cache_store=self._static_cache_store,
        )
        return sample


def dynamic_collate(batch):
    """
    Dynamic topology samples contain sparse tensors and lists whose edge counts
    can differ per sample. Start with batch size 1.
    """
    if len(batch) != 1:
        raise ValueError("Dynamic topology mode currently supports only batch_size=1.")
    return batch[0]


def _tensor_to_device(value, device, dtype=None):
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def move_dynamic_sample_to_device(sample, device, dtype):
    """
    Moves one dynamic sample to GPU/CPU.

    Keeps metadata untouched.
    Keeps integer tensors as integer tensors.
    Casts floating tensors/sparse tensors to props.dtype.
    """
    out = {}

    for key, value in sample.items():
        if key == "metadata":
            out[key] = value

        elif key in ["edge_index", "padded_edge_ids_per_path"]:
            out[key] = [x.to(device=device) for x in value]

        elif key in ["capacities", "paths_to_edges"]:
            out[key] = [x.to(device=device, dtype=dtype) for x in value]

        elif torch.is_tensor(value):
            if key in ["future_edge_state_labels"]:
                out[key] = value.to(device=device, dtype=torch.long)
            elif key in [
                    "node_features",
                    "tm",
                    "tm_pred",
                    "opt",
                    "opt_splits",
                    "future_capacities",
                    "future_capacity_multipliers",
                    "failure_candidate_probs",
                    "failure_candidate_capacities",
                    "failure_candidate_capacity_multipliers",
                ]:
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)

        else:
            out[key] = value

    return out


def run_model_on_dynamic_sample(model, props, sample):
    """
    Runs HARP on one dynamic sample.

    The HARP dynamic branch is triggered because edge_index/capacities/
    padded_edge_ids_per_path/paths_to_edges are lists over time.

    Dynamic training asks HARP to return both edge utilizations and path split
    ratios so we can optionally add a Gurobi split imitation loss.
    """
    old_return_splits = getattr(props, "return_splits", False)
    old_return_failure_logits = getattr(props, "return_failure_logits", False)
    props.return_splits = True
    props.return_failure_logits = True

    try:
        return model(
            props,
            sample["node_features"],
            sample["edge_index"],
            sample["capacities"],
            sample["padded_edge_ids_per_path"],
            sample["tm"],
            sample["tm_pred"] if props.pred else sample["tm"],
            sample["paths_to_edges"],
            None,
            None,
        )
    finally:
        props.return_splits = old_return_splits
        props.return_failure_logits = old_return_failure_logits



DYNAMIC_METRIC_KEYS = [
    "loss",
    "current_term",
    "raw_mlu",
    "expected_failure_mlu",
    "resilience_score",
    "expected_disconnected",
    "failure_prediction_loss",
    "failure_prediction_accuracy",
    "backup_penalty",
    "split_loss",
    "total_util",
    "tm_sum",
]


def _new_dynamic_metric_tracker():
    return {key: [] for key in DYNAMIC_METRIC_KEYS}


def _build_dynamic_metric_record(
    loss_value,
    current_term_value,
    raw_mlu_value,
    total_util_value,
    tm_sum_value,
    split_loss_value,
    expected_failure_mlu_value,
    resilience_score_value,
    backup_penalty_value,
    expected_disconnected_value,
    failure_prediction_loss_value=0.0,
    failure_prediction_accuracy_value=0.0,
):
    return {
        "loss": float(loss_value),
        "current_term": float(current_term_value),
        "raw_mlu": float(raw_mlu_value),
        "expected_failure_mlu": float(expected_failure_mlu_value),
        "resilience_score": float(resilience_score_value),
        "expected_disconnected": float(expected_disconnected_value),
        "failure_prediction_loss": float(failure_prediction_loss_value),
        "failure_prediction_accuracy": float(failure_prediction_accuracy_value),
        "backup_penalty": float(backup_penalty_value),
        "split_loss": float(split_loss_value),
        "total_util": float(total_util_value),
        "tm_sum": float(tm_sum_value),
    }


def _append_dynamic_metric_record(tracker, record):
    for key in DYNAMIC_METRIC_KEYS:
        value = record.get(key, 0.0)
        tracker[key].append(float(value))


def _safe_mean(values):
    if len(values) == 0:
        return float("nan")
    return float(statistics.mean(values))


def _safe_median(values):
    if len(values) == 0:
        return float("nan")
    return float(statistics.median(values))


def _nearest_percentile(values, pct):
    """
    Percentile helper matching the old code style: sort and take the nearest
    available index. pct should be in [0, 100].
    """
    if len(values) == 0:
        return float("nan")
    values_sorted = sorted(float(v) for v in values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    pct = max(0.0, min(100.0, float(pct)))
    idx = int(round((pct / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def _distribution_summary(values):
    values = [float(v) for v in values]
    return {
        "avg": _safe_mean(values),
        "median": _safe_median(values),
        "p25": _nearest_percentile(values, 25),
        "p75": _nearest_percentile(values, 75),
        "p90": _nearest_percentile(values, 90),
        "p95": _nearest_percentile(values, 95),
        "p99": _nearest_percentile(values, 99),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def _summarize_dynamic_metrics(tracker):
    return {key: _distribution_summary(tracker[key]) for key in DYNAMIC_METRIC_KEYS}


def _running_postfix(tracker, skipped):
    """
    TQDM display values. These are running averages, not the last sample's values.
    That avoids confusing a single sample with the aggregate result.
    """
    return {
        "loss": _safe_mean(tracker["loss"]),
        "current": _safe_mean(tracker["current_term"]),
        "raw_mlu": _safe_mean(tracker["raw_mlu"]),
        "exp_fail": _safe_mean(tracker["expected_failure_mlu"]),
        "score": _safe_mean(tracker["resilience_score"]),
        "disc": _safe_mean(tracker["expected_disconnected"]),
        "pred_loss": _safe_mean(tracker["failure_prediction_loss"]),
        "pred_acc": _safe_mean(tracker["failure_prediction_accuracy"]),
        "backup": _safe_mean(tracker["backup_penalty"]),
        "split_loss": _safe_mean(tracker["split_loss"]),
        "total_util": _safe_mean(tracker["total_util"]),
        "tm_sum": _safe_mean(tracker["tm_sum"]),
        "skipped": skipped,
    }


def _dynamic_metrics_csv_path(props):
    """
    Per-epoch/stage CSV used for plotting training/validation/test curves.

    If run_harp.py later attaches props.dynamic_metrics_csv, this function will
    respect it. Otherwise, it writes a default file under results/.
    """
    configured = getattr(props, "dynamic_metrics_csv", None)
    if configured:
        path = Path(configured)
    else:
        topo = getattr(props, "topo", "dynamic")
        k_paths = getattr(props, "num_paths_per_pair", "k")
        path = Path("results") / f"dynamic_metrics_{topo}_{k_paths}sp.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_stage_summary_csv(props, stage, tracker, skipped, epoch=None, n_epochs=None):
    """
    Appends one aggregate row for a train/validation/test stage.
    This gives you a simple CSV to plot later.
    """
    path = _dynamic_metrics_csv_path(props)
    summary = _summarize_dynamic_metrics(tracker)

    row = {
        "stage": stage,
        "epoch": "" if epoch is None else int(epoch),
        "n_epochs": "" if n_epochs is None else int(n_epochs),
        "count": len(tracker["loss"]),
        "skipped": int(skipped),
    }

    for key in DYNAMIC_METRIC_KEYS:
        for stat_name in ["avg", "median", "p25", "p75", "p90", "p95", "p99", "min", "max"]:
            row[f"{key}_{stat_name}"] = summary[key][stat_name]

    # Also save the aggregate ratio of means, which is different from the mean
    # of per-sample ratios. Both are useful diagnostics.
    mean_raw = summary["raw_mlu"]["avg"]
    mean_expected = summary["expected_failure_mlu"]["avg"]
    if mean_expected and mean_expected == mean_expected and abs(mean_expected) > 1e-12:
        row["aggregate_score_raw_over_expected"] = mean_raw / mean_expected
    else:
        row["aggregate_score_raw_over_expected"] = float("nan")

    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row


def _print_stage_summary(stage, tracker, skipped):
    summary = _summarize_dynamic_metrics(tracker)
    count = len(tracker["loss"])
    print(f"{stage} aggregate metrics over {count} samples, skipped={skipped}")
    print(f"  avg loss: {summary['loss']['avg']:.6f}")
    print(f"  avg current term: {summary['current_term']['avg']:.6f}")
    print(f"  avg raw current MLU: {summary['raw_mlu']['avg']:.6f}")
    print(f"  avg expected scenario MLU: {summary['expected_failure_mlu']['avg']:.6f}")
    print(f"  avg resiliency score: {summary['resilience_score']['avg']:.6f}")
    print(f"  median resiliency score: {summary['resilience_score']['median']:.6f}")
    print(f"  p90 expected scenario MLU: {summary['expected_failure_mlu']['p90']:.6f}")
    print(f"  p95 expected scenario MLU: {summary['expected_failure_mlu']['p95']:.6f}")
    print(f"  p99 expected scenario MLU: {summary['expected_failure_mlu']['p99']:.6f}")
    print(f"  avg disconnected fraction: {summary['expected_disconnected']['avg']:.6f}")
    print(f"  avg failure prediction loss: {summary['failure_prediction_loss']['avg']:.6f}")
    print(f"  avg failure prediction accuracy: {summary['failure_prediction_accuracy']['avg']:.6f}")


def _write_test_stats_file(stats_path, tracker, skipped):
    """
    Writes aggregate test statistics for all core metrics, not just loss.
    """
    summary = _summarize_dynamic_metrics(tracker)
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w") as f:
        f.write("Dynamic Test Aggregate Metrics\n")
        f.write("Count: " + str(len(tracker["loss"])) + "\n")
        f.write("Skipped: " + str(skipped) + "\n\n")

        for key in DYNAMIC_METRIC_KEYS:
            f.write(key + "\n")
            for stat_name in ["avg", "median", "p25", "p75", "p90", "p95", "p99", "min", "max"]:
                f.write(f"  {stat_name}: {summary[key][stat_name]}\n")
            f.write("\n")

        mean_raw = summary["raw_mlu"]["avg"]
        mean_expected = summary["expected_failure_mlu"]["avg"]
        if mean_expected and mean_expected == mean_expected and abs(mean_expected) > 1e-12:
            aggregate_score = mean_raw / mean_expected
        else:
            aggregate_score = float("nan")

        f.write("Derived metrics\n")
        f.write("  aggregate_score_raw_over_expected: " + str(aggregate_score) + "\n")
        f.write("  mean_per_sample_resilience_score: " + str(summary["resilience_score"]["avg"]) + "\n")
        f.write("\n")
        f.write("Notes\n")
        f.write("  aggregate_score_raw_over_expected = avg(raw_mlu) / avg(expected_scenario_mlu).\n")
        f.write("  mean_per_sample_resilience_score = avg(raw_mlu_i / expected_scenario_mlu_i).\n")
        f.write("  The tqdm score is a running average in this updated file, not just the last sample.\n")


def train_dynamic(model, props, train_dl, optimizer, epoch, n_epochs):
    tracker = _new_dynamic_metric_tracker()
    skipped = 0

    # Used by validate_dynamic() so validation rows can be tagged with the epoch
    # even though run_harp.py currently calls validate_dynamic without epoch args.
    props._dynamic_epoch = epoch + 1
    props._dynamic_n_epochs = n_epochs

    with tqdm(train_dl) as tepoch:
        tepoch.set_description(f"Dynamic Epoch {epoch + 1}/{n_epochs}")

        for sample in tepoch:
            sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

            optimizer.zero_grad(set_to_none=True)

            model_output = run_model_on_dynamic_sample(model, props, sample)
            predicted, pred_splits, failure_logits = unpack_model_output(model_output)

            if not torch.isfinite(predicted).all():
                skipped += 1
                print("[WARN] Non-finite predicted output detected. Skipping batch.")
                continue

            try:
                (
                    loss,
                    loss_value,
                    current_term_value,
                    raw_mlu_value,
                    total_util_value,
                    tm_sum_value,
                    split_loss_value,
                    expected_failure_mlu_value,
                    resilience_score_value,
                    backup_penalty_value,
                    expected_disconnected_value,
                    failure_prediction_loss_value,
                    failure_prediction_accuracy_value,
                ) = dynamic_mlu_loss(
                    predicted,
                    sample,
                    pred_splits=pred_splits,
                    failure_logits=failure_logits,
                    use_opt=getattr(props, "use_dynamic_opt", True),
                    split_loss_weight=getattr(props, "split_loss_weight", 0.0),
                    use_resilience_objective=getattr(props, "use_resilience_objective", False),
                    resilience_loss_weight=getattr(props, "resilience_loss_weight", 0.0),
                    failure_prediction_loss_weight=getattr(props, "failure_prediction_loss_weight", 0.0),
                    soft_backup_penalty_weight=getattr(props, "soft_backup_penalty_weight", 0.0),
                    local_rescale=getattr(props, "future_local_rescale", getattr(props, "candidate_local_rescale", True)),
                    disconnected_penalty=getattr(props, "disconnected_penalty", 100.0),
                    num_paths_per_pair=props.num_paths_per_pair,
                    num_failure_scenarios=getattr(props, "num_failure_scenarios", 5),
                    scenario_probability_loss_weight=getattr(props, "scenario_probability_loss_weight", 1.0),
                    scenario_capacity_mode=getattr(props, "scenario_capacity_mode", "hard"),
                    detach_scenario_probabilities=getattr(props, "detach_scenario_probabilities", True),
                    detach_scenario_capacities=getattr(props, "detach_scenario_capacities", True),
                )
            except RuntimeError as exc:
                skipped += 1
                print(f"[WARN] {exc}. Skipping batch.")
                continue

            if not torch.isfinite(loss):
                skipped += 1
                print("[WARN] Non-finite dynamic loss detected. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            record = _build_dynamic_metric_record(
                loss_value=loss_value,
                current_term_value=current_term_value,
                raw_mlu_value=raw_mlu_value,
                total_util_value=total_util_value,
                tm_sum_value=tm_sum_value,
                split_loss_value=split_loss_value,
                expected_failure_mlu_value=expected_failure_mlu_value,
                resilience_score_value=resilience_score_value,
                backup_penalty_value=backup_penalty_value,
                expected_disconnected_value=expected_disconnected_value,
                failure_prediction_loss_value=failure_prediction_loss_value,
                failure_prediction_accuracy_value=failure_prediction_accuracy_value,
            )
            _append_dynamic_metric_record(tracker, record)

            tepoch.set_postfix(**_running_postfix(tracker, skipped))

    if len(tracker["loss"]) == 0:
        return float("nan")

    _append_stage_summary_csv(
        props=props,
        stage="train",
        tracker=tracker,
        skipped=skipped,
        epoch=epoch + 1,
        n_epochs=n_epochs,
    )
    _print_stage_summary(f"Dynamic Train Epoch {epoch + 1}/{n_epochs}", tracker, skipped)

    return _safe_mean(tracker["loss"])

def validate_dynamic(model, props, val_dl):
    tracker = _new_dynamic_metric_tracker()
    skipped = 0

    epoch = getattr(props, "_dynamic_epoch", None)
    n_epochs = getattr(props, "_dynamic_n_epochs", None)

    with torch.no_grad():
        with tqdm(val_dl) as vals:
            vals.set_description("Dynamic Validation")

            for sample in vals:
                sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

                model_output = run_model_on_dynamic_sample(model, props, sample)
                predicted, pred_splits, failure_logits = unpack_model_output(model_output)

                if not torch.isfinite(predicted).all():
                    skipped += 1
                    print("[WARN] Non-finite predicted output detected during validation. Skipping batch.")
                    continue

                try:
                    (
                        loss,
                        loss_value,
                        current_term_value,
                        raw_mlu_value,
                        total_util_value,
                        tm_sum_value,
                        split_loss_value,
                        expected_failure_mlu_value,
                        resilience_score_value,
                        backup_penalty_value,
                        expected_disconnected_value,
                        failure_prediction_loss_value,
                        failure_prediction_accuracy_value,
                    ) = dynamic_mlu_loss(
                        predicted,
                        sample,
                        pred_splits=pred_splits,
                        failure_logits=failure_logits,
                        use_opt=getattr(props, "use_dynamic_opt", True),
                        split_loss_weight=getattr(props, "split_loss_weight", 0.0),
                        use_resilience_objective=getattr(props, "use_resilience_objective", False),
                        resilience_loss_weight=getattr(props, "resilience_loss_weight", 0.0),
                        failure_prediction_loss_weight=getattr(props, "failure_prediction_loss_weight", 0.0),
                        soft_backup_penalty_weight=getattr(props, "soft_backup_penalty_weight", 0.0),
                        local_rescale=getattr(props, "future_local_rescale", getattr(props, "candidate_local_rescale", True)),
                        disconnected_penalty=getattr(props, "disconnected_penalty", 100.0),
                        num_paths_per_pair=props.num_paths_per_pair,
                        num_failure_scenarios=getattr(props, "num_failure_scenarios", 5),
                        scenario_probability_loss_weight=getattr(props, "scenario_probability_loss_weight", 1.0),
                        scenario_capacity_mode=getattr(props, "scenario_capacity_mode", "hard"),
                        detach_scenario_probabilities=getattr(props, "detach_scenario_probabilities", True),
                        detach_scenario_capacities=getattr(props, "detach_scenario_capacities", True),
                    )
                except RuntimeError as exc:
                    skipped += 1
                    print(f"[WARN] {exc}. Skipping validation batch.")
                    continue

                if not torch.isfinite(loss):
                    skipped += 1
                    print("[WARN] Non-finite dynamic validation loss detected. Skipping batch.")
                    continue

                record = _build_dynamic_metric_record(
                    loss_value=loss_value,
                    current_term_value=current_term_value,
                    raw_mlu_value=raw_mlu_value,
                    total_util_value=total_util_value,
                    tm_sum_value=tm_sum_value,
                    split_loss_value=split_loss_value,
                    expected_failure_mlu_value=expected_failure_mlu_value,
                    resilience_score_value=resilience_score_value,
                    backup_penalty_value=backup_penalty_value,
                    expected_disconnected_value=expected_disconnected_value,
                    failure_prediction_loss_value=failure_prediction_loss_value,
                    failure_prediction_accuracy_value=failure_prediction_accuracy_value,
                )
                _append_dynamic_metric_record(tracker, record)

                vals.set_postfix(**_running_postfix(tracker, skipped))

    if len(tracker["loss"]) == 0:
        return float("nan")

    _append_stage_summary_csv(
        props=props,
        stage="validation",
        tracker=tracker,
        skipped=skipped,
        epoch=epoch,
        n_epochs=n_epochs,
    )
    label = "Dynamic Validation"
    if epoch is not None and n_epochs is not None:
        label += f" Epoch {epoch}/{n_epochs}"
    _print_stage_summary(label, tracker, skipped)

    return _safe_mean(tracker["loss"])

def test_dynamic(model, props, test_dl, values_path, stats_path):
    tracker = _new_dynamic_metric_tracker()
    skipped = 0

    values_path = Path(values_path)
    values_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        with tqdm(test_dl) as tests:
            tests.set_description("Dynamic Test")

            with open(values_path, "w", newline="") as values_file:
                fieldnames = ["sample_idx"] + DYNAMIC_METRIC_KEYS
                writer = csv.DictWriter(values_file, fieldnames=fieldnames)
                writer.writeheader()

                for sample_idx, sample in enumerate(tests):
                    sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

                    model_output = run_model_on_dynamic_sample(model, props, sample)
                    predicted, pred_splits, failure_logits = unpack_model_output(model_output)

                    if not torch.isfinite(predicted).all():
                        skipped += 1
                        print("[WARN] Non-finite predicted output detected during test. Skipping batch.")
                        continue

                    try:
                        (
                            loss,
                            loss_value,
                            current_term_value,
                            raw_mlu_value,
                            total_util_value,
                            tm_sum_value,
                            split_loss_value,
                            expected_failure_mlu_value,
                            resilience_score_value,
                            backup_penalty_value,
                            expected_disconnected_value,
                            failure_prediction_loss_value,
                            failure_prediction_accuracy_value,
                        ) = dynamic_mlu_loss(
                            predicted,
                            sample,
                            pred_splits=pred_splits,
                            failure_logits=failure_logits,
                            use_opt=getattr(props, "use_dynamic_opt", True),
                            split_loss_weight=getattr(props, "split_loss_weight", 0.0),
                            use_resilience_objective=getattr(props, "use_resilience_objective", False),
                            resilience_loss_weight=getattr(props, "resilience_loss_weight", 0.0),
                            failure_prediction_loss_weight=getattr(props, "failure_prediction_loss_weight", 0.0),
                            soft_backup_penalty_weight=getattr(props, "soft_backup_penalty_weight", 0.0),
                            local_rescale=getattr(props, "future_local_rescale", getattr(props, "candidate_local_rescale", True)),
                            disconnected_penalty=getattr(props, "disconnected_penalty", 100.0),
                            num_paths_per_pair=props.num_paths_per_pair,
                            num_failure_scenarios=getattr(props, "num_failure_scenarios", 5),
                            scenario_probability_loss_weight=getattr(props, "scenario_probability_loss_weight", 1.0),
                            scenario_capacity_mode=getattr(props, "scenario_capacity_mode", "hard"),
                            detach_scenario_probabilities=getattr(props, "detach_scenario_probabilities", True),
                            detach_scenario_capacities=getattr(props, "detach_scenario_capacities", True),
                        )
                    except RuntimeError as exc:
                        skipped += 1
                        print(f"[WARN] {exc}. Skipping test batch.")
                        continue

                    if not torch.isfinite(loss):
                        skipped += 1
                        print("[WARN] Non-finite dynamic test loss detected. Skipping batch.")
                        continue

                    record = _build_dynamic_metric_record(
                        loss_value=loss_value,
                        current_term_value=current_term_value,
                        raw_mlu_value=raw_mlu_value,
                        total_util_value=total_util_value,
                        tm_sum_value=tm_sum_value,
                        split_loss_value=split_loss_value,
                        expected_failure_mlu_value=expected_failure_mlu_value,
                        resilience_score_value=resilience_score_value,
                        backup_penalty_value=backup_penalty_value,
                        expected_disconnected_value=expected_disconnected_value,
                        failure_prediction_loss_value=failure_prediction_loss_value,
                        failure_prediction_accuracy_value=failure_prediction_accuracy_value,
                    )
                    _append_dynamic_metric_record(tracker, record)

                    writer.writerow({"sample_idx": sample_idx, **record})

                    tests.set_postfix(**_running_postfix(tracker, skipped))

    if len(tracker["loss"]) == 0:
        avg_loss = float("nan")
        print("Dynamic Test Error:\nAvg loss: nan\n")
        return avg_loss

    _append_stage_summary_csv(
        props=props,
        stage="test",
        tracker=tracker,
        skipped=skipped,
        epoch=None,
        n_epochs=None,
    )
    _print_stage_summary("Dynamic Test", tracker, skipped)
    _write_test_stats_file(stats_path, tracker, skipped)

    summary = _summarize_dynamic_metrics(tracker)
    avg_loss = summary["loss"]["avg"]
    avg_current = summary["current_term"]["avg"]
    avg_raw_mlu = summary["raw_mlu"]["avg"]
    avg_expected_candidate = summary["expected_failure_mlu"]["avg"]
    avg_score = summary["resilience_score"]["avg"]

    print("Dynamic Test Error:")
    print(f"Avg combined loss: {avg_loss:>8f}")
    print(f"Avg current term:  {avg_current:>8f}")
    print(f"Avg raw MLU:       {avg_raw_mlu:>8f}")
    print(f"Avg expected scenario MLU:  {avg_expected_candidate:>8f}")
    print(f"Avg score:         {avg_score:>8f}")
    print(f"Median score:      {summary['resilience_score']['median']:>8f}")
    print(f"P95 expected scenario MLU:  {summary['expected_failure_mlu']['p95']:>8f}")
    print(f"Stats written to:  {stats_path}")
    print(f"Per-sample values: {values_path}\n")

    return avg_loss

def create_dataloaders(props, batch_size_dl, training=True, shuffle=True):
    ds_list = []
    dl_list = []

    if training:
        clusters = props.train_clusters
        start_indices = props.train_start_indices
        end_indices = props.train_end_indices
    else:
        clusters = props.val_clusters
        start_indices = props.val_start_indices
        end_indices = props.val_end_indices

    for clstr, start, end in zip(clusters, start_indices, end_indices):
        dataset = DM_Dataset_within_Cluster(props, clstr, start, end)
        dl = DataLoader(dataset, batch_size=batch_size_dl, shuffle=shuffle)
        ds_list.append(dataset)
        dl_list.append(dl)

    return ds_list, dl_list


def train(model, props, train_ds_list, train_dl_list, optimizer, epoch, n_epochs):
    for i in range(len(train_ds_list)):
        train_dataset = train_ds_list[i]
        train_dl = train_dl_list[i]

        train_dataset.pte = train_dataset.pte.to(device=props.device, dtype=props.dtype)
        train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device=props.device)

        move_to_device(train_dataset.edge_ids_dict_tensor, props.device)
        move_to_device(train_dataset.original_pos_edge_ids_dict_tensor, props.device)

        with tqdm(train_dl) as tepoch:
            loss_sum = 0
            loss_count = 0

            for i, inputs in enumerate(tepoch):
                optimizer.zero_grad()
                tepoch.set_description(f"Epoch {epoch + 1}/{n_epochs}")

                node_features, capacities, tms, tms_pred, opt = inputs

                if not props.dynamic:
                    node_features = node_features[:1]
                    capacities = capacities[:1]

                node_features = node_features.to(device=props.device, dtype=props.dtype)
                capacities = capacities.to(device=props.device, dtype=props.dtype)
                tms = tms.to(device=props.device, dtype=props.dtype)
                opt = opt.to(device=props.device, dtype=props.dtype)

                if props.pred:
                    tms_pred = tms_pred.to(device=props.device, dtype=props.dtype)

                    predicted = model(
                        props,
                        node_features,
                        train_dataset.edge_index,
                        capacities,
                        train_dataset.padded_edge_ids_per_path,
                        tms,
                        tms_pred,
                        train_dataset.pte,
                        train_dataset.edge_ids_dict_tensor,
                        train_dataset.original_pos_edge_ids_dict_tensor,
                    )
                else:
                    predicted = model(
                        props,
                        node_features,
                        train_dataset.edge_index,
                        capacities,
                        train_dataset.padded_edge_ids_per_path,
                        tms,
                        tms,
                        train_dataset.pte,
                        train_dataset.edge_ids_dict_tensor,
                        train_dataset.original_pos_edge_ids_dict_tensor,
                    )

                loss, loss_val = loss_mlu(predicted, opt)
                loss.backward()
                optimizer.step()

                loss_sum += loss_val
                loss_count += 1
                loss_avg = loss_sum / loss_count
                tepoch.set_postfix(loss=loss_avg)

        train_dataset.pte = train_dataset.pte.to(device="cpu")
        train_dataset.padded_edge_ids_per_path = train_dataset.padded_edge_ids_per_path.to(device="cpu")

        move_to_device(train_dataset.edge_ids_dict_tensor, "cpu")
        move_to_device(train_dataset.original_pos_edge_ids_dict_tensor, "cpu")


def validate(model, props, val_ds, val_dl):
    val_norm_mlu = []

    with torch.no_grad():
        with tqdm(val_dl) as vals:
            for i, inputs in enumerate(vals):
                node_features, capacities, tms, tms_pred, opt = inputs

                node_features = node_features.to(device=props.device, dtype=props.dtype)
                capacities = capacities.to(device=props.device, dtype=props.dtype)
                tms = tms.to(device=props.device, dtype=props.dtype)
                tms_pred = tms_pred.to(device=props.device, dtype=props.dtype)
                opt = opt.to(device=props.device, dtype=props.dtype)

                if props.pred:
                    tms_pred = tms_pred.to(device=props.device, dtype=props.dtype)

                    predicted = model(
                        props,
                        node_features,
                        val_ds.edge_index,
                        capacities,
                        val_ds.padded_edge_ids_per_path,
                        tms,
                        tms_pred,
                        val_ds.pte,
                        val_ds.edge_ids_dict_tensor,
                        val_ds.original_pos_edge_ids_dict_tensor,
                    )
                else:
                    predicted = model(
                        props,
                        node_features,
                        val_ds.edge_index,
                        capacities,
                        val_ds.padded_edge_ids_per_path,
                        tms,
                        tms,
                        val_ds.pte,
                        val_ds.edge_ids_dict_tensor,
                        val_ds.original_pos_edge_ids_dict_tensor,
                    )

                val_loss, value_loss_value = loss_mlu(predicted, opt)
                val_norm_mlu.append(value_loss_value)

    return val_norm_mlu