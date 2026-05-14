import statistics

import torch


def _canonical_edge(u, v):
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


def _normalize_risk(risk):
    risk_sum = risk.sum()
    if float(risk_sum.detach().cpu()) <= 0.0:
        num_edges = risk.numel()
        return torch.full(
            (num_edges,),
            1.0 / max(num_edges, 1),
            device=risk.device,
            dtype=risk.dtype,
        )

    return risk / risk_sum.clamp_min(1e-12)


def final_edge_risk_from_history(
    sample,
    final_edge_index,
    prior=1e-3,
    recency_power=2.0,
):
    """
    Extrapolate per-directed-edge scenario probabilities from failure history.

    Dynamic Abilene stores failures as undirected edges in
    sample["metadata"]["failed_by_t"]. We map those historical failures onto
    the directed edges present at the final timestep and build a binary history
    for each final link. The next-step failure score is a recency-weighted
    one-step linear extrapolation of that binary sequence, rather than an
    average over all link-failure scenarios. A small positive prior keeps every
    final link possible when the short history contains no usable signal.
    """
    num_edges = final_edge_index.shape[1]
    device = final_edge_index.device
    dtype = torch.float32

    prior = max(float(prior), 0.0)
    risk = torch.full((num_edges,), prior, device=device, dtype=dtype)

    metadata = sample.get("metadata", {})
    failed_by_t = metadata.get("failed_by_t", [])

    if not failed_by_t:
        return _normalize_risk(risk)

    final_edges = [
        _canonical_edge(final_edge_index[0, i].item(), final_edge_index[1, i].item())
        for i in range(num_edges)
    ]

    num_timesteps = len(failed_by_t)
    failure_history = torch.zeros(
        (num_edges, num_timesteps),
        device=device,
        dtype=dtype,
    )

    for t, failed_edges in enumerate(failed_by_t):
        failed_set = set()
        for edge in failed_edges:
            if len(edge) != 2:
                continue
            failed_set.add(_canonical_edge(edge[0], edge[1]))

        if not failed_set:
            continue

        for edge_idx, final_edge in enumerate(final_edges):
            if final_edge in failed_set:
                failure_history[edge_idx, t] = 1.0

    if num_timesteps == 1:
        extrapolated = failure_history[:, 0]
    else:
        time = torch.arange(num_timesteps, device=device, dtype=dtype)
        weights = ((time + 1.0) / float(num_timesteps)).pow(float(recency_power))
        weights = weights / weights.sum().clamp_min(1e-12)

        x_mean = (weights * time).sum()
        centered_time = time - x_mean
        y_mean = (failure_history * weights.view(1, -1)).sum(dim=-1)
        covariance = (
            failure_history - y_mean.view(-1, 1)
        ).mul(weights.view(1, -1)).mul(centered_time.view(1, -1)).sum(dim=-1)
        variance = (weights * centered_time.pow(2)).sum().clamp_min(1e-12)
        slope = covariance / variance

        next_time = torch.tensor(float(num_timesteps), device=device, dtype=dtype)
        extrapolated = y_mean + slope * (next_time - x_mean)

    risk = risk + extrapolated.clamp(0.0, 1.0)

    return _normalize_risk(risk)


def select_scenario_edges(edge_probs, scenario_top_k=0):
    """
    Select scenario edges and renormalized probabilities.

    scenario_top_k <= 0 means use all final edges.
    """
    num_edges = edge_probs.numel()

    if scenario_top_k is None or scenario_top_k <= 0 or scenario_top_k >= num_edges:
        indices = torch.arange(num_edges, device=edge_probs.device)
        probs = edge_probs
    else:
        probs, indices = torch.topk(edge_probs, k=scenario_top_k)

    probs = probs / probs.sum().clamp_min(1e-12)
    return indices, probs


def scenario_mlus_for_single_link_degradation(
    data_on_links,
    capacities,
    scenario_edge_indices,
    failure_capacity_fraction=0.25,
):
    """
    Compute MLU under single-link degradation scenarios.

    Each scenario degrades one final directed edge to
    failure_capacity_fraction * original_capacity while keeping the learned
    split fixed. This is a differentiable stress metric, not a post-failure LP.
    """
    if data_on_links.dim() != 2 or capacities.dim() != 2:
        raise ValueError(
            "data_on_links and capacities must have shape [B, E]. "
            f"Got {tuple(data_on_links.shape)} and {tuple(capacities.shape)}"
        )

    if data_on_links.shape != capacities.shape:
        raise ValueError(
            "data_on_links/capacities shape mismatch: "
            f"{tuple(data_on_links.shape)} vs {tuple(capacities.shape)}"
        )

    if not 0.0 < failure_capacity_fraction <= 1.0:
        raise ValueError("--failure_capacity_fraction must be in (0, 1]")

    scenario_mlus = []

    for edge_idx in scenario_edge_indices.tolist():
        scenario_caps = capacities.clone()
        scenario_caps[:, edge_idx] = (
            scenario_caps[:, edge_idx] * failure_capacity_fraction
        ).clamp_min(1e-6)

        scenario_utils = data_on_links / scenario_caps
        scenario_mlus.append(scenario_utils.max(dim=-1).values)

    return torch.stack(scenario_mlus, dim=-1)


def resilience_loss_from_details(
    details,
    sample,
    current_weight=1.0,
    resilience_weight=0.25,
    worst_case_weight=0.5,
    failure_capacity_fraction=0.25,
    risk_prior=1e-3,
    risk_recency_power=2.0,
    scenario_top_k=0,
):
    """
    Combined myopic + future-failure stress objective.

    current_norm = current final-timestep MLU / current Gurobi optimum
    expected_failure_norm = expected single-link-degradation MLU / current opt
    worst_failure_norm = worst selected degradation MLU / current opt

    combined = current_weight * current_norm
             + resilience_weight * ((1-worst_case_weight) * expected
                                    + worst_case_weight * worst)
    """
    edges_util = details["edges_util"].clamp_min(0.0)
    data_on_links = details["data_on_links"].clamp_min(0.0)
    capacities = details["capacities"].clamp_min(1e-6)
    edge_index = details["edge_index"]

    opt = sample["opt"]
    if not torch.is_tensor(opt):
        opt = torch.tensor(opt, device=edges_util.device, dtype=edges_util.dtype)
    else:
        opt = opt.to(device=edges_util.device, dtype=edges_util.dtype)
    opt = opt.reshape(()).clamp_min(1e-12)

    current_mlu = edges_util.max()
    current_norm = current_mlu / opt

    edge_probs = final_edge_risk_from_history(
        sample=sample,
        final_edge_index=edge_index,
        prior=risk_prior,
        recency_power=risk_recency_power,
    ).to(device=edges_util.device, dtype=edges_util.dtype)

    scenario_edge_indices, scenario_probs = select_scenario_edges(
        edge_probs,
        scenario_top_k=scenario_top_k,
    )
    scenario_probs = scenario_probs.to(device=edges_util.device, dtype=edges_util.dtype)

    scenario_mlus = scenario_mlus_for_single_link_degradation(
        data_on_links=data_on_links,
        capacities=capacities,
        scenario_edge_indices=scenario_edge_indices,
        failure_capacity_fraction=failure_capacity_fraction,
    )

    scenario_norms = scenario_mlus / opt
    expected_failure_norm = (scenario_norms * scenario_probs.view(1, -1)).sum(dim=-1)
    worst_failure_norm = scenario_norms.max(dim=-1).values

    robust_term = (
        (1.0 - worst_case_weight) * expected_failure_norm
        + worst_case_weight * worst_failure_norm
    ).mean()

    loss = current_weight * current_norm + resilience_weight * robust_term

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite resilience loss computed")

    return {
        "loss": loss,
        "combined_value": float(loss.detach().cpu()),
        "current_norm": float(current_norm.detach().cpu()),
        "current_raw_mlu": float(current_mlu.detach().cpu()),
        "expected_failure_norm": float(expected_failure_norm.mean().detach().cpu()),
        "worst_failure_norm": float(worst_failure_norm.max().detach().cpu()),
        "num_scenarios": int(scenario_edge_indices.numel()),
    }


def percentile(sorted_values, fraction):
    if len(sorted_values) == 0:
        return float("nan")
    return sorted_values[int(len(sorted_values) * fraction)]


def write_distribution_stats(values, skipped, path):
    dists = [float(v) for v in values]
    dists.sort()

    with open(path, "w") as f:
        if dists:
            f.write("Average: " + str(statistics.mean(dists)) + "\n")
            f.write("Median: " + str(percentile(dists, 0.5)) + "\n")
            f.write("25TH: " + str(percentile(dists, 0.25)) + "\n")
            f.write("75TH: " + str(percentile(dists, 0.75)) + "\n")
            f.write("90TH: " + str(percentile(dists, 0.90)) + "\n")
            f.write("95TH: " + str(percentile(dists, 0.95)) + "\n")
            f.write("99TH: " + str(percentile(dists, 0.99)) + "\n")
            f.write("100TH: " + str(dists[-1]) + "\n")
        else:
            f.write("Average: nan\n")
        f.write("Skipped: " + str(skipped) + "\n")
