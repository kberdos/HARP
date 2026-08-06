"""
Shared Monte Carlo topology candidate generation for HARP/GRATE dynamic
failure evaluation.

This module intentionally does NOT read or use future_capacities. Candidate
scenarios are generated from the clean/nominal topology reconstructed from the
current observed capacity state and history metadata.

Unlike the older hand-built candidate list, this file builds an external
Monte Carlo distribution over full next-step topology states. It conditions on
recent/current capacity multipliers, samples many possible future topology
states, collapses duplicate states, and returns the most likely states with
probabilities estimated from sample frequency.

Final evaluation should compare every method against the same clean-topology
baseline and the same shared external topology distribution.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch

EdgeKey = Tuple[int, int]


LEVELS_5 = [1.0, 0.75, 0.50, 0.25, 0.0]

# Transition prior used when the short sample history does not contain enough
# empirical transitions. Rows are P(next_multiplier | current_multiplier).
# These are deliberately conservative: most edges persist, but healthy edges can
# degrade/fail, impaired edges can persist/worsen/recover, and failed edges can
# persist or recover.
DEFAULT_TRANSITION_PRIOR: Dict[float, Dict[float, float]] = {
    1.0: {1.0: 0.950, 0.75: 0.025, 0.50: 0.010, 0.25: 0.005, 0.0: 0.010},
    0.75: {1.0: 0.150, 0.75: 0.650, 0.50: 0.150, 0.25: 0.025, 0.0: 0.025},
    0.50: {1.0: 0.100, 0.75: 0.150, 0.50: 0.550, 0.25: 0.100, 0.0: 0.100},
    0.25: {1.0: 0.100, 0.75: 0.050, 0.50: 0.150, 0.25: 0.450, 0.0: 0.250},
    0.0: {1.0: 0.100, 0.75: 0.050, 0.50: 0.050, 0.25: 0.100, 0.0: 0.700},
}


def canonical_edge_key(u: int, v: int) -> EdgeKey:
    return tuple(sorted((int(u), int(v))))


def _edge_list_from_edge_index(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise RuntimeError(f"Expected edge_index [2,E], got {tuple(edge_index.shape)}")
    return [(int(u), int(v)) for u, v in edge_index.t().detach().cpu().tolist()]


def _final_edge_index_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    edge_index = sample["edge_index"]
    if isinstance(edge_index, list):
        return edge_index[-1]
    return edge_index


def _final_capacities_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    capacities = sample["capacities"]

    if isinstance(capacities, list):
        return capacities[-1]

    if not torch.is_tensor(capacities):
        raise TypeError(f"sample['capacities'] must be a tensor or list, got {type(capacities)}")

    # Compact shared-cache format: [1, T, E]
    if capacities.dim() == 3:
        return capacities[:, -1, :]

    # Already final/current: [1, E] or [T, E]. In the usual hydrated eval path,
    # capacities is a list, so this branch is mainly for robustness.
    if capacities.dim() == 2:
        return capacities

    # [E]
    if capacities.dim() == 1:
        return capacities

    raise RuntimeError(f"Unsupported capacities shape: {tuple(capacities.shape)}")


def _parse_metadata_edge_tuple(raw_edge: Any) -> EdgeKey:
    # metadata may store edge as tuple/list, e.g. (3, 5) or [3, 5]
    if isinstance(raw_edge, (list, tuple)) and len(raw_edge) == 2:
        return canonical_edge_key(raw_edge[0], raw_edge[1])
    raise RuntimeError(f"Could not parse edge key from metadata value: {raw_edge!r}")


def _state_from_metadata_entries(entries: Iterable[Any]) -> Dict[EdgeKey, float]:
    """
    Converts metadata['capacity_multipliers_by_t'][t] entries into a dict.

    Expected entry format from generate_dynamic_failures.py:
      [((u, v), multiplier), ...]
    or JSON-ish equivalent:
      [[[u, v], multiplier], ...]

    Missing edges are healthy with multiplier 1.0.
    """
    state: Dict[EdgeKey, float] = {}
    for rec in entries or []:
        if not isinstance(rec, (list, tuple)) or len(rec) != 2:
            continue
        edge_raw, multiplier_raw = rec
        edge = _parse_metadata_edge_tuple(edge_raw)
        state[edge] = max(0.0, min(1.0, float(multiplier_raw)))
    return state


def _history_states_from_sample(sample: Dict[str, Any]) -> List[Dict[EdgeKey, float]]:
    meta = sample.get("metadata", {}) or {}
    by_t = meta.get("capacity_multipliers_by_t", None) or []
    return [_state_from_metadata_entries(entries) for entries in by_t]


def current_state_from_sample(sample: Dict[str, Any], final_edge_list: List[Tuple[int, int]]) -> Dict[EdgeKey, float]:
    """
    Returns the current observed multiplier state at the final timestep.

    Missing edges are treated as healthy by _current_multiplier().
    """
    history = _history_states_from_sample(sample)
    if not history:
        return {}
    return history[-1]


def recent_history_counts_from_sample(sample: Dict[str, Any], final_edges: List[EdgeKey]) -> Dict[EdgeKey, int]:
    """
    Counts how often each physical/undirected edge was impaired in the history.
    This is used only for candidate ranking/scoring, not for reading the future.
    """
    history = _history_states_from_sample(sample)
    final_edge_set = set(final_edges)
    counts: Dict[EdgeKey, int] = {e: 0 for e in final_edges}

    for state in history:
        for e, m in state.items():
            if e in final_edge_set and float(m) < 1.0 - 1e-8:
                counts[e] = counts.get(e, 0) + 1

    return counts


def _current_multiplier(current_state: Dict[EdgeKey, float], edge: EdgeKey) -> float:
    return max(0.0, min(1.0, float(current_state.get(edge, 1.0))))


def _snap_to_level(value: float, levels: List[float] = LEVELS_5) -> float:
    value = max(0.0, min(1.0, float(value)))
    return float(min(levels, key=lambda x: abs(float(x) - value)))


def _next_lower_level(current_multiplier: float, levels: List[float] = LEVELS_5) -> float:
    current_multiplier = max(0.0, min(1.0, float(current_multiplier)))
    for level in levels:
        if level < current_multiplier - 1e-8:
            return float(level)
    return 0.0


def _candidate_signature(multipliers: List[float]) -> Tuple[float, ...]:
    return tuple(round(float(x), 6) for x in multipliers)


def _state_signature_by_edges(state: Dict[EdgeKey, float], edges: List[EdgeKey]) -> Tuple[float, ...]:
    return tuple(round(float(state.get(e, 1.0)), 6) for e in edges)


def _node_to_incident_edges(final_edges: List[EdgeKey]) -> Dict[int, List[EdgeKey]]:
    out: Dict[int, set] = {}
    for u, v in final_edges:
        out.setdefault(int(u), set()).add((int(u), int(v)))
        out.setdefault(int(v), set()).add((int(u), int(v)))
    return {n: sorted(edges) for n, edges in out.items()}


def _make_base_capacities_and_current_multipliers(
    current_capacities: torch.Tensor,
    final_edge_list: List[Tuple[int, int]],
    current_state: Dict[EdgeKey, float],
) -> Tuple[List[float], List[float]]:
    """
    Reconstructs clean/nominal capacities.

    Generator convention:
      current_capacity = clean_capacity * current_multiplier

    Therefore:
      clean_capacity = current_capacity / current_multiplier

    If a fully failed edge is still present with current_multiplier=0, clean
    capacity is not recoverable from that value alone. In that rare case, this
    function falls back to treating the observed capacity as the clean capacity
    to avoid division by zero. In the normal generated datasets, fully failed
    edges should be removed from the current graph.
    """
    current_caps = current_capacities.detach().cpu().reshape(-1).float().tolist()

    if len(current_caps) != len(final_edge_list):
        raise RuntimeError(
            f"current_capacities length {len(current_caps)} != final edge count {len(final_edge_list)}"
        )

    base_caps: List[float] = []
    current_mults: List[float] = []

    for cap, (u, v) in zip(current_caps, final_edge_list):
        e = canonical_edge_key(u, v)
        cur_m = _current_multiplier(current_state, e)

        if cur_m <= 1e-8:
            cur_m_for_base = 1.0
        else:
            cur_m_for_base = cur_m

        current_mults.append(cur_m)
        base_caps.append(float(cap) / max(cur_m_for_base, 1e-8))

    return base_caps, current_mults


def get_clean_capacities_from_sample(sample: Dict[str, Any]) -> torch.Tensor:
    """
    Returns the clean/nominal capacity vector aligned with the final/current
    edge list.

    Returned shape:
      [E]

    This is the clean topology baseline used as the numerator for the final
    resilience metric.
    """
    edge_index_final = _final_edge_index_from_sample(sample)
    final_edge_list = _edge_list_from_edge_index(edge_index_final)

    current_caps = _final_capacities_from_sample(sample)
    current_caps_1d = current_caps.reshape(-1)

    current_state = current_state_from_sample(sample, final_edge_list)
    base_caps, _ = _make_base_capacities_and_current_multipliers(
        current_capacities=current_caps_1d,
        final_edge_list=final_edge_list,
        current_state=current_state,
    )

    dtype = current_caps.dtype if torch.is_floating_point(current_caps) else torch.float32
    device = current_caps.device

    return torch.tensor(base_caps, dtype=dtype, device=device)


def _build_transition_counts(
    sample: Dict[str, Any],
    undirected_edges: List[EdgeKey],
    prior_strength: float,
) -> Dict[float, Dict[float, float]]:
    """
    Builds a smoothed Markov transition table P(next_state | current_state).

    The empirical part is estimated from the short history stored in the sample.
    The prior part prevents the table from degenerating to all-clean persistence
    when a 6-step window contains no failures.
    """
    counts: Dict[float, Dict[float, float]] = {
        cur: {nxt: float(prior_strength) * float(prob) for nxt, prob in prior.items()}
        for cur, prior in DEFAULT_TRANSITION_PRIOR.items()
    }

    history = _history_states_from_sample(sample)
    if len(history) < 2:
        return counts

    for t in range(len(history) - 1):
        cur_state = history[t]
        next_state = history[t + 1]

        for e in undirected_edges:
            cur_m = _snap_to_level(cur_state.get(e, 1.0))
            next_m = _snap_to_level(next_state.get(e, 1.0))
            counts.setdefault(cur_m, {level: 0.0 for level in LEVELS_5})
            counts[cur_m][next_m] = counts[cur_m].get(next_m, 0.0) + 1.0

    return counts


def _sample_from_transition_row(
    cur_m: float,
    transition_counts: Dict[float, Dict[float, float]],
    generator: torch.Generator,
) -> float:
    cur_m = _snap_to_level(cur_m)
    row = transition_counts.get(cur_m, DEFAULT_TRANSITION_PRIOR[cur_m])

    weights = torch.tensor([max(0.0, float(row.get(level, 0.0))) for level in LEVELS_5], dtype=torch.float64)
    if float(weights.sum().item()) <= 0.0:
        weights = torch.tensor([DEFAULT_TRANSITION_PRIOR[cur_m][level] for level in LEVELS_5], dtype=torch.float64)

    probs = weights / weights.sum().clamp_min(1e-12)
    idx = int(torch.multinomial(probs, num_samples=1, replacement=True, generator=generator).item())
    return float(LEVELS_5[idx])


def _stable_int_hash(values: Iterable[Any]) -> int:
    """
    Stable deterministic hash independent of Python's randomized hash seed.
    """
    h = 2166136261
    text = repr(list(values))
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _edge_priority_score(
    e: EdgeKey,
    candidate_m: float,
    current_state: Dict[EdgeKey, float],
    recent_counts: Dict[EdgeKey, int],
) -> float:
    recent = float(recent_counts.get(e, 0))
    current_imp = 1.0 if _current_multiplier(current_state, e) < 1.0 - 1e-8 else 0.0
    severity = 1.0 - float(candidate_m)
    return recent + 2.0 * current_imp + 4.0 * severity


def _cap_total_impaired_edges(
    state: Dict[EdgeKey, float],
    undirected_edges: List[EdgeKey],
    current_state: Dict[EdgeKey, float],
    recent_counts: Dict[EdgeKey, int],
    max_impaired_edges: int,
) -> Dict[EdgeKey, float]:
    """
    Limits candidate topology size so Monte Carlo samples resemble the realistic
    generator's small-failure regime instead of independent many-edge failures.
    """
    if max_impaired_edges <= 0:
        return {e: 1.0 for e in undirected_edges}

    impaired = [e for e in undirected_edges if float(state.get(e, 1.0)) < 1.0 - 1e-8]
    if len(impaired) <= max_impaired_edges:
        return state

    ranked = sorted(
        impaired,
        key=lambda e: (-_edge_priority_score(e, state.get(e, 1.0), current_state, recent_counts), e),
    )
    keep = set(ranked[:max_impaired_edges])

    capped = dict(state)
    for e in impaired:
        if e not in keep:
            capped[e] = 1.0
    return capped


def _maybe_apply_correlated_event(
    state: Dict[EdgeKey, float],
    undirected_edges: List[EdgeKey],
    current_state: Dict[EdgeKey, float],
    recent_counts: Dict[EdgeKey, int],
    generator: torch.Generator,
    correlated_probability: float,
    max_edges_per_event: int,
) -> Dict[EdgeKey, float]:
    """
    Adds a small node-correlated degradation event with configurable probability.
    This keeps the external distribution able to produce whole-topology states
    where more than one related physical edge changes together.
    """
    correlated_probability = max(0.0, min(1.0, float(correlated_probability)))
    if correlated_probability <= 0.0:
        return state

    draw = float(torch.rand((), generator=generator).item())
    if draw >= correlated_probability:
        return state

    node_to_edges = _node_to_incident_edges(undirected_edges)
    if not node_to_edges:
        return state

    nodes = sorted(node_to_edges.keys())
    node_weights = []
    for n in nodes:
        incident = node_to_edges[n]
        score = 1.0
        score += sum(float(recent_counts.get(e, 0)) for e in incident)
        score += 2.0 * sum(1.0 if _current_multiplier(current_state, e) < 1.0 - 1e-8 else 0.0 for e in incident)
        node_weights.append(score)

    weights = torch.tensor(node_weights, dtype=torch.float64)
    weights = weights / weights.sum().clamp_min(1e-12)
    node_idx = int(torch.multinomial(weights, num_samples=1, replacement=True, generator=generator).item())
    node = nodes[node_idx]

    incident_ranked = sorted(
        node_to_edges[node],
        key=lambda e: (-_edge_priority_score(e, state.get(e, 1.0), current_state, recent_counts), e),
    )

    out = dict(state)
    for e in incident_ranked[: max(1, int(max_edges_per_event))]:
        out[e] = min(float(out.get(e, 1.0)), 0.75)

    return out


def _state_to_directed_multipliers(
    state: Dict[EdgeKey, float],
    final_edge_list: List[Tuple[int, int]],
) -> List[float]:
    mults = []
    for u, v in final_edge_list:
        e = canonical_edge_key(u, v)
        mults.append(max(0.0, min(1.0, float(state.get(e, 1.0)))))
    return mults


def _candidate_features(
    candidate_mults: List[float],
    baseline_mults: List[float],
    observed_current_mults: List[float],
    final_edge_list: List[Tuple[int, int]],
    recent_counts: Dict[EdgeKey, int],
) -> Dict[str, Any]:
    """
    Candidate features are computed relative to the clean baseline.

    baseline_mults is normally all 1.0.
    observed_current_mults is retained only so metadata can show whether sampled
    topology changes involve recently/currently impaired edges.
    """
    changed_edges = []
    num_full = 0
    num_partial = 0
    num_recoveries = 0
    recent_history_count = 0.0
    current_impairment_count = 0
    edge_risk_sum = 0.0

    seen_undirected: set[EdgeKey] = set()

    for idx, ((u, v), base_m, cur_m, cand_m) in enumerate(
        zip(final_edge_list, baseline_mults, observed_current_mults, candidate_mults)
    ):
        e = canonical_edge_key(u, v)

        if e in seen_undirected:
            # For bidirectional WAN edges, both directed arcs change together.
            # Count the underlying physical edge once in metadata/scoring.
            continue

        seen_undirected.add(e)

        base_m = float(base_m)
        cur_m = float(cur_m)
        cand_m = float(cand_m)

        if abs(base_m - cand_m) <= 1e-8:
            continue

        recent = int(recent_counts.get(e, 0))
        is_current_impaired = cur_m < 1.0 - 1e-8

        changed_edges.append(
            {
                "edge": tuple(e),
                "baseline_multiplier": base_m,
                "current_multiplier": cur_m,
                "candidate_multiplier": cand_m,
                "recent_history_count": recent,
                "current_impaired": bool(is_current_impaired),
            }
        )

        edge_risk_sum += 1.0
        recent_history_count += float(recent)
        current_impairment_count += int(is_current_impaired)

        if cand_m <= 1e-8:
            num_full += 1
        elif cand_m < base_m - 1e-8:
            num_partial += 1
        elif cand_m > base_m + 1e-8:
            num_recoveries += 1

    return {
        "changed_edges": changed_edges,
        "num_changed_edges": len(changed_edges),
        "edge_risk_sum": float(edge_risk_sum),
        "recent_history_count": float(recent_history_count),
        "current_impairment_count": int(current_impairment_count),
        "flaky_count": 0,
        "num_full_failures": int(num_full),
        "num_partial_degradations": int(num_partial),
        "num_recoveries": int(num_recoveries),
    }


def generate_heuristic_candidates(
    sample: Dict[str, Any],
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]], torch.Tensor, List[float]]:
    """
    Generate a shared external Monte Carlo distribution over full future topology
    states from the clean/nominal topology.

    Returns:
      candidate_capacities: [C, E_final]
      candidate_multipliers: [C, E_final]
      candidate_records: list of metadata dicts
      candidate_probs: [C] probabilities renormalized over the selected states,
                       sums to 1
      raw_scores: unnormalized Monte Carlo counts

    Candidate selection:
      - Rank unique sampled topology states by Monte Carlo frequency.
      - Select the minimum number needed to reach candidate_mc_target_coverage
        (default 0.95 sampled probability mass).
      - Stop early at candidate_mc_max_candidates (default 32) if necessary.
      - candidate_records report selected/dropped mass and whether the target
        was actually reached.

    Important:
      - No future labels are used.
      - GRATE logits are not used.
      - The model is not rerun per candidate.
      - Every method can be evaluated under the same sampled topology states.
    """
    # Select the minimum number of high-frequency topology states needed to
    # explain the requested fraction of Monte Carlo probability mass, subject
    # to a hard cap so downstream evaluation cost remains bounded.
    target_coverage = float(getattr(args, "candidate_mc_target_coverage", 0.95))
    max_candidates = int(getattr(args, "candidate_mc_max_candidates", 32))

    if not (0.0 < target_coverage <= 1.0):
        raise RuntimeError("candidate_mc_target_coverage must be in (0, 1]")
    if max_candidates <= 0:
        raise RuntimeError("candidate_mc_max_candidates must be positive")

    num_rollouts = int(getattr(args, "candidate_mc_rollouts", 512))
    num_rollouts = max(max_candidates, num_rollouts)

    prior_strength = float(getattr(args, "candidate_mc_prior_strength", 20.0))
    max_impaired_edges = int(getattr(args, "candidate_mc_max_impaired_edges", 2))
    correlated_probability = float(getattr(args, "candidate_mc_correlated_probability", 0.10))
    correlated_max_edges = int(getattr(args, "candidate_mc_correlated_max_edges", 2))
    seed_base = int(getattr(args, "candidate_mc_seed", 2680))

    edge_index_final = _final_edge_index_from_sample(sample)
    final_edge_list = _edge_list_from_edge_index(edge_index_final)
    undirected_edges = sorted({canonical_edge_key(u, v) for u, v in final_edge_list})

    current_caps = _final_capacities_from_sample(sample)
    current_caps_1d = current_caps.reshape(-1)

    current_state = current_state_from_sample(sample, final_edge_list)
    recent_counts = recent_history_counts_from_sample(sample, undirected_edges)

    base_caps, observed_current_mults = _make_base_capacities_and_current_multipliers(
        current_capacities=current_caps_1d,
        final_edge_list=final_edge_list,
        current_state=current_state,
    )

    clean_baseline_mults = [1.0 for _ in final_edge_list]
    transition_counts = _build_transition_counts(
        sample=sample,
        undirected_edges=undirected_edges,
        prior_strength=prior_strength,
    )

    # Deterministic per-sample seed. This keeps candidates/probabilities stable
    # across repeated evaluations and across different models.
    current_sig = _state_signature_by_edges(current_state, undirected_edges)
    recent_sig = tuple(int(recent_counts.get(e, 0)) for e in undirected_edges)
    stable_offset = _stable_int_hash([current_sig, recent_sig, len(undirected_edges)])
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed_base + stable_offset) % (2**31 - 1))

    counts: Dict[Tuple[float, ...], int] = {}
    exemplar_states: Dict[Tuple[float, ...], Dict[EdgeKey, float]] = {}

    for _ in range(num_rollouts):
        sampled_state: Dict[EdgeKey, float] = {}

        for e in undirected_edges:
            cur_m = _current_multiplier(current_state, e)
            sampled_state[e] = _sample_from_transition_row(
                cur_m=cur_m,
                transition_counts=transition_counts,
                generator=generator,
            )

        sampled_state = _maybe_apply_correlated_event(
            state=sampled_state,
            undirected_edges=undirected_edges,
            current_state=current_state,
            recent_counts=recent_counts,
            generator=generator,
            correlated_probability=correlated_probability,
            max_edges_per_event=correlated_max_edges,
        )

        sampled_state = _cap_total_impaired_edges(
            state=sampled_state,
            undirected_edges=undirected_edges,
            current_state=current_state,
            recent_counts=recent_counts,
            max_impaired_edges=max_impaired_edges,
        )

        sig = _state_signature_by_edges(sampled_state, undirected_edges)
        counts[sig] = counts.get(sig, 0) + 1
        exemplar_states.setdefault(sig, sampled_state)

    if not counts:
        raise RuntimeError("Monte Carlo topology sampler produced no candidates")

    all_ranked_sigs = sorted(
        counts.keys(),
        key=lambda sig: (-counts[sig], sig),
    )

    # Keep adding the most frequent unique topology until the selected set
    # reaches the target sampled probability mass, or until max_candidates is
    # reached. This replaces the old fixed top-N selection rule.
    ranked_sigs: List[Tuple[float, ...]] = []
    selected_count = 0

    for sig in all_ranked_sigs:
        ranked_sigs.append(sig)
        selected_count += int(counts[sig])

        sampled_coverage = float(selected_count) / float(num_rollouts)
        if sampled_coverage >= target_coverage or len(ranked_sigs) >= max_candidates:
            break

    selected_total = float(selected_count)
    if selected_total <= 0.0:
        raise RuntimeError("Selected Monte Carlo topology candidates have zero total count")

    selected_mass = selected_total / float(num_rollouts)
    dropped_mass = max(0.0, 1.0 - selected_mass)
    target_reached = bool(selected_mass >= target_coverage - 1e-12)
    num_unique_topologies = len(all_ranked_sigs)

    candidate_capacities: List[List[float]] = []
    candidate_multipliers: List[List[float]] = []
    candidate_records: List[Dict[str, Any]] = []
    raw_scores: List[float] = []

    clean_sig = tuple(1.0 for _ in undirected_edges)
    cumulative_selected_count = 0

    for candidate_id, sig in enumerate(ranked_sigs):
        state = exemplar_states[sig]
        mults = _state_to_directed_multipliers(state, final_edge_list)
        caps = [float(base) * float(m) for base, m in zip(base_caps, mults)]

        features = _candidate_features(
            candidate_mults=mults,
            baseline_mults=clean_baseline_mults,
            observed_current_mults=observed_current_mults,
            final_edge_list=final_edge_list,
            recent_counts=recent_counts,
        )

        count = int(counts[sig])
        probability = float(count) / selected_total
        sampled_probability = float(count) / float(num_rollouts)
        cumulative_selected_count += count
        cumulative_sampled_mass = float(cumulative_selected_count) / float(num_rollouts)
        raw_scores.append(float(count))

        if sig == clean_sig:
            name = "mc_clean_no_change"
        else:
            changed = features.get("changed_edges", [])
            if len(changed) == 1:
                edge = changed[0]["edge"]
                m = float(changed[0]["candidate_multiplier"])
                name = f"mc_edge_{edge}_{m:.2f}"
            else:
                name = f"mc_topology_{candidate_id}_{len(changed)}changed"

        candidate_capacities.append(caps)
        candidate_multipliers.append(mults)
        candidate_records.append(
            {
                "candidate_id": int(candidate_id),
                "name": name,
                "candidate_base": "clean",
                "candidate_source": "monte_carlo_topology_sampler",
                "mc_count": count,
                # Conditional probability after truncating to the selected set.
                "mc_probability": probability,
                # Probability mass in the original Monte Carlo sample.
                "mc_sampled_probability": float(sampled_probability),
                "mc_cumulative_sampled_mass": float(cumulative_sampled_mass),
                "mc_rollouts": int(num_rollouts),
                "mc_unique_topologies": int(num_unique_topologies),
                "mc_selected_candidates": int(len(ranked_sigs)),
                "mc_selected_total": int(selected_total),
                "mc_selected_mass": float(selected_mass),
                "mc_dropped_mass": float(dropped_mass),
                "mc_target_coverage": float(target_coverage),
                "mc_target_reached": bool(target_reached),
                "mc_max_candidates": int(max_candidates),
                "mc_prior_strength": float(prior_strength),
                "mc_max_impaired_edges": int(max_impaired_edges),
                "mc_correlated_probability": float(correlated_probability),
                **features,
            }
        )

    candidate_probs = torch.tensor(raw_scores, dtype=torch.float64)
    candidate_probs = candidate_probs / candidate_probs.sum().clamp_min(1e-12)

    device = current_caps.device
    dtype = current_caps.dtype if torch.is_floating_point(current_caps) else torch.float32

    return (
        torch.tensor(candidate_capacities, dtype=dtype, device=device),
        torch.tensor(candidate_multipliers, dtype=dtype, device=device),
        candidate_records,
        candidate_probs,
        raw_scores,
    )