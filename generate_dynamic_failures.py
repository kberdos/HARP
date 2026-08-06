# generate_dynamic_failures.py
#
# Generic dynamic failure/capacity-state dataset generator for HARP/GRATE-style temporal topology inputs.
# This version supports:
#   - full link failures (capacity multiplier 0.0, edge removed from graph)
#   - partial link degradations (capacity multiplier in e.g. {0.75, 0.5, 0.25})
#   - heterogeneous / flaky links through heavy-tailed edge risk weights
#   - correlated node/SRG-style events
#   - maintenance-style long-duration events
#   - one global failure timeline so overlapping training windows see consistent history
#   - true next-step failure/degradation labels for learned failure prediction
#
# Assumption kept from the current project:
#   tm_pred = tm.clone()
# so this is NOT demand prediction. The controller is assumed to observe current demand
# and current capacity/topology telemetry at timestep boundaries. The model is trained
# to predict the next-step capacity/failure state from temporal history. Candidate
# stress scenarios/probabilities are deliberately generated later by evaluators,
# never stored in this dataset.

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


EdgeKey = Tuple[int, int]
FailureState = Dict[EdgeKey, float]


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def parse_topology_json(topology_data: Dict[str, Any]):
    """
    Tries to parse common HARP-style topology JSON formats.

    Expected output:
        nodes: list of node ids
        links: list of (src, dst, capacity)
    """

    # Case 1: {"nodes": [...], "links": [{"source": ..., "target": ..., "capacity": ...}, ...]}
    if "nodes" in topology_data and ("links" in topology_data or "edges" in topology_data):
        nodes_raw = topology_data["nodes"]
        links_raw = topology_data.get("links", topology_data.get("edges"))

        nodes = []
        for n in nodes_raw:
            if isinstance(n, dict):
                nodes.append(n.get("id", n.get("name", len(nodes))))
            else:
                nodes.append(n)

        links = []
        for e in links_raw:
            if isinstance(e, dict):
                src = e.get("source", e.get("src", e.get("u")))
                dst = e.get("target", e.get("dst", e.get("v")))
                cap = e.get("capacity", e.get("cap", e.get("weight", 1.0)))
            else:
                src, dst = e[0], e[1]
                cap = e[2] if len(e) > 2 else 1.0

            links.append((src, dst, float(cap)))

        return nodes, links

    # Case 2: {"topology": {"nodes": ..., "links": ...}}
    if "topology" in topology_data:
        return parse_topology_json(topology_data["topology"])

    raise ValueError(
        "Could not parse topology JSON. Print the JSON keys and edit parse_topology_json(). "
        f"Top-level keys: {list(topology_data.keys())}"
    )


def normalize_pairs(pair_data):
    """
    Converts pair data into list of (src, dst).

    Handles common cases:
        [(s, d), ...]
        [[s, d], ...]
        {"pairs": [(s, d), ...]}
    """

    if isinstance(pair_data, dict):
        if "pairs" in pair_data:
            pair_data = pair_data["pairs"]
        elif "sd_pairs" in pair_data:
            pair_data = pair_data["sd_pairs"]
        else:
            raise ValueError(f"Unknown pair dict format. Keys: {list(pair_data.keys())}")

    pairs = []
    for p in pair_data:
        pairs.append((p[0], p[1]))

    return pairs


def normalize_traffic_matrices(tm_data, pairs: List[Tuple[Any, Any]], node_to_idx: Dict[Any, int]):
    """
    Converts traffic data into shape:
        [S, num_pairs]

    Handles:
        [S, num_pairs]
        [S, N, N]
        list of matrices
    """

    tm = np.asarray(tm_data, dtype=np.float32)

    if tm.ndim == 2:
        # Already [S, num_pairs]
        if tm.shape[1] != len(pairs):
            raise ValueError(
                f"TM is 2D but second dim {tm.shape[1]} != num_pairs {len(pairs)}"
            )
        return tm

    if tm.ndim == 3:
        # [S, N, N] -> [S, num_pairs]
        S = tm.shape[0]
        out = np.zeros((S, len(pairs)), dtype=np.float32)

        for i, (src, dst) in enumerate(pairs):
            s_idx = node_to_idx[src]
            d_idx = node_to_idx[dst]
            out[:, i] = tm[:, s_idx, d_idx]

        return out

    raise ValueError(f"Unsupported traffic matrix shape: {tm.shape}")


def canonical_edge_key(u, v) -> EdgeKey:
    return tuple(sorted((int(u), int(v))))


def get_undirected_edges(links) -> List[EdgeKey]:
    return sorted({canonical_edge_key(u, v) for u, v, _ in links})


def parse_float_list(text: str) -> List[float]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError(f"Could not parse float list from: {text}")
    return vals


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise ValueError(f"At least one event weight must be positive. Got {weights}")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def weighted_choice(items: List[Any], weights: List[float]):
    if len(items) == 0:
        raise ValueError("Cannot choose from an empty list")
    total = float(sum(weights))
    if total <= 0:
        return random.choice(items)
    r = random.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return item
    return items[-1]


def weighted_sample_without_replacement(items: List[Any], weights: List[float], k: int) -> List[Any]:
    items = list(items)
    weights = [float(w) for w in weights]
    chosen = []
    k = min(k, len(items))
    for _ in range(k):
        pick = weighted_choice(items, weights)
        idx = items.index(pick)
        chosen.append(pick)
        items.pop(idx)
        weights.pop(idx)
        if not items:
            break
    return chosen


def build_edge_risk_profile(
    links,
    flaky_edge_fraction: float,
    flaky_edge_multiplier: float,
    edge_risk_sigma: float,
):
    """
    Returns stable per-edge risk weights for the whole generated dataset.

    Real WAN failures are not IID across all links: a small number of links tend to be
    much more failure-prone. This approximates that behavior with a lognormal risk
    distribution plus a few explicit flaky links.
    """

    undirected_edges = get_undirected_edges(links)
    if not undirected_edges:
        raise ValueError("No undirected edges found in topology")

    raw = np.random.lognormal(mean=0.0, sigma=max(0.0, edge_risk_sigma), size=len(undirected_edges))
    risk = {e: float(w) for e, w in zip(undirected_edges, raw)}

    n_flaky = int(round(flaky_edge_fraction * len(undirected_edges)))
    n_flaky = max(0, min(n_flaky, len(undirected_edges)))
    flaky_edges = set(random.sample(undirected_edges, k=n_flaky)) if n_flaky > 0 else set()

    for e in flaky_edges:
        risk[e] *= max(1.0, flaky_edge_multiplier)

    return risk, flaky_edges


def build_node_to_incident_edges(nodes, links):
    node_to_edges = {int(n): set() for n in nodes}
    for u, v, _ in links:
        key = canonical_edge_key(u, v)
        node_to_edges[int(u)].add(key)
        node_to_edges[int(v)].add(key)
    return {n: sorted(edges) for n, edges in node_to_edges.items()}


def choose_duration(args, event_type: str) -> int:
    if event_type == "maintenance":
        return random.randint(args.maintenance_duration_min, args.maintenance_duration_max)

    weights = {
        "short": args.short_duration_weight,
        "medium": args.medium_duration_weight,
        "long": args.long_duration_weight,
    }
    probs = normalize_weights(weights)
    bucket = weighted_choice(list(probs.keys()), list(probs.values()))

    if bucket == "short":
        return random.randint(args.short_duration_min, args.short_duration_max)
    if bucket == "medium":
        return random.randint(args.medium_duration_min, args.medium_duration_max)
    return random.randint(args.long_duration_min, args.long_duration_max)


def apply_event_to_timeline(
    timeline: List[FailureState],
    events_by_t: List[List[Dict[str, Any]]],
    start_t: int,
    duration: int,
    edge_to_multiplier: Dict[EdgeKey, float],
    event_type: str,
):
    end_t = min(len(timeline), start_t + duration)
    edge_records = []

    for e, multiplier in edge_to_multiplier.items():
        edge_records.append({"edge": tuple(e), "multiplier": float(multiplier)})

    event_record = {
        "type": event_type,
        "start_t": int(start_t),
        "end_t_exclusive": int(end_t),
        "duration": int(end_t - start_t),
        "edges": edge_records,
    }

    for t in range(start_t, end_t):
        for e, multiplier in edge_to_multiplier.items():
            # If multiple events overlap on a link, use the most severe one.
            old = timeline[t].get(e, 1.0)
            timeline[t][e] = min(float(old), float(multiplier))
        events_by_t[t].append(event_record)


def make_legacy_failure_timeline(links, T: int, args):
    """
    Backward-compatible full-failure-only timeline.
    """
    timeline: List[FailureState] = [dict() for _ in range(T)]
    events_by_t: List[List[Dict[str, Any]]] = [[] for _ in range(T)]
    undirected_edges = get_undirected_edges(links)

    for t in range(T):
        if random.random() < args.failure_probability:
            num_failures = random.randint(1, args.max_failed_edges)
            chosen = random.sample(undirected_edges, k=min(num_failures, len(undirected_edges)))
            duration = random.randint(args.min_failure_duration, args.max_failure_duration)
            apply_event_to_timeline(
                timeline=timeline,
                events_by_t=events_by_t,
                start_t=t,
                duration=duration,
                edge_to_multiplier={e: 0.0 for e in chosen},
                event_type="legacy_full",
            )

    return timeline, events_by_t, {e: 1.0 for e in undirected_edges}, set()


def make_realistic_failure_timeline(links, nodes, T: int, args):
    """
    Creates one global topology/capacity-state timeline.

    Each timestep has a dictionary:
        {undirected_edge_key: capacity_multiplier}

    multiplier semantics:
        1.0 -> healthy, omitted from dictionary
        0.75/0.5/0.25 -> partial degradation
        0.0 -> full link failure, removed from graph
    """

    timeline: List[FailureState] = [dict() for _ in range(T)]
    events_by_t: List[List[Dict[str, Any]]] = [[] for _ in range(T)]

    undirected_edges = get_undirected_edges(links)
    edge_risk, flaky_edges = build_edge_risk_profile(
        links=links,
        flaky_edge_fraction=args.flaky_edge_fraction,
        flaky_edge_multiplier=args.flaky_edge_multiplier,
        edge_risk_sigma=args.edge_risk_sigma,
    )
    edge_weights = [edge_risk[e] for e in undirected_edges]
    partial_levels = parse_float_list(args.partial_capacity_levels)
    partial_levels = [x for x in partial_levels if 0.0 < x < 1.0]
    if not partial_levels:
        raise ValueError("partial_capacity_levels must contain values strictly between 0 and 1")

    node_to_edges = build_node_to_incident_edges(nodes, links)

    event_weights = normalize_weights(
        {
            "partial": args.partial_event_weight,
            "full": args.full_event_weight,
            "correlated": args.correlated_event_weight,
            "maintenance": args.maintenance_event_weight,
            "flaky_burst": args.flaky_burst_event_weight,
        }
    )
    event_types = list(event_weights.keys())
    event_probs = list(event_weights.values())

    for t in range(T):
        if random.random() >= args.failure_probability:
            continue

        event_type = weighted_choice(event_types, event_probs)
        duration = choose_duration(args, event_type)
        edge_to_multiplier: Dict[EdgeKey, float] = {}

        if event_type == "partial":
            e = weighted_sample_without_replacement(undirected_edges, edge_weights, 1)[0]
            edge_to_multiplier[e] = random.choice(partial_levels)

        elif event_type == "full":
            n = random.randint(1, max(1, args.max_failed_edges))
            chosen = weighted_sample_without_replacement(undirected_edges, edge_weights, n)
            edge_to_multiplier = {e: 0.0 for e in chosen}

        elif event_type == "flaky_burst":
            if flaky_edges:
                e = random.choice(sorted(flaky_edges))
            else:
                e = weighted_sample_without_replacement(undirected_edges, edge_weights, 1)[0]
            # Flaky bursts are often short and repeated; sometimes full, sometimes degraded.
            if random.random() < args.flaky_full_probability:
                edge_to_multiplier[e] = 0.0
            else:
                edge_to_multiplier[e] = random.choice(partial_levels)

        elif event_type == "correlated":
            # Synthetic SRG approximation for Abilene: choose a node and affect a few incident edges.
            candidate_nodes = [n for n, inc in node_to_edges.items() if len(inc) > 0]
            node = random.choice(candidate_nodes)
            incident_edges = node_to_edges[node]
            n = random.randint(2, max(2, args.max_correlated_edges))
            chosen = random.sample(incident_edges, k=min(n, len(incident_edges)))
            if len(chosen) < 2:
                chosen = weighted_sample_without_replacement(undirected_edges, edge_weights, min(2, len(undirected_edges)))

            for e in chosen:
                if random.random() < args.correlated_full_probability:
                    edge_to_multiplier[e] = 0.0
                else:
                    edge_to_multiplier[e] = random.choice(partial_levels)

        elif event_type == "maintenance":
            # Long event. Usually one link, sometimes a small node group.
            if random.random() < args.maintenance_node_event_probability:
                candidate_nodes = [n for n, inc in node_to_edges.items() if len(inc) > 0]
                node = random.choice(candidate_nodes)
                incident_edges = node_to_edges[node]
                n = random.randint(1, max(1, args.max_maintenance_edges))
                chosen = random.sample(incident_edges, k=min(n, len(incident_edges)))
            else:
                n = random.randint(1, max(1, args.max_maintenance_edges))
                chosen = weighted_sample_without_replacement(undirected_edges, edge_weights, n)

            for e in chosen:
                if random.random() < args.maintenance_full_probability:
                    edge_to_multiplier[e] = 0.0
                else:
                    edge_to_multiplier[e] = random.choice(partial_levels)

        else:
            raise ValueError(f"Unknown event_type={event_type}")

        # For small topologies, avoid absurdly large simultaneous outages.
        if args.max_edges_per_event > 0 and len(edge_to_multiplier) > args.max_edges_per_event:
            keep = random.sample(list(edge_to_multiplier.keys()), k=args.max_edges_per_event)
            edge_to_multiplier = {e: edge_to_multiplier[e] for e in keep}

        apply_event_to_timeline(
            timeline=timeline,
            events_by_t=events_by_t,
            start_t=t,
            duration=duration,
            edge_to_multiplier=edge_to_multiplier,
            event_type=event_type,
        )

    return timeline, events_by_t, edge_risk, flaky_edges


def make_failure_timeline(links, nodes, T: int, args):
    if args.failure_model == "legacy":
        return make_legacy_failure_timeline(links, T, args)
    if args.failure_model == "realistic":
        return make_realistic_failure_timeline(links, nodes, T, args)
    raise ValueError(f"Unsupported failure_model={args.failure_model}")


def build_graph(nodes, links, edge_multipliers: Optional[FailureState] = None):
    """
    Builds a directed graph from an undirected capacity-state view.

    edge_multipliers contains canonical undirected keys:
        tuple(sorted((u, v))) -> capacity multiplier

    If multiplier == 0, the edge is removed in both directions.
    If 0 < multiplier < 1, both directions remain but with reduced capacity.
    """

    edge_multipliers = edge_multipliers or {}

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for src, dst, cap in links:
        key = canonical_edge_key(src, dst)
        multiplier = float(edge_multipliers.get(key, 1.0))
        multiplier = max(0.0, min(1.0, multiplier))

        if multiplier <= 1e-8:
            continue

        cap_t = float(cap) * multiplier

        # Add directed edge as written.
        G.add_edge(src, dst, capacity=cap_t, base_capacity=float(cap), multiplier=multiplier, weight=1.0)

        # If your topology file is undirected, this makes it bidirectional.
        # If your HARP topology is already directed, remove this line.
        G.add_edge(dst, src, capacity=cap_t, base_capacity=float(cap), multiplier=multiplier, weight=1.0)

    return G


def canonical_edges_from_graph(G: nx.DiGraph):
    """
    Creates stable local edge ids for the current timestep.

    Returns:
        edge_list: [(u, v), ...]
        edge_to_id: {(u, v): eid}
    """

    edge_list = list(G.edges())
    edge_to_id = {e: i for i, e in enumerate(edge_list)}
    return edge_list, edge_to_id


def graph_to_edge_index_and_capacities(G: nx.DiGraph, edge_list, device="cpu"):
    edge_index = []
    capacities = []

    for u, v in edge_list:
        edge_index.append([u, v])
        capacities.append(float(G[u][v].get("capacity", 1.0)))

    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
    capacities = torch.tensor(capacities, dtype=torch.float32, device=device)

    return edge_index, capacities


def make_node_features(G: nx.DiGraph, num_nodes: int, device="cpu"):
    """
    HARP GNN expects 2 node features because GNN(2, ...).

    Simple features:
        feature 0 = normalized total degree
        feature 1 = normalized incident capacity sum
    """

    degree = np.zeros(num_nodes, dtype=np.float32)
    cap_sum = np.zeros(num_nodes, dtype=np.float32)

    for n in G.nodes():
        degree[n] = G.in_degree(n) + G.out_degree(n)

    for u, v, data in G.edges(data=True):
        cap = float(data.get("capacity", 1.0))
        cap_sum[u] += cap
        cap_sum[v] += cap

    if degree.max() > 0:
        degree = degree / degree.max()

    if cap_sum.max() > 0:
        cap_sum = cap_sum / cap_sum.max()

    features = np.stack([degree, cap_sum], axis=-1)
    return torch.tensor(features, dtype=torch.float32, device=device)


def k_shortest_paths_for_pair(G: nx.DiGraph, src, dst, k: int):
    try:
        gen = nx.shortest_simple_paths(G, src, dst, weight="weight")
        paths = []

        for path in gen:
            if len(path) >= 2:
                paths.append(path)
            if len(paths) >= k:
                break

        return paths

    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def node_path_to_edge_ids(path: List[int], edge_to_id: Dict[Tuple[int, int], int]):
    edge_ids = []

    for u, v in zip(path[:-1], path[1:]):
        if (u, v) not in edge_to_id:
            return None
        edge_ids.append(edge_to_id[(u, v)])

    return edge_ids


def build_paths_for_graph(
    G: nx.DiGraph,
    pairs: List[Tuple[int, int]],
    k_paths: int,
    edge_to_id: Dict[Tuple[int, int], int],
):
    """
    Returns exactly P = num_pairs * k_paths paths.

    If a pair has fewer than k paths, repeat the last valid path.
    If a pair has zero paths, reject the sample.
    """

    all_path_edge_ids = []

    for pair_idx, (src, dst) in enumerate(pairs):
        paths = k_shortest_paths_for_pair(G, src, dst, k_paths)

        if len(paths) == 0:
            print(f"[DEBUG] No path for pair_idx={pair_idx}, src={src}, dst={dst}")
            print(f"[DEBUG] Graph nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            print(f"[DEBUG] src in graph={src in G.nodes}, dst in graph={dst in G.nodes}")
            return None

        valid_edge_paths = []

        for p in paths:
            edge_ids = node_path_to_edge_ids(p, edge_to_id)

            if edge_ids is not None and len(edge_ids) > 0:
                valid_edge_paths.append(edge_ids)

        if len(valid_edge_paths) == 0:
            print(f"[DEBUG] Node paths found but edge-id conversion failed.")
            print(f"[DEBUG] pair_idx={pair_idx}, src={src}, dst={dst}")
            print(f"[DEBUG] example paths={paths[:3]}")
            return None

        # Repeat last valid path until we have exactly K paths.
        while len(valid_edge_paths) < k_paths:
            valid_edge_paths.append(valid_edge_paths[-1])

        valid_edge_paths = valid_edge_paths[:k_paths]

        all_path_edge_ids.extend(valid_edge_paths)

    return all_path_edge_ids


def pad_path_edge_ids(all_path_edge_ids: List[List[int]], device="cpu"):
    max_len = max(len(p) for p in all_path_edge_ids)
    P = len(all_path_edge_ids)

    padded = torch.full((P, max_len), -1, dtype=torch.long, device=device)

    for i, edge_ids in enumerate(all_path_edge_ids):
        padded[i, : len(edge_ids)] = torch.tensor(edge_ids, dtype=torch.long, device=device)

    return padded


def build_paths_to_edges(all_path_edge_ids: List[List[int]], num_edges: int, device="cpu"):
    """
    Sparse tensor [P, E].
    paths_to_edges[p, e] = 1 if path p uses edge e.
    """

    row_indices = []
    col_indices = []
    values = []

    for p_idx, edge_ids in enumerate(all_path_edge_ids):
        for e_idx in edge_ids:
            row_indices.append(p_idx)
            col_indices.append(e_idx)
            values.append(1.0)

    indices = torch.tensor([row_indices, col_indices], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(len(all_path_edge_ids), num_edges),
        device=device,
    ).coalesce()


def build_tm_window(
    traffic_pair_series: np.ndarray,
    start_idx: int,
    history_len: int,
    k_paths: int,
    device="cpu",
):
    """
    traffic_pair_series:
        [S, num_pairs]

    Returns:
        tm_window: [1, T, P, 1]

    HARP expects demand repeated for each path slot.
    If pair i has demand d, then its K paths each receive d before softmax splitting.
    """

    window = traffic_pair_series[start_idx : start_idx + history_len]  # [T, num_pairs]

    repeated = np.repeat(window, repeats=k_paths, axis=1)  # [T, P]
    repeated = repeated[None, :, :, None]                  # [1, T, P, 1]

    return torch.tensor(repeated, dtype=torch.float32, device=device)


def state_to_metadata(state: FailureState):
    full = []
    partial = []
    multipliers = []
    for e, m in sorted(state.items()):
        rec = (tuple(e), float(m))
        multipliers.append(rec)
        if float(m) <= 1e-8:
            full.append(tuple(e))
        elif float(m) < 1.0:
            partial.append(rec)
    return full, partial, multipliers


def build_base_capacity_by_edge(links) -> Dict[EdgeKey, float]:
    """
    Returns undirected base capacities.
    """

    base_capacity_by_edge: Dict[EdgeKey, float] = {}
    for u, v, cap in links:
        base_capacity_by_edge[canonical_edge_key(u, v)] = float(cap)
    return base_capacity_by_edge


def multiplier_to_state_label(multiplier: float, partial_levels: List[float]) -> int:
    """
    Maps a capacity multiplier to a class label.

    Default labels when partial_levels="0.75,0.5,0.25":
        0 -> 1.00 healthy
        1 -> 0.75 capacity
        2 -> 0.50 capacity
        3 -> 0.25 capacity
        4 -> 0.00 full failure

    If a multiplier is slightly off due to floating point, choose the closest
    level in the ladder.
    """
    levels = [1.0] + sorted([x for x in partial_levels if 0.0 < x < 1.0], reverse=True) + [0.0]
    multiplier = max(0.0, min(1.0, float(multiplier)))
    distances = [abs(multiplier - level) for level in levels]
    return int(min(range(len(levels)), key=lambda i: distances[i]))


def label_semantics_for_partial_levels(partial_levels: List[float]) -> Dict[str, str]:
    levels = [1.0] + sorted([x for x in partial_levels if 0.0 < x < 1.0], reverse=True) + [0.0]
    semantics = {}
    for i, level in enumerate(levels):
        if level >= 1.0 - 1e-8:
            semantics[str(i)] = "healthy_1.0"
        elif level <= 1e-8:
            semantics[str(i)] = "full_failure_0.0"
        else:
            semantics[str(i)] = f"partial_{level:.2f}"
    return semantics


def build_true_future_tensors(
    future_state: FailureState,
    final_edge_list: List[Tuple[int, int]],
    links,
    args,
    device="cpu",
):
    """
    Builds the actual t+1 failure/capacity labels aligned with the final/current
    directed edge ids.

    This is the supervised target for the learned failure head. We only label
    edges present in the final/current graph because the current routing decision
    and its path set are aligned to that graph.
    """
    base_capacity_by_edge = build_base_capacity_by_edge(links)
    partial_levels = parse_float_list(args.partial_capacity_levels)
    partial_levels = [x for x in partial_levels if 0.0 < x < 1.0]
    if not partial_levels:
        partial_levels = [0.75, 0.5, 0.25]

    caps = []
    mults = []
    labels = []
    edge_records = []

    for u, v in final_edge_list:
        key = canonical_edge_key(u, v)
        multiplier = float(future_state.get(key, 1.0))
        multiplier = max(0.0, min(1.0, multiplier))
        base_cap = float(base_capacity_by_edge[key])

        caps.append(base_cap * multiplier)
        mults.append(multiplier)
        labels.append(multiplier_to_state_label(multiplier, partial_levels))

        if multiplier < 1.0 - 1e-8:
            edge_records.append({"edge": tuple(key), "multiplier": multiplier})

    return {
        "future_capacities": torch.tensor(caps, dtype=torch.float32, device=device),
        "future_capacity_multipliers": torch.tensor(mults, dtype=torch.float32, device=device),
        "future_edge_state_labels": torch.tensor(labels, dtype=torch.long, device=device),
        "future_metadata": {
            "edges": edge_records,
            "label_semantics": label_semantics_for_partial_levels(partial_levels),
        },
    }

def _levels_from_args(args) -> List[float]:
    partial_levels = parse_float_list(args.partial_capacity_levels)
    partial_levels = [x for x in partial_levels if 0.0 < x < 1.0]
    if not partial_levels:
        partial_levels = [0.75, 0.5, 0.25]
    return [1.0] + sorted(partial_levels, reverse=True) + [0.0]


def _candidate_multiplier_for_edge(state: FailureState, edge: EdgeKey) -> float:
    return max(0.0, min(1.0, float(state.get(edge, 1.0))))


def _candidate_signature(state: FailureState, final_edge_list: List[Tuple[int, int]]) -> Tuple[float, ...]:
    return tuple(
        round(_candidate_multiplier_for_edge(state, canonical_edge_key(u, v)), 6)
        for u, v in final_edge_list
    )


def _next_lower_level(current_multiplier: float, levels: List[float]) -> float:
    current_multiplier = max(0.0, min(1.0, float(current_multiplier)))
    for level in levels:
        if level < current_multiplier - 1e-8:
            return float(level)
    return 0.0


def _candidate_features(
    candidate_state: FailureState,
    current_state: FailureState,
    states_window: List[FailureState],
    final_edge_list: List[Tuple[int, int]],
    edge_risk: Dict[EdgeKey, float],
    flaky_edges: set,
) -> Dict[str, Any]:
    """
    Computes deterministic scoring features for a candidate state.

    These are not probabilities. They are observable/metadata features used by
    the evaluator's deterministic heuristic to assign probability weights after
    the model has already produced routing.
    """
    final_edges = sorted({canonical_edge_key(u, v) for u, v in final_edge_list})
    max_risk = max([float(edge_risk.get(e, 1.0)) for e in final_edges] + [1.0])

    recent_counts = {}
    for state in states_window:
        for e in final_edges:
            if _candidate_multiplier_for_edge(state, e) < 1.0 - 1e-8:
                recent_counts[e] = recent_counts.get(e, 0) + 1

    changed_edges = []
    edge_risk_sum = 0.0
    recent_history_count = 0.0
    current_impairment_count = 0
    flaky_count = 0
    num_full_failures = 0
    num_partial_degradations = 0
    num_recoveries = 0

    for e in final_edges:
        cur_m = _candidate_multiplier_for_edge(current_state, e)
        cand_m = _candidate_multiplier_for_edge(candidate_state, e)

        if abs(cur_m - cand_m) <= 1e-8:
            continue

        edge_risk_norm = float(edge_risk.get(e, 1.0)) / max(max_risk, 1e-12)
        recent = int(recent_counts.get(e, 0))
        is_current_impaired = cur_m < 1.0 - 1e-8
        is_flaky = e in flaky_edges

        changed_edges.append(
            {
                "edge": tuple(e),
                "current_multiplier": float(cur_m),
                "candidate_multiplier": float(cand_m),
                "edge_risk_norm": float(edge_risk_norm),
                "recent_history_count": int(recent),
                "current_impaired": bool(is_current_impaired),
                "flaky": bool(is_flaky),
            }
        )

        edge_risk_sum += edge_risk_norm
        recent_history_count += recent
        current_impairment_count += int(is_current_impaired)
        flaky_count += int(is_flaky)

        if cand_m <= 1e-8:
            num_full_failures += 1
        elif cand_m < cur_m - 1e-8:
            num_partial_degradations += 1
        elif cand_m > cur_m + 1e-8:
            num_recoveries += 1

    return {
        "changed_edges": changed_edges,
        "num_changed_edges": len(changed_edges),
        "edge_risk_sum": float(edge_risk_sum),
        "recent_history_count": float(recent_history_count),
        "current_impairment_count": int(current_impairment_count),
        "flaky_count": int(flaky_count),
        "num_full_failures": int(num_full_failures),
        "num_partial_degradations": int(num_partial_degradations),
        "num_recoveries": int(num_recoveries),
    }


def build_failure_candidate_tensors(
    current_state: FailureState,
    future_state: FailureState,
    states_window: List[FailureState],
    final_edge_list: List[Tuple[int, int]],
    links,
    edge_risk: Dict[EdgeKey, float],
    flaky_edges: set,
    args,
    device="cpu",
):
    """
    Builds shared candidate future topology states for post-hoc evaluation.

    Important: this function does NOT compute or store probabilities. Candidate
    probabilities are computed later inside eval_failure_stability.py after a
    model has produced routing. This keeps the dataset shared/fair across GRATE,
    HARP, DOTE, and TEAL.
    """
    num_candidates = int(getattr(args, "num_failure_candidates", 0))
    if num_candidates <= 0:
        return {}

    base_capacity_by_edge = build_base_capacity_by_edge(links)
    levels = _levels_from_args(args)
    final_edges = sorted({canonical_edge_key(u, v) for u, v in final_edge_list})

    recent_counts = {}
    for state in states_window:
        for e in final_edges:
            if _candidate_multiplier_for_edge(state, e) < 1.0 - 1e-8:
                recent_counts[e] = recent_counts.get(e, 0) + 1

    max_risk = max([float(edge_risk.get(e, 1.0)) for e in final_edges] + [1.0])

    def edge_priority(e: EdgeKey):
        risk = float(edge_risk.get(e, 1.0)) / max(max_risk, 1e-12)
        recent = float(recent_counts.get(e, 0)) / max(len(states_window), 1)
        current_imp = 1.0 if _candidate_multiplier_for_edge(current_state, e) < 1.0 - 1e-8 else 0.0
        flaky = 1.0 if e in flaky_edges else 0.0
        return risk + 0.5 * recent + 0.5 * current_imp + 0.25 * flaky

    ranked_edges = sorted(final_edges, key=lambda e: (-edge_priority(e), e))

    candidates: List[Tuple[str, FailureState]] = []
    seen = set()

    def add_candidate(name: str, state: FailureState):
        if len(candidates) >= num_candidates:
            return
        normalized = {canonical_edge_key(*e): max(0.0, min(1.0, float(m))) for e, m in state.items()}
        sig = _candidate_signature(normalized, final_edge_list)
        if sig in seen:
            return
        seen.add(sig)
        candidates.append((name, normalized))

    # Candidate 0: current state persists into the next timestep.
    add_candidate("current_no_change", dict(current_state))

    # Candidate 1: the realized next-step state. This is shared held-out label
    # information in the dataset, not a probability.
    add_candidate("realized_next_step", dict(future_state))

    # Candidate pool: deterministic one-edge worsen/full/recovery states.
    for e in ranked_edges:
        if len(candidates) >= num_candidates:
            break

        cur_m = _candidate_multiplier_for_edge(current_state, e)

        worsen_m = _next_lower_level(cur_m, levels)
        if worsen_m < cur_m - 1e-8:
            s = dict(current_state)
            s[e] = worsen_m
            add_candidate(f"worsen_{e}_{worsen_m:.2f}", s)

        if len(candidates) >= num_candidates:
            break

        if cur_m > 1e-8:
            s = dict(current_state)
            s[e] = 0.0
            add_candidate(f"full_failure_{e}", s)

        if len(candidates) >= num_candidates:
            break

        if cur_m < 1.0 - 1e-8:
            s = dict(current_state)
            s[e] = 1.0
            add_candidate(f"recovery_{e}", s)

    # Candidate pool: deterministic small correlated events around nodes.
    if len(candidates) < num_candidates:
        node_to_edges = build_node_to_incident_edges(list({n for edge in final_edges for n in edge}), [(u, v, 1.0) for u, v in final_edges])
        ranked_nodes = sorted(
            node_to_edges.keys(),
            key=lambda n: (-sum(edge_priority(e) for e in node_to_edges[n]), n),
        )
        for n in ranked_nodes:
            if len(candidates) >= num_candidates:
                break
            incident = sorted(node_to_edges[n], key=lambda e: (-edge_priority(e), e))[:2]
            if len(incident) < 2:
                continue
            s = dict(current_state)
            for e in incident:
                cur_m = _candidate_multiplier_for_edge(current_state, e)
                s[e] = _next_lower_level(cur_m, levels)
            add_candidate(f"correlated_node_{n}", s)

    if len(candidates) == 0:
        raise RuntimeError("No failure candidates were generated despite num_failure_candidates > 0")

    candidate_capacities = []
    candidate_multipliers = []
    candidate_records = []

    for idx, (name, state) in enumerate(candidates):
        caps = []
        mults = []
        for u, v in final_edge_list:
            key = canonical_edge_key(u, v)
            multiplier = _candidate_multiplier_for_edge(state, key)
            caps.append(float(base_capacity_by_edge[key]) * multiplier)
            mults.append(multiplier)

        features = _candidate_features(
            candidate_state=state,
            current_state=current_state,
            states_window=states_window,
            final_edge_list=final_edge_list,
            edge_risk=edge_risk,
            flaky_edges=flaky_edges,
        )

        candidate_capacities.append(caps)
        candidate_multipliers.append(mults)
        candidate_records.append(
            {
                "candidate_id": int(idx),
                "name": name,
                **features,
            }
        )

    return {
        "failure_candidate_capacities": torch.tensor(candidate_capacities, dtype=torch.float32, device=device),
        "failure_candidate_capacity_multipliers": torch.tensor(candidate_multipliers, dtype=torch.float32, device=device),
        "failure_candidate_metadata": {
            "num_candidates_requested": int(num_candidates),
            "num_candidates_saved": int(len(candidates)),
            "probability_note": "probabilities_not_saved; eval_failure_stability.py_assigns_probabilities_at_eval_time_using_GRATE_model_logits_or_baseline_heuristic",
            "candidate_level_semantics": label_semantics_for_partial_levels(parse_float_list(args.partial_capacity_levels)),
            "candidates": candidate_records,
        },
    }


def build_dynamic_sample(
    nodes,
    links,
    pairs,
    traffic_pair_series,
    start_idx: int,
    history_len: int,
    k_paths: int,
    failure_timeline: List[FailureState],
    events_by_t: List[List[Dict[str, Any]]],
    edge_risk: Dict[EdgeKey, float],
    flaky_edges: set,
    args,
    max_resample_attempts: int,
    device="cpu",
):
    """
    Builds one temporal HARP/GRATE sample using a slice of the global failure timeline.

    Output:
        dict with:
            node_features: [1, T, N, 2]
            edge_index: list length T, each [2, E_t]
            capacities: list length T, each [1, E_t]
            padded_edge_ids_per_path: list length T, each [P, L_t]
            paths_to_edges: list length T, each sparse [P, E_t]
            tm: [1, T, P, 1]
            tm_pred: [1, T, P, 1]
            future_capacities: [E_final]
            future_capacity_multipliers: [E_final]
            future_edge_state_labels: [E_final]

        Important:
            No failure_candidate_* keys are written by the generator.
            Candidate stress states/probabilities are generated later by the evaluator.
    """

    num_nodes = len(nodes)

    if start_idx + history_len >= len(failure_timeline):
        raise ValueError(
            f"Failure timeline too short for next-step labels at start_idx={start_idx}, "
            f"history_len={history_len}. len(failure_timeline)={len(failure_timeline)}"
        )

    # We keep max_resample_attempts for API compatibility. With a global timeline,
    # the sample is deterministic for a start index. Invalid samples are rejected.
    for _ in range(max(1, max_resample_attempts)):
        states_window = failure_timeline[start_idx : start_idx + history_len]
        events_window = events_by_t[start_idx : start_idx + history_len]
        future_state = failure_timeline[start_idx + history_len]
        future_events = events_by_t[start_idx + history_len]

        node_features_seq = []
        edge_index_seq = []
        capacities_seq = []
        padded_paths_seq = []
        paths_to_edges_seq = []
        edge_lists_seq = []

        valid = True

        for local_t in range(history_len):
            G_t = build_graph(
                nodes=nodes,
                links=links,
                edge_multipliers=states_window[local_t],
            )

            edge_list_t, edge_to_id_t = canonical_edges_from_graph(G_t)

            if len(edge_list_t) == 0:
                valid = False
                break

            edge_index_t, capacities_t = graph_to_edge_index_and_capacities(
                G_t,
                edge_list_t,
                device=device,
            )

            all_path_edge_ids_t = build_paths_for_graph(
                G=G_t,
                pairs=pairs,
                k_paths=k_paths,
                edge_to_id=edge_to_id_t,
            )

            if all_path_edge_ids_t is None:
                valid = False
                break

            padded_paths_t = pad_path_edge_ids(
                all_path_edge_ids_t,
                device=device,
            )

            paths_to_edges_t = build_paths_to_edges(
                all_path_edge_ids_t,
                num_edges=len(edge_list_t),
                device=device,
            )

            node_features_t = make_node_features(
                G_t,
                num_nodes=num_nodes,
                device=device,
            )

            node_features_seq.append(node_features_t)
            edge_index_seq.append(edge_index_t)
            capacities_seq.append(capacities_t.unsqueeze(0))  # [1, E_t]
            padded_paths_seq.append(padded_paths_t)
            paths_to_edges_seq.append(paths_to_edges_t)
            edge_lists_seq.append(edge_list_t)

        if not valid:
            raise RuntimeError(
                f"Global failure timeline produced disconnected/invalid sample at start_idx={start_idx}. "
                "Lower failure_probability, max_failed_edges, max_correlated_edges, or maintenance settings."
            )

        node_features = torch.stack(node_features_seq, dim=0).unsqueeze(0)
        # [1, T, N, 2]

        tm_window = build_tm_window(
            traffic_pair_series=traffic_pair_series,
            start_idx=start_idx,
            history_len=history_len,
            k_paths=k_paths,
            device=device,
        )

        failed_by_t = []
        partial_by_t = []
        capacity_multipliers_by_t = []
        for state in states_window:
            full, partial, multipliers = state_to_metadata(state)
            failed_by_t.append(full)
            partial_by_t.append(partial)
            capacity_multipliers_by_t.append(multipliers)

        future_data = build_true_future_tensors(
            future_state=future_state,
            final_edge_list=edge_lists_seq[-1],
            links=links,
            args=args,
            device=device,
        )

        sample = {
            "node_features": node_features,
            "edge_index": edge_index_seq,
            "capacities": capacities_seq,
            "padded_edge_ids_per_path": padded_paths_seq,
            "paths_to_edges": paths_to_edges_seq,
            "tm": tm_window,
            "tm_pred": tm_window.clone(),
            "future_capacities": future_data["future_capacities"],
            "future_capacity_multipliers": future_data["future_capacity_multipliers"],
            "future_edge_state_labels": future_data["future_edge_state_labels"],
            "metadata": {
                "start_idx": start_idx,
                "history_len": history_len,
                "k_paths": k_paths,
                "failed_by_t": failed_by_t,
                "partial_by_t": partial_by_t,
                "capacity_multipliers_by_t": capacity_multipliers_by_t,
                "events_by_t": events_window,
                "future_timestep": start_idx + history_len,
                "future_events": future_events,
                "future_state_labels": future_data["future_metadata"],
                "future_label_assumption": "actual_next_step_capacity_state_from_global_failure_timeline",
                "demand_assumption": "tm_pred_is_tm_perfect_current_demand",
                "telemetry_assumption": "capacities_are_observed_generated_telemetry",
                "failure_model_note": "model_learns_next_step_edge_state_labels; no_candidate_states_or_probabilities_saved",
                "evaluation_note": "candidate scenarios and heuristic probabilities are generated outside the dataset",
            },
        }

        return sample

    raise RuntimeError("Unexpected build_dynamic_sample failure")



# -----------------------------------------------------------------------------
# Shared static-cache dataset support
# -----------------------------------------------------------------------------
#
# The original dynamic generator writes edge_index, padded paths, and
# paths_to_edges into every sample. That is fine for Abilene/GEANT/small KDL
# subsets, but it explodes for full KDL because these tensors are mostly static.
#
# When --shared_static_cache 1 is used:
#   - static_cache.pt stores the fixed topology/path tensors once:
#       edge_index, base_capacities, padded_edge_ids_per_path, paths_to_edges
#   - each sample_*.pt stores only temporal tensors:
#       node_features [1,T,N,2]
#       capacities [1,T,E]
#       tm/tm_pred [1,T,P,1]
#       future_* labels [E]
#       static_cache_path
#
# This remains compatible with utils.training_utils.py after hydration.
# -----------------------------------------------------------------------------


def make_node_features_from_edge_list(
    nodes,
    edge_list: List[Tuple[int, int]],
    capacities: torch.Tensor,
    device="cpu",
):
    """
    Builds the same two node features as make_node_features(), but from a fixed
    edge list and an aligned capacity vector instead of a NetworkX graph.

    Features:
        feature 0 = normalized total directed degree
        feature 1 = normalized incident capacity sum
    """

    num_nodes = len(nodes)
    degree = np.zeros(num_nodes, dtype=np.float32)
    cap_sum = np.zeros(num_nodes, dtype=np.float32)

    caps_np = capacities.detach().cpu().numpy().astype(np.float32)

    for edge_idx, (u, v) in enumerate(edge_list):
        degree[int(u)] += 1.0
        degree[int(v)] += 1.0
        cap = float(caps_np[edge_idx])
        cap_sum[int(u)] += cap
        cap_sum[int(v)] += cap

    if degree.max() > 0:
        degree = degree / degree.max()

    if cap_sum.max() > 0:
        cap_sum = cap_sum / cap_sum.max()

    features = np.stack([degree, cap_sum], axis=-1)
    return torch.tensor(features, dtype=torch.float32, device=device)


def capacities_for_static_edge_list(
    edge_list: List[Tuple[int, int]],
    links,
    state: FailureState,
    current_failure_capacity_floor: float,
    device="cpu",
):
    """
    Converts an undirected capacity-state dictionary into a fixed directed
    capacity vector aligned with static_cache["edge_list"].

    For current/history telemetry in shared-cache mode we do NOT remove edges.
    Full failures are represented with a tiny positive capacity floor so HARP's
    forward pass never divides by zero. Future/evaluation labels still use 0.0
    through build_true_future_tensors().
    """

    base_capacity_by_edge = build_base_capacity_by_edge(links)
    caps = []

    for u, v in edge_list:
        key = canonical_edge_key(u, v)
        multiplier = float(state.get(key, 1.0))
        multiplier = max(0.0, min(1.0, multiplier))
        base_cap = float(base_capacity_by_edge[key])

        if multiplier <= 1e-8:
            caps.append(float(current_failure_capacity_floor))
        else:
            caps.append(base_cap * multiplier)

    return torch.tensor(caps, dtype=torch.float32, device=device)


def build_or_load_shared_static_cache(
    nodes,
    links,
    pairs,
    k_paths: int,
    out_dir: Path,
    args,
    device="cpu",
):
    """
    Builds or reuses a static topology/path cache.

    The fixed cache is built from the healthy base topology. It is appropriate
    for KDL-style shared-cache generation where failures/degradations are encoded
    as time-varying capacity vectors rather than by recomputing/removing paths
    in every sample.
    """

    cache_path = out_dir / args.static_cache_name

    if bool(int(getattr(args, "reuse_static_cache", 1))) and cache_path.exists():
        print(f"Reusing existing shared static cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        if "edge_list" not in cache:
            raise RuntimeError(f"Existing cache is missing edge_list: {cache_path}")
        return cache, [tuple(e) for e in cache["edge_list"]], cache_path

    print("Building shared static cache from healthy base topology...")
    print("  num nodes:", len(nodes))
    print("  num SD pairs:", len(pairs))
    print("  k_paths:", k_paths)

    G_base = build_graph(nodes=nodes, links=links, edge_multipliers={})
    edge_list, edge_to_id = canonical_edges_from_graph(G_base)

    if len(edge_list) == 0:
        raise RuntimeError("Healthy base graph has no edges.")

    edge_index, base_capacities = graph_to_edge_index_and_capacities(
        G_base,
        edge_list,
        device=device,
    )

    all_path_edge_ids = build_paths_for_graph(
        G=G_base,
        pairs=pairs,
        k_paths=k_paths,
        edge_to_id=edge_to_id,
    )

    if all_path_edge_ids is None:
        raise RuntimeError(
            "Could not build K shortest paths for the healthy base graph. "
            "The graph may be disconnected for at least one SD pair."
        )

    padded_paths = pad_path_edge_ids(all_path_edge_ids, device=device)
    paths_to_edges = build_paths_to_edges(
        all_path_edge_ids,
        num_edges=len(edge_list),
        device=device,
    )

    cache = {
        "format": "shared_static_cache_v1",
        "edge_index": edge_index,
        "base_capacities": base_capacities.unsqueeze(0),  # [1, E]
        "padded_edge_ids_per_path": padded_paths,
        "paths_to_edges": paths_to_edges,
        "edge_list": [tuple(e) for e in edge_list],
        "num_nodes": int(len(nodes)),
        "num_pairs": int(len(pairs)),
        "k_paths": int(k_paths),
        "num_paths": int(len(all_path_edge_ids)),
        "num_edges": int(len(edge_list)),
        "max_path_length": int(padded_paths.shape[1]),
        "note": (
            "Static topology/path tensors saved once. Samples store temporal "
            "capacity vectors and reference this file via static_cache_path."
        ),
    }

    torch.save(cache, cache_path)

    print("Saved shared static cache:", cache_path)
    print("  edge_index:", tuple(edge_index.shape))
    print("  base_capacities:", tuple(cache["base_capacities"].shape))
    print("  padded_edge_ids_per_path:", tuple(padded_paths.shape))
    print("  paths_to_edges:", tuple(paths_to_edges.shape))
    print("  static cache size MB:", cache_path.stat().st_size / 1024 / 1024)

    return cache, [tuple(e) for e in edge_list], cache_path


def build_dynamic_sample_shared_static_cache(
    nodes,
    links,
    pairs,
    traffic_pair_series,
    start_idx: int,
    history_len: int,
    k_paths: int,
    failure_timeline: List[FailureState],
    events_by_t: List[List[Dict[str, Any]]],
    edge_risk: Dict[EdgeKey, float],
    flaky_edges: set,
    static_edge_list: List[Tuple[int, int]],
    static_cache_path: Path,
    args,
    device="cpu",
):
    """
    Builds one compact temporal sample for shared-static-cache mode.

    Output sample intentionally does NOT include:
        edge_index
        padded_edge_ids_per_path
        paths_to_edges

    Those are stored once in static_cache.pt and attached at load time by
    utils.training_utils.hydrate_dynamic_sample_static_cache().
    """

    num_nodes = len(nodes)

    if start_idx + history_len >= len(failure_timeline):
        raise ValueError(
            f"Failure timeline too short for next-step labels at start_idx={start_idx}, "
            f"history_len={history_len}. len(failure_timeline)={len(failure_timeline)}"
        )

    states_window = failure_timeline[start_idx : start_idx + history_len]
    events_window = events_by_t[start_idx : start_idx + history_len]
    future_state = failure_timeline[start_idx + history_len]
    future_events = events_by_t[start_idx + history_len]

    capacities_seq = []
    node_features_seq = []

    for local_t in range(history_len):
        caps_t = capacities_for_static_edge_list(
            edge_list=static_edge_list,
            links=links,
            state=states_window[local_t],
            current_failure_capacity_floor=float(args.current_failure_capacity_floor),
            device=device,
        )

        node_features_t = make_node_features_from_edge_list(
            nodes=nodes,
            edge_list=static_edge_list,
            capacities=caps_t,
            device=device,
        )

        capacities_seq.append(caps_t)
        node_features_seq.append(node_features_t)

    node_features = torch.stack(node_features_seq, dim=0).unsqueeze(0)
    # [1, T, N, 2]

    capacities = torch.stack(capacities_seq, dim=0).unsqueeze(0)
    # [1, T, E]

    tm_window = build_tm_window(
        traffic_pair_series=traffic_pair_series,
        start_idx=start_idx,
        history_len=history_len,
        k_paths=k_paths,
        device=device,
    )

    failed_by_t = []
    partial_by_t = []
    capacity_multipliers_by_t = []
    for state in states_window:
        full, partial, multipliers = state_to_metadata(state)
        failed_by_t.append(full)
        partial_by_t.append(partial)
        capacity_multipliers_by_t.append(multipliers)

    future_data = build_true_future_tensors(
        future_state=future_state,
        final_edge_list=static_edge_list,
        links=links,
        args=args,
        device=device,
    )

    sample = {
        "node_features": node_features,
        "capacities": capacities,
        "tm": tm_window,
        "tm_pred": tm_window.clone(),
        "future_capacities": future_data["future_capacities"],
        "future_capacity_multipliers": future_data["future_capacity_multipliers"],
        "future_edge_state_labels": future_data["future_edge_state_labels"],
        "static_cache_path": Path(static_cache_path).name,
        "metadata": {
            "format": "shared_static_cache_v1",
            "static_cache_path": Path(static_cache_path).name,
            "start_idx": start_idx,
            "history_len": history_len,
            "k_paths": k_paths,
            "num_nodes": int(num_nodes),
            "num_pairs": int(len(pairs)),
            "num_paths": int(tm_window.shape[2]),
            "num_edges": int(len(static_edge_list)),
            "current_failure_capacity_floor": float(args.current_failure_capacity_floor),
            "failed_by_t": failed_by_t,
            "partial_by_t": partial_by_t,
            "capacity_multipliers_by_t": capacity_multipliers_by_t,
            "events_by_t": events_window,
            "future_timestep": start_idx + history_len,
            "future_events": future_events,
            "future_state_labels": future_data["future_metadata"],
            "future_label_assumption": "actual_next_step_capacity_state_from_global_failure_timeline",
            "demand_assumption": "tm_pred_is_tm_perfect_current_demand",
            "telemetry_assumption": (
                "capacities_are_observed_generated_telemetry; in shared-cache mode "
                "current full failures use a small positive floor to avoid divide-by-zero"
            ),
            "failure_model_note": "model_learns_next_step_edge_state_labels; no_candidate_states_or_probabilities_saved",
            "evaluation_note": "candidate scenarios and heuristic probabilities are generated outside the dataset",
        },
    }

    return sample

def print_sample_shapes(sample):
    print("==== SAMPLE SHAPES ====")
    print("node_features:", tuple(sample["node_features"].shape))

    if "edge_index" in sample:
        print("len(edge_index):", len(sample["edge_index"]))
        print("edge_index[0]:", tuple(sample["edge_index"][0].shape))
        print("edge_index[-1]:", tuple(sample["edge_index"][-1].shape))
    else:
        print("edge_index: <stored in shared static cache>")

    if "capacities" in sample:
        if isinstance(sample["capacities"], list):
            print("len(capacities):", len(sample["capacities"]))
            print("capacities[0]:", tuple(sample["capacities"][0].shape))
            print("capacities[-1]:", tuple(sample["capacities"][-1].shape))
        else:
            print("capacities:", tuple(sample["capacities"].shape))

    if "padded_edge_ids_per_path" in sample:
        print("len(padded_edge_ids_per_path):", len(sample["padded_edge_ids_per_path"]))
        print("padded_paths[0]:", tuple(sample["padded_edge_ids_per_path"][0].shape))
        print("padded_paths[-1]:", tuple(sample["padded_edge_ids_per_path"][-1].shape))
    else:
        print("padded_edge_ids_per_path: <stored in shared static cache>")

    if "paths_to_edges" in sample:
        print("len(paths_to_edges):", len(sample["paths_to_edges"]))
        print("paths_to_edges[0]:", tuple(sample["paths_to_edges"][0].shape))
        print("paths_to_edges[-1]:", tuple(sample["paths_to_edges"][-1].shape))
    else:
        print("paths_to_edges: <stored in shared static cache>")

    if "static_cache_path" in sample:
        print("static_cache_path:", sample["static_cache_path"])

    print("tm:", tuple(sample["tm"].shape))
    print("tm_pred:", tuple(sample["tm_pred"].shape))
    if "future_capacities" in sample:
        print("future_capacities:", tuple(sample["future_capacities"].shape))
        print("future_capacity_multipliers:", tuple(sample["future_capacity_multipliers"].shape))
        print("future_edge_state_labels:", tuple(sample["future_edge_state_labels"].shape))
        print("future labels seen:", sorted(set(int(x) for x in sample["future_edge_state_labels"].flatten())))
    bad_candidate_keys = sorted(k for k in sample.keys() if k.startswith("failure_candidate_"))
    print("failure_candidate_* keys present:", bad_candidate_keys)
    if bad_candidate_keys:
        raise RuntimeError(f"Generator must not write candidate artifacts, found: {bad_candidate_keys}")
    print("metadata keys:", sorted(sample["metadata"].keys()))
    print("=======================")


def summarize_timeline(failure_timeline, events_by_t, out_dir: Path, edge_risk, flaky_edges):
    total_slots = len(failure_timeline)
    slots_with_any = 0
    slots_with_full = 0
    slots_with_partial = 0
    max_full = 0
    max_partial = 0
    multipliers_seen = set([1.0])

    for state in failure_timeline:
        full = 0
        partial = 0
        if state:
            slots_with_any += 1
        for _, m in state.items():
            multipliers_seen.add(round(float(m), 4))
            if float(m) <= 1e-8:
                full += 1
            elif float(m) < 1.0:
                partial += 1
        if full > 0:
            slots_with_full += 1
        if partial > 0:
            slots_with_partial += 1
        max_full = max(max_full, full)
        max_partial = max(max_partial, partial)

    event_type_counts = {}
    seen_event_ids = set()
    for events in events_by_t:
        for ev in events:
            key = (
                ev.get("type"),
                ev.get("start_t"),
                ev.get("end_t_exclusive"),
                tuple((tuple(x["edge"]), x["multiplier"]) for x in ev.get("edges", [])),
            )
            if key in seen_event_ids:
                continue
            seen_event_ids.add(key)
            event_type_counts[ev.get("type", "unknown")] = event_type_counts.get(ev.get("type", "unknown"), 0) + 1

    summary = {
        "total_slots": total_slots,
        "slots_with_any_failure_or_degradation": slots_with_any,
        "fraction_slots_with_any_failure_or_degradation": slots_with_any / max(total_slots, 1),
        "slots_with_full_failure": slots_with_full,
        "fraction_slots_with_full_failure": slots_with_full / max(total_slots, 1),
        "slots_with_partial_degradation": slots_with_partial,
        "fraction_slots_with_partial_degradation": slots_with_partial / max(total_slots, 1),
        "max_full_edges_in_slot": max_full,
        "max_partial_edges_in_slot": max_partial,
        "multipliers_seen": sorted(multipliers_seen),
        "event_type_counts": event_type_counts,
        "flaky_edges": [tuple(e) for e in sorted(flaky_edges)],
        "edge_risk": {str(tuple(k)): float(v) for k, v in sorted(edge_risk.items())},
    }

    with open(out_dir / "failure_timeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("==== FAILURE TIMELINE SUMMARY ====")
    for k, v in summary.items():
        if k == "edge_risk":
            print(k, "<omitted; saved to JSON>")
        else:
            print(k, v)
    print("==================================")


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--topology_json", type=str, required=True)
    parser.add_argument("--pairs_pkl", type=str, required=True)
    parser.add_argument("--traffic_pkl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--history_len", type=int, default=6)
    parser.add_argument("--k_paths", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)

    # Backward-compatible failure controls.
    parser.add_argument("--failure_probability", type=float, default=0.025,
                        help="Probability of starting a new failure/degradation event at each timestep.")
    parser.add_argument("--max_failed_edges", type=int, default=1,
                        help="Max full-failed edges for a single full-failure event.")
    parser.add_argument("--min_failure_duration", type=int, default=2,
                        help="Legacy-only min full-failure duration.")
    parser.add_argument("--max_failure_duration", type=int, default=4,
                        help="Legacy-only max full-failure duration.")
    parser.add_argument("--max_resample_attempts", type=int, default=50)

    # Realistic model controls.
    parser.add_argument("--failure_model", type=str, default="realistic", choices=["legacy", "realistic"])
    parser.add_argument("--partial_capacity_levels", type=str, default="0.75,0.5,0.25")

    parser.add_argument("--partial_event_weight", type=float, default=0.35)
    parser.add_argument("--full_event_weight", type=float, default=0.25)
    parser.add_argument("--correlated_event_weight", type=float, default=0.15)
    parser.add_argument("--maintenance_event_weight", type=float, default=0.10)
    parser.add_argument("--flaky_burst_event_weight", type=float, default=0.15)

    parser.add_argument("--short_duration_min", type=int, default=1)
    parser.add_argument("--short_duration_max", type=int, default=2)
    parser.add_argument("--medium_duration_min", type=int, default=3)
    parser.add_argument("--medium_duration_max", type=int, default=12)
    parser.add_argument("--long_duration_min", type=int, default=12)
    parser.add_argument("--long_duration_max", type=int, default=48)
    parser.add_argument("--short_duration_weight", type=float, default=0.50)
    parser.add_argument("--medium_duration_weight", type=float, default=0.35)
    parser.add_argument("--long_duration_weight", type=float, default=0.15)

    parser.add_argument("--maintenance_duration_min", type=int, default=12)
    parser.add_argument("--maintenance_duration_max", type=int, default=72)
    parser.add_argument("--maintenance_full_probability", type=float, default=0.60)
    parser.add_argument("--maintenance_node_event_probability", type=float, default=0.30)
    parser.add_argument("--max_maintenance_edges", type=int, default=2)

    parser.add_argument("--correlated_full_probability", type=float, default=0.35)
    parser.add_argument("--max_correlated_edges", type=int, default=2)
    parser.add_argument("--max_edges_per_event", type=int, default=2,
                        help="Safety cap for small Abilene topology. 0 disables cap.")

    parser.add_argument("--flaky_edge_fraction", type=float, default=0.15)
    parser.add_argument("--flaky_edge_multiplier", type=float, default=5.0)
    parser.add_argument("--flaky_full_probability", type=float, default=0.35)
    parser.add_argument("--edge_risk_sigma", type=float, default=1.0)

    # Deprecated compatibility args. Candidate scenarios are evaluation-only artifacts
    # and must not be written into generated .pt files. These flags are accepted so
    # older SLURM scripts do not crash, but num_failure_candidates is forced to 0.
    parser.add_argument("--num_failure_candidates", type=int, default=0,
                        help="Deprecated/ignored. The generator never writes failure_candidate_* keys.")
    parser.add_argument("--candidate_no_change_weight", type=float, default=1.0)
    parser.add_argument("--candidate_edge_risk_weight", type=float, default=1.0)
    parser.add_argument("--candidate_recent_history_weight", type=float, default=2.0)
    parser.add_argument("--candidate_current_impairment_weight", type=float, default=3.0)
    parser.add_argument("--candidate_flaky_bonus_weight", type=float, default=1.0)
    parser.add_argument("--candidate_worsen_weight", type=float, default=1.25)
    parser.add_argument("--candidate_full_failure_weight", type=float, default=0.70)
    parser.add_argument("--candidate_correlated_weight", type=float, default=0.80)
    parser.add_argument("--candidate_recovery_weight", type=float, default=0.40)

    # Shared static-cache mode for large fixed-topology datasets such as full KDL.
    # Default is 0 so existing Abilene/GEANT/old KDL scripts keep producing the
    # original full per-sample format.
    parser.add_argument("--shared_static_cache", type=int, default=0,
                        help="If 1, write static topology/path tensors once to static_cache.pt and keep sample_*.pt compact.")
    parser.add_argument("--static_cache_name", type=str, default="static_cache.pt",
                        help="Filename for the shared static cache inside out_dir.")
    parser.add_argument("--reuse_static_cache", type=int, default=1,
                        help="If 1 and static_cache_name already exists in out_dir, reuse it instead of rebuilding paths.")
    parser.add_argument("--current_failure_capacity_floor", type=float, default=1e-4,
                        help="Positive capacity used for current/history full failures in shared-cache mode to avoid divide-by-zero.")

    parser.add_argument("--seed", type=int, default=0)


def validate_args(args):
    if args.history_len <= 0:
        raise ValueError("history_len must be positive")
    if args.k_paths <= 0:
        raise ValueError("k_paths must be positive")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not (0.0 <= args.failure_probability <= 1.0):
        raise ValueError("failure_probability must be in [0, 1]")
    for name in [
        "short_duration_min", "short_duration_max",
        "medium_duration_min", "medium_duration_max",
        "long_duration_min", "long_duration_max",
        "maintenance_duration_min", "maintenance_duration_max",
    ]:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.short_duration_min > args.short_duration_max:
        raise ValueError("short_duration_min > short_duration_max")
    if args.medium_duration_min > args.medium_duration_max:
        raise ValueError("medium_duration_min > medium_duration_max")
    if args.long_duration_min > args.long_duration_max:
        raise ValueError("long_duration_min > long_duration_max")
    if args.maintenance_duration_min > args.maintenance_duration_max:
        raise ValueError("maintenance_duration_min > maintenance_duration_max")
    if int(getattr(args, "shared_static_cache", 0)) and args.current_failure_capacity_floor <= 0:
        raise ValueError("current_failure_capacity_floor must be positive in shared_static_cache mode")


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    validate_args(args)

    if int(getattr(args, "num_failure_candidates", 0)) != 0:
        print(
            "[WARN] --num_failure_candidates is deprecated in the clean learned-failure "
            "generator. Forcing it to 0; no failure_candidate_* keys will be written."
        )
        args.num_failure_candidates = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topology_data = load_json(args.topology_json)
    raw_nodes, raw_links = parse_topology_json(topology_data)

    # Important:
    # GNN indexing expects node ids to be 0..N-1.
    # If topology uses arbitrary node labels, remap them.
    node_labels = list(raw_nodes)
    node_to_idx = {node: i for i, node in enumerate(node_labels)}

    nodes = list(range(len(node_labels)))

    links = []
    for u, v, cap in raw_links:
        links.append((node_to_idx[u], node_to_idx[v], float(cap)))

    pair_data = load_pickle(args.pairs_pkl)
    raw_pairs = normalize_pairs(pair_data)

    pairs = []
    for u, v in raw_pairs:
        pairs.append((node_to_idx[u], node_to_idx[v]))

    tm_data = load_pickle(args.traffic_pkl)
    traffic_pair_series = normalize_traffic_matrices(
        tm_data,
        pairs=raw_pairs,
        node_to_idx=node_to_idx,
    )

    # Need one extra timestep after the history window for the true t+1 label.
    max_start = traffic_pair_series.shape[0] - args.history_len - 1

    if max_start <= args.start_idx:
        raise ValueError(
            f"Not enough traffic timesteps. traffic length={traffic_pair_series.shape[0]}, "
            f"history_len={args.history_len}, start_idx={args.start_idx}. "
            "Need one extra timestep after the window for true t+1 labels."
        )

    timeline_len = traffic_pair_series.shape[0]
    print(f"Generating global failure/capacity timeline with T={timeline_len}")
    failure_timeline, events_by_t, edge_risk, flaky_edges = make_failure_timeline(
        links=links,
        nodes=nodes,
        T=timeline_len,
        args=args,
    )
    summarize_timeline(failure_timeline, events_by_t, out_dir, edge_risk, flaky_edges)

    shared_static_cache = bool(int(getattr(args, "shared_static_cache", 0)))
    static_cache = None
    static_edge_list = None
    static_cache_path = None

    if shared_static_cache:
        static_cache, static_edge_list, static_cache_path = build_or_load_shared_static_cache(
            nodes=nodes,
            links=links,
            pairs=pairs,
            k_paths=args.k_paths,
            out_dir=out_dir,
            args=args,
            device="cpu",
        )
        print("Shared static-cache mode enabled.")
        print("  static_cache_path:", static_cache_path)
        print("  static edge count:", len(static_edge_list))
        print("  static path count:", int(static_cache["num_paths"]))

    saved = 0
    skipped = 0

    start_idx = args.start_idx
    while saved < args.num_samples and start_idx <= max_start:
        try:
            if shared_static_cache:
                sample = build_dynamic_sample_shared_static_cache(
                    nodes=nodes,
                    links=links,
                    pairs=pairs,
                    traffic_pair_series=traffic_pair_series,
                    start_idx=start_idx,
                    history_len=args.history_len,
                    k_paths=args.k_paths,
                    failure_timeline=failure_timeline,
                    events_by_t=events_by_t,
                    edge_risk=edge_risk,
                    flaky_edges=flaky_edges,
                    static_edge_list=static_edge_list,
                    static_cache_path=static_cache_path,
                    args=args,
                    device="cpu",
                )
            else:
                sample = build_dynamic_sample(
                    nodes=nodes,
                    links=links,
                    pairs=pairs,
                    traffic_pair_series=traffic_pair_series,
                    start_idx=start_idx,
                    history_len=args.history_len,
                    k_paths=args.k_paths,
                    failure_timeline=failure_timeline,
                    events_by_t=events_by_t,
                    edge_risk=edge_risk,
                    flaky_edges=flaky_edges,
                    args=args,
                    max_resample_attempts=args.max_resample_attempts,
                    device="cpu",
                )
        except RuntimeError as exc:
            skipped += 1
            if skipped <= 10:
                print(f"[WARN] Skipping start_idx={start_idx}: {exc}")
            start_idx += 1
            continue

        if saved == 0:
            print_sample_shapes(sample)

        out_path = out_dir / f"sample_{saved:06d}.pt"
        torch.save(sample, out_path)
        saved += 1
        start_idx += 1

    print(f"Saved {saved} samples to {out_dir}")
    print(f"Skipped {skipped} invalid windows")

    if saved < args.num_samples:
        print(
            f"[WARN] Requested {args.num_samples} samples but only saved {saved}. "
            "Try reducing failure_probability or increasing available traffic length."
        )


if __name__ == "__main__":
    main()