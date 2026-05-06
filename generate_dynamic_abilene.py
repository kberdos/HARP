# generate_dynamic_abilene.py

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np
import torch


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

    You may need to lightly edit this depending on your exact t1.json format.
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


def build_graph(nodes, links, failed_undirected_edges=None):
    """
    Builds a directed graph.

    failed_undirected_edges contains canonical undirected keys:
        tuple(sorted((u, v)))
    """

    failed_undirected_edges = failed_undirected_edges or set()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for src, dst, cap in links:
        key = tuple(sorted((src, dst)))
        if key in failed_undirected_edges:
            continue

        # Add directed edge as written.
        G.add_edge(src, dst, capacity=float(cap), weight=1.0)

        # If your topology file is undirected, this makes it bidirectional.
        # If your HARP topology is already directed, remove this line.
        G.add_edge(dst, src, capacity=float(cap), weight=1.0)

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

    This is necessary because HARP expects fixed K paths per pair,
    but Abilene may not always have K distinct simple paths for every SD pair.
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


def make_failure_schedule(
    links,
    T: int,
    failure_probability: float,
    max_failed_edges: int,
    min_duration: int,
    max_duration: int,
):
    """
    Creates a list of failed edge sets, one per timestep.

    Failure is undirected:
        if link A-B fails, both A->B and B->A are removed.
    """

    failed_by_t = [set() for _ in range(T)]

    undirected_edges = list({tuple(sorted((u, v))) for u, v, _ in links})

    for t in range(T):
        if random.random() < failure_probability:
            num_failures = random.randint(1, max_failed_edges)
            chosen = random.sample(
                undirected_edges,
                k=min(num_failures, len(undirected_edges)),
            )

            duration = random.randint(min_duration, max_duration)
            end_t = min(T, t + duration)

            for tt in range(t, end_t):
                failed_by_t[tt].update(chosen)

    return failed_by_t


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


def build_dynamic_sample(
    nodes,
    links,
    pairs,
    traffic_pair_series,
    start_idx: int,
    history_len: int,
    k_paths: int,
    failure_probability: float,
    max_failed_edges: int,
    min_failure_duration: int,
    max_failure_duration: int,
    max_resample_attempts: int,
    device="cpu",
):
    """
    Builds one Option A Temporal HARP sample.

    Output:
        dict with:
            node_features: [1, T, N, 2]
            edge_index: list length T, each [2, E_t]
            capacities: list length T, each [1, E_t]
            padded_edge_ids_per_path: list length T, each [P, L_t]
            paths_to_edges: list length T, each sparse [P, E_t]
            tm: [1, T, P, 1]
            tm_pred: [1, T, P, 1]
    """

    num_nodes = len(nodes)

    for attempt in range(max_resample_attempts):
        failed_by_t = make_failure_schedule(
            links=links,
            T=history_len,
            failure_probability=failure_probability,
            max_failed_edges=max_failed_edges,
            min_duration=min_failure_duration,
            max_duration=max_failure_duration,
        )

        node_features_seq = []
        edge_index_seq = []
        capacities_seq = []
        padded_paths_seq = []
        paths_to_edges_seq = []

        valid = True

        for local_t in range(history_len):
            G_t = build_graph(
                nodes=nodes,
                links=links,
                failed_undirected_edges=failed_by_t[local_t],
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

        if not valid:
            continue

        node_features = torch.stack(node_features_seq, dim=0).unsqueeze(0)
        # [1, T, N, 2]

        tm_window = build_tm_window(
            traffic_pair_series=traffic_pair_series,
            start_idx=start_idx,
            history_len=history_len,
            k_paths=k_paths,
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
            "metadata": {
                "start_idx": start_idx,
                "history_len": history_len,
                "k_paths": k_paths,
                "failed_by_t": [list(x) for x in failed_by_t],
            },
        }

        return sample

    raise RuntimeError(
        f"Could not build a valid dynamic sample after {max_resample_attempts} attempts. "
        "Try lowering failure_probability/max_failed_edges or reducing k_paths."
    )


def print_sample_shapes(sample):
    print("==== SAMPLE SHAPES ====")
    print("node_features:", tuple(sample["node_features"].shape))

    print("len(edge_index):", len(sample["edge_index"]))
    print("edge_index[0]:", tuple(sample["edge_index"][0].shape))
    print("edge_index[-1]:", tuple(sample["edge_index"][-1].shape))

    print("len(capacities):", len(sample["capacities"]))
    print("capacities[0]:", tuple(sample["capacities"][0].shape))
    print("capacities[-1]:", tuple(sample["capacities"][-1].shape))

    print("len(padded_edge_ids_per_path):", len(sample["padded_edge_ids_per_path"]))
    print("padded_paths[0]:", tuple(sample["padded_edge_ids_per_path"][0].shape))
    print("padded_paths[-1]:", tuple(sample["padded_edge_ids_per_path"][-1].shape))

    print("len(paths_to_edges):", len(sample["paths_to_edges"]))
    print("paths_to_edges[0]:", tuple(sample["paths_to_edges"][0].shape))
    print("paths_to_edges[-1]:", tuple(sample["paths_to_edges"][-1].shape))

    print("tm:", tuple(sample["tm"].shape))
    print("tm_pred:", tuple(sample["tm_pred"].shape))
    print("=======================")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--topology_json", type=str, required=True)
    parser.add_argument("--pairs_pkl", type=str, required=True)
    parser.add_argument("--traffic_pkl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--history_len", type=int, default=6)
    parser.add_argument("--k_paths", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)

    parser.add_argument("--failure_probability", type=float, default=0.25)
    parser.add_argument("--max_failed_edges", type=int, default=2)
    parser.add_argument("--min_failure_duration", type=int, default=2)
    parser.add_argument("--max_failure_duration", type=int, default=4)
    parser.add_argument("--max_resample_attempts", type=int, default=50)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

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

    max_start = traffic_pair_series.shape[0] - args.history_len

    if max_start <= args.start_idx:
        raise ValueError(
            f"Not enough traffic timesteps. traffic length={traffic_pair_series.shape[0]}, "
            f"history_len={args.history_len}, start_idx={args.start_idx}"
        )

    saved = 0

    for i in range(args.num_samples):
        start_idx = args.start_idx + i

        if start_idx > max_start:
            break

        sample = build_dynamic_sample(
            nodes=nodes,
            links=links,
            pairs=pairs,
            traffic_pair_series=traffic_pair_series,
            start_idx=start_idx,
            history_len=args.history_len,
            k_paths=args.k_paths,
            failure_probability=args.failure_probability,
            max_failed_edges=args.max_failed_edges,
            min_failure_duration=args.min_failure_duration,
            max_failure_duration=args.max_failure_duration,
            max_resample_attempts=args.max_resample_attempts,
            device="cpu",
        )

        if i == 0:
            print_sample_shapes(sample)

        out_path = out_dir / f"sample_{saved:06d}.pt"
        torch.save(sample, out_path)

        saved += 1

    print(f"Saved {saved} samples to {out_dir}")


if __name__ == "__main__":
    main()