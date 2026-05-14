#!/usr/bin/env python3
import csv
import json
import pickle
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DOTE_DIR = PROJECT_ROOT / "DOTE" / "networking_envs" / "data" / "Abilene"
TEAL_DIR = PROJECT_ROOT / "teal"
OUT_DIR = ROOT / "results" / "cross_baselines_resilience"

CURRENT_WEIGHT = 1.0
RESILIENCE_WEIGHT = 0.25
WORST_CASE_WEIGHT = 0.5
FAILURE_CAPACITY_FRACTION = 0.25


class NeuralNetworkMaxUtil(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


def summary(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {}

    def pct(frac):
        return values[int(len(values) * frac)]

    return {
        "count": len(values),
        "average": statistics.mean(values),
        "median": pct(0.5),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
    }


def write_values(prefix, results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ("combined", "current", "expected_failure", "worst_failure"):
        values = [row[metric] for row in results]
        with open(OUT_DIR / f"{prefix}_{metric}_values.txt", "w") as f:
            for value in values:
                f.write(f"{value}\n")
        stats = summary(values)
        with open(OUT_DIR / f"{prefix}_{metric}_stats.txt", "w") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")


def combined_metrics(edge_loads, capacities, opt):
    edge_loads = np.asarray(edge_loads, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.float64)
    opt = max(float(opt), 1e-12)

    current_raw = np.max(edge_loads / np.maximum(capacities, 1e-12))
    current = current_raw / opt

    scenario_mlus = []
    for edge_idx in range(len(capacities)):
        scenario_caps = capacities.copy()
        scenario_caps[edge_idx] = max(scenario_caps[edge_idx] * FAILURE_CAPACITY_FRACTION, 1e-12)
        scenario_mlus.append(np.max(edge_loads / scenario_caps))
    scenario_norms = np.asarray(scenario_mlus) / opt
    expected = float(np.mean(scenario_norms))
    worst = float(np.max(scenario_norms))
    combined = (
        CURRENT_WEIGHT * current
        + RESILIENCE_WEIGHT
        * ((1.0 - WORST_CASE_WEIGHT) * expected + WORST_CASE_WEIGHT * worst)
    )

    return {
        "combined": float(combined),
        "current": float(current),
        "current_raw": float(current_raw),
        "expected_failure": expected,
        "worst_failure": worst,
    }


def dote_read_edges():
    directed_caps = {}
    with open(DOTE_DIR / "Abilene_int.pickle.nnet") as f:
        for line in f:
            src, dst, cap = line.strip().split(",")
            src = int(src)
            dst = int(dst)
            cap = float(cap)
            directed_caps[(src, dst)] = cap
            directed_caps.setdefault((dst, src), cap)
    edge_list = sorted(directed_caps)
    return edge_list, np.asarray([directed_caps[e] for e in edge_list], dtype=np.float64)


def dote_read_paths(edge_to_idx):
    paths_by_commodity = {}
    with open(DOTE_DIR / "tunnels.txt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair, paths = line.split(":")
            src, dst = [int(v) for v in pair.split()]
            parsed = []
            for path in paths.split(","):
                nodes = [int(v) for v in path.split("-")]
                parsed.append([edge_to_idx[(u, v)] for u, v in zip(nodes, nodes[1:])])
            paths_by_commodity[(src, dst)] = parsed
    return paths_by_commodity


def dote_edge_loads_from_weights(weights, demand, num_nodes, paths_by_commodity, num_edges):
    weights = np.asarray(weights, dtype=np.float64)
    edge_loads = np.zeros(num_edges, dtype=np.float64)
    path_offset = 0
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue
            paths = paths_by_commodity[(src, dst)]
            path_weights = weights[path_offset : path_offset + len(paths)] + 1e-16
            path_splits = path_weights / np.sum(path_weights)
            commodity_demand = demand[src * num_nodes + dst]
            for split, edge_path in zip(path_splits, paths):
                flow = commodity_demand * split
                for edge_idx in edge_path:
                    edge_loads[edge_idx] += flow
            path_offset += len(paths)
    return edge_loads


def evaluate_dote():
    edge_list, capacities = dote_read_edges()
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}
    paths_by_commodity = dote_read_paths(edge_to_idx)

    hist = np.loadtxt(DOTE_DIR / "test" / "4.hist")
    opts = np.loadtxt(DOTE_DIR / "test" / "4.opt")
    num_nodes = int(np.sqrt(hist.shape[1]))
    mask = np.ones((num_nodes, num_nodes), dtype=bool)
    np.fill_diagonal(mask, False)

    model = torch.load(DOTE_DIR / "model_dote.pkl", map_location="cpu", weights_only=False)
    model.eval()

    results = []
    with torch.no_grad():
        for idx in range(1, len(hist)):
            prev = hist[idx - 1].reshape(num_nodes, num_nodes)[mask] / 1e9
            pred = model(torch.tensor(prev.reshape(1, -1), dtype=torch.float64)).numpy()[0]
            edge_loads = dote_edge_loads_from_weights(
                pred,
                demand=hist[idx],
                num_nodes=num_nodes,
                paths_by_commodity=paths_by_commodity,
                num_edges=len(edge_list),
            )
            row = combined_metrics(edge_loads, capacities, opts[idx])
            row["sample"] = idx
            results.append(row)
    write_values("dote_abilene_maxutil", results)
    return results


def teal_load_topology():
    try:
        import networkx as nx
        from networkx.readwrite import json_graph
    except ImportError as exc:
        raise RuntimeError("networkx is required for Teal topology parsing") from exc

    with open(TEAL_DIR / "topologies" / "B4.json") as f:
        data = json.load(f)
    graph = json_graph.node_link_graph(data)
    edge_list = list(graph.edges)
    capacities = np.asarray([float(graph.edges[e]["capacity"]) for e in edge_list], dtype=np.float64)
    return graph, edge_list, capacities


def teal_regular_paths(num_path=4):
    path_file = (
        TEAL_DIR
        / "topologies"
        / "paths"
        / "path-form"
        / "B4.json-4-paths_edge-disjoint-True_dist-metric-min-hop-dict.pkl"
    )
    with open(path_file, "rb") as f:
        path_dict = pickle.load(f)
    for key, paths in list(path_dict.items()):
        if len(paths) < num_path:
            path_dict[key] = [paths[0] for _ in range(num_path - len(paths))] + paths
        elif len(paths) > num_path:
            path_dict[key] = paths[:num_path]
    return path_dict


def teal_demands_from_tm(tm):
    return np.asarray(
        [float(ele) for i, ele in enumerate(tm.flatten()) if i % len(tm) != i // len(tm)],
        dtype=np.float64,
    )


def solve_optimal_mlu(num_nodes, demands, capacities, edge_to_idx, path_dict, num_path=4):
    commodities = [(s, t) for s in range(num_nodes) for t in range(num_nodes) if s != t]
    num_paths = len(commodities) * num_path
    util_idx = num_paths
    objective = np.zeros(num_paths + 1)
    objective[util_idx] = 1.0

    a_eq = np.zeros((len(commodities), num_paths + 1))
    b_eq = np.ones(len(commodities))
    a_ub = np.zeros((len(capacities), num_paths + 1))
    b_ub = np.zeros(len(capacities))

    path_id = 0
    for commodity_id, (src, dst) in enumerate(commodities):
        for path in path_dict[(src, dst)]:
            a_eq[commodity_id, path_id] = 1.0
            for u, v in zip(path[:-1], path[1:]):
                a_ub[edge_to_idx[(u, v)], path_id] = demands[commodity_id]
            path_id += 1

    for edge_idx, capacity in enumerate(capacities):
        a_ub[edge_idx, util_idx] = -capacity

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * num_paths + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[util_idx])


def evaluate_teal():
    graph, edge_list, capacities = teal_load_topology()
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}
    path_dict = teal_regular_paths()
    num_nodes = graph.number_of_nodes()

    results = []
    for seed in range(28, 36):
        sol_path = (
            TEAL_DIR
            / "run"
            / "teal-logs"
            / f"B4.json-real-{seed}-teal_objective-min_max_link_util_4-paths_edge-disjoint-True_dist-metric-min-hop_sol-mat.pt"
        )
        tm_path = TEAL_DIR / "traffic-matrices" / "real" / f"B4.json_real_{seed}_1.0_traffic-matrix.pkl"
        sol_mat = torch.load(sol_path, map_location="cpu").to_dense().numpy()
        edge_loads = sol_mat.sum(axis=0)
        with open(tm_path, "rb") as f:
            tm = pickle.load(f)
        demands = teal_demands_from_tm(tm)
        opt = solve_optimal_mlu(num_nodes, demands, capacities, edge_to_idx, path_dict)
        row = combined_metrics(edge_loads, capacities, opt)
        row["sample"] = seed
        row["opt"] = opt
        results.append(row)
    write_values("teal_b4_minmax", results)
    return results


def write_summary_csv(all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "count", "average", "median", "p95", "max"])
        for model_name, results in all_results.items():
            for metric in ("combined", "current", "expected_failure", "worst_failure"):
                stats = summary([row[metric] for row in results])
                writer.writerow(
                    [
                        model_name,
                        metric,
                        stats["count"],
                        stats["average"],
                        stats["median"],
                        stats["p95"],
                        stats["max"],
                    ]
                )


def main():
    dote_results = evaluate_dote()
    teal_results = evaluate_teal()
    all_results = {
        "DOTE Abilene MAXUTIL": dote_results,
        "Teal B4 min_max_link_util": teal_results,
    }
    write_summary_csv(all_results)
    for name, results in all_results.items():
        print(name)
        for metric in ("combined", "current", "expected_failure", "worst_failure"):
            stats = summary([row[metric] for row in results])
            print(
                f"  {metric}: avg={stats['average']:.6f}, "
                f"median={stats['median']:.6f}, p95={stats['p95']:.6f}, "
                f"max={stats['max']:.6f}, n={stats['count']}"
            )


if __name__ == "__main__":
    main()
