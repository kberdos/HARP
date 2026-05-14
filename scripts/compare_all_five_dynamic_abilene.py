#!/usr/bin/env python3
import csv
import math
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.resilience_utils import resilience_loss_from_details
from utils.training_utils import DynamicAbileneDataset, dynamic_collate


SAMPLES_DIR = ROOT / "dynamic_abilene_h6_1000_samples"
OUT_DIR = ROOT / "results" / "dynamic_abilene" / "4sp" / "0" / "resilience_compare_all_five"
HARP_COMPARE_DIR = ROOT / "results" / "dynamic_abilene" / "4sp" / "0" / "resilience_compare_all"

TRAIN_START = 0
TRAIN_END = 800
TEST_START = 800
TEST_END = 1000
EPOCHS = 10
LR = 1e-3
NUM_PATHS = 4
SEED = 7

RESILIENCE_ARGS = SimpleNamespace(
    current_weight=1.0,
    resilience_weight=0.25,
    worst_case_weight=0.5,
    failure_capacity_fraction=0.25,
    risk_prior=1e-3,
    risk_recency_power=2.0,
    scenario_top_k=0,
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def percentile(sorted_values, fraction):
    if not sorted_values:
        return float("nan")
    return sorted_values[int(len(sorted_values) * fraction)]


def stats(values):
    values = sorted(float(v) for v in values)
    return {
        "count": len(values),
        "average": statistics.mean(values) if values else float("nan"),
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1] if values else float("nan"),
    }


def read_values(path):
    return [float(line.strip()) for line in open(path) if line.strip()]


def load_existing_harp_results():
    specs = {
        "resilient_temporal": (
            HARP_COMPARE_DIR / "resilient_temporal",
            "harp_resilient_temporal_dynamic_failure_id_None",
        ),
        "vanilla_temporal": (
            HARP_COMPARE_DIR / "vanilla_temporal",
            "harp_resilient_temporal_dynamic_failure_id_None",
        ),
        "snapshot_baseline": (
            HARP_COMPARE_DIR / "snapshot_baseline",
            "harp_resilient_baseline_dynamic_failure_id_None",
        ),
    }
    results = {}
    for model_name, (directory, prefix) in specs.items():
        results[model_name] = {}
        for metric in ["combined", "current", "expected_failure", "worst_failure"]:
            results[model_name][metric] = read_values(directory / f"{prefix}_{metric}_values.txt")
    return results


class DoteMaxUtilPolicy(nn.Module):
    """
    DOTE's MAXUTIL MLP architecture adapted to the 12-node HARP demand vector.

    The architecture is the same 4x128 ReLU MLP with sigmoid path weights used
    by DOTE; only the input/output dimensions come from the HARP 12-node,
    4-path dynamic Abilene samples.
    """

    def __init__(self, input_dim=132, output_dim=528):
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


class TealFlowGNNPolicy(nn.Module):
    """
    Teal FlowGNN actor architecture adapted to HARP dynamic final topologies.

    The GNN/DNN layer structure and final mean projection match Teal. The
    adapter supplies each sample's final topology graph and interprets the raw
    action as full per-commodity path splits for HARP's MLU objective.
    """

    def __init__(self, num_layers=6, num_paths=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_paths = num_paths
        self.gnn_list = nn.ModuleList([nn.Linear(i + 1, i + 1) for i in range(num_layers)])
        self.dnn_list = nn.ModuleList(
            [nn.Linear(num_paths * (i + 1), num_paths * (i + 1)) for i in range(num_layers)]
        )
        self.mean_linear = nn.Linear(num_paths * (num_layers + 1), num_paths)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, capacities, demands_by_path, teal_edge_index, teal_edge_values):
        num_edges = capacities.numel()
        num_path_nodes = demands_by_path.numel()
        h0 = torch.cat([capacities.reshape(-1), demands_by_path.reshape(-1)]).reshape(-1, 1)
        h = h0
        graph_size = num_edges + num_path_nodes

        for i in range(self.num_layers):
            h = self.gnn_list[i](h)
            adj = torch.sparse_coo_tensor(
                teal_edge_index,
                teal_edge_values,
                (graph_size, graph_size),
                device=h.device,
                dtype=h.dtype,
            ).coalesce()
            h = torch.sparse.mm(adj, h)
            path_h = self.dnn_list[i](
                h[-num_path_nodes:, :].reshape(-1, self.num_paths * (i + 1))
            ).reshape(num_path_nodes, i + 1)
            h = torch.cat([h[:-num_path_nodes, :], path_h], dim=0)
            h = torch.cat([h, h0], dim=-1)

        raw = self.mean_linear(h[-num_path_nodes:, :].reshape(-1, self.num_paths * (self.num_layers + 1)))
        return raw.reshape(-1)


def final_demands_by_path(sample):
    return sample["tm"][0, -1, :, 0].float()


def previous_demands_by_commodity(sample):
    return sample["tm"][0, -2, ::NUM_PATHS, 0].float().reshape(1, -1)


def normalize_path_weights(raw_weights):
    weights = raw_weights.reshape(-1, NUM_PATHS).clamp_min(1e-12)
    return (weights / weights.sum(dim=1, keepdim=True)).reshape(-1)


def softmax_path_splits(raw_weights):
    return torch.softmax(raw_weights.reshape(-1, NUM_PATHS), dim=1).reshape(-1)


def edge_load_details(sample, path_splits):
    paths_to_edges = sample["paths_to_edges"][-1].float()
    capacities = sample["capacities"][-1].reshape(-1).float().clamp_min(1e-9)
    edge_index = sample["edge_index"][-1]
    path_flow = path_splits * final_demands_by_path(sample)
    data_on_links = torch.sparse.mm(paths_to_edges.t(), path_flow.reshape(-1, 1)).t()
    edges_util = data_on_links / capacities.reshape(1, -1)
    return {
        "edges_util": edges_util,
        "data_on_links": data_on_links,
        "capacities": capacities.reshape(1, -1),
        "edge_index": edge_index,
    }


def current_norm_loss(details, sample):
    opt = sample["opt"].float().reshape(()).clamp_min(1e-12)
    return details["edges_util"].max() / opt


def teal_graph_from_sample(sample):
    paths_to_edges = sample["paths_to_edges"][-1].coalesce()
    indices = paths_to_edges.indices()
    path_ids = indices[0]
    edge_ids = indices[1]
    num_edges = sample["capacities"][-1].numel()
    src = num_edges + path_ids
    dst = edge_ids

    degree = torch.zeros(num_edges + paths_to_edges.shape[0], dtype=torch.float32)
    degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

    both_src = torch.cat([src, dst])
    both_dst = torch.cat([dst, src])
    values = 1.0 / torch.sqrt(degree[both_src].clamp_min(1.0) * degree[both_dst].clamp_min(1.0))
    return torch.stack([both_src, both_dst]).long(), values.float()


def make_loader(start, end, shuffle):
    return DataLoader(
        DynamicAbileneDataset(SAMPLES_DIR, start_idx=start, end_idx=end),
        batch_size=1,
        shuffle=shuffle,
        collate_fn=dynamic_collate,
    )


def train_dote():
    model_path = OUT_DIR / "models" / "dote_harp_dynamic.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = DoteMaxUtilPolicy()
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(TRAIN_START, TRAIN_END, shuffle=True)
    for epoch in range(EPOCHS):
        losses = []
        for sample in tqdm(loader, desc=f"DOTE adapter epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad(set_to_none=True)
            raw = model(previous_demands_by_commodity(sample)).reshape(-1)
            splits = normalize_path_weights(raw)
            details = edge_load_details(sample, splits)
            loss = current_norm_loss(details, sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"DOTE adapter epoch {epoch + 1}: avg current_norm={sum(losses)/len(losses):.6f}")
    torch.save(model.state_dict(), model_path)
    return model


def train_teal():
    model_path = OUT_DIR / "models" / "teal_harp_dynamic.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = TealFlowGNNPolicy()
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(TRAIN_START, TRAIN_END, shuffle=True)
    for epoch in range(EPOCHS):
        losses = []
        for sample in tqdm(loader, desc=f"Teal adapter epoch {epoch + 1}/{EPOCHS}"):
            optimizer.zero_grad(set_to_none=True)
            capacities = sample["capacities"][-1].reshape(-1).float()
            demands = final_demands_by_path(sample)
            graph_index, graph_values = teal_graph_from_sample(sample)
            raw = model(capacities, demands, graph_index, graph_values)
            splits = softmax_path_splits(raw)
            details = edge_load_details(sample, splits)
            loss = current_norm_loss(details, sample)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"Teal adapter epoch {epoch + 1}: avg current_norm={sum(losses)/len(losses):.6f}")
    torch.save(model.state_dict(), model_path)
    return model


def evaluate_adapter(model, kind):
    loader = make_loader(TEST_START, TEST_END, shuffle=False)
    results = {"combined": [], "current": [], "expected_failure": [], "worst_failure": []}
    model.eval()
    with torch.no_grad():
        for sample in tqdm(loader, desc=f"Evaluate {kind} adapter"):
            if kind == "dote":
                raw = model(previous_demands_by_commodity(sample)).reshape(-1)
                splits = normalize_path_weights(raw)
            elif kind == "teal":
                capacities = sample["capacities"][-1].reshape(-1).float()
                demands = final_demands_by_path(sample)
                graph_index, graph_values = teal_graph_from_sample(sample)
                raw = model(capacities, demands, graph_index, graph_values)
                splits = softmax_path_splits(raw)
            else:
                raise ValueError(kind)

            details = edge_load_details(sample, splits)
            metrics = resilience_loss_from_details(
                details=details,
                sample=sample,
                current_weight=RESILIENCE_ARGS.current_weight,
                resilience_weight=RESILIENCE_ARGS.resilience_weight,
                worst_case_weight=RESILIENCE_ARGS.worst_case_weight,
                failure_capacity_fraction=RESILIENCE_ARGS.failure_capacity_fraction,
                risk_prior=RESILIENCE_ARGS.risk_prior,
                risk_recency_power=RESILIENCE_ARGS.risk_recency_power,
                scenario_top_k=RESILIENCE_ARGS.scenario_top_k,
            )
            results["combined"].append(metrics["combined_value"])
            results["current"].append(metrics["current_norm"])
            results["expected_failure"].append(metrics["expected_failure_norm"])
            results["worst_failure"].append(metrics["worst_failure_norm"])
    return results


def write_distribution(model_name, values_by_metric):
    safe_name = model_name.replace(" ", "_")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for metric, values in values_by_metric.items():
        with open(OUT_DIR / f"{safe_name}_{metric}_values.txt", "w") as f:
            for value in values:
                f.write(f"{value}\n")
        with open(OUT_DIR / f"{safe_name}_{metric}_stats.txt", "w") as f:
            for key, value in stats(values).items():
                f.write(f"{key}: {value}\n")


def write_summary(all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "count", "average", "median", "p95", "max"])
        for model_name, values_by_metric in all_results.items():
            for metric, values in values_by_metric.items():
                s = stats(values)
                writer.writerow([model_name, metric, s["count"], s["average"], s["median"], s["p95"], s["max"]])


def main():
    set_seed(SEED)
    all_results = load_existing_harp_results()

    dote_model = train_dote()
    all_results["dote_adapter"] = evaluate_adapter(dote_model, "dote")

    teal_model = train_teal()
    all_results["teal_adapter"] = evaluate_adapter(teal_model, "teal")

    for model_name, values_by_metric in all_results.items():
        write_distribution(model_name, values_by_metric)
    write_summary(all_results)

    print("Five-model dynamic Abilene resilience comparison")
    for model_name, values_by_metric in all_results.items():
        print(model_name)
        for metric in ["combined", "current", "expected_failure", "worst_failure"]:
            s = stats(values_by_metric[metric])
            print(
                f"  {metric}: avg={s['average']:.6f}, median={s['median']:.6f}, "
                f"p95={s['p95']:.6f}, max={s['max']:.6f}, n={s['count']}"
            )


if __name__ == "__main__":
    main()
