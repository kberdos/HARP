import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from frameworks.harp_baseline_system import BaselineHARP
from frameworks.harp_system import HARP
from utils.args_parser import parse_args
from utils.resilience_utils import (
    resilience_loss_from_details,
    write_distribution_stats,
)
from utils.training_utils import (
    DynamicAbileneDataset,
    dynamic_collate,
    find_first_nonfinite_gradient,
    find_first_nonfinite_parameter,
    move_dynamic_sample_to_device,
)


def parse_resilience_args():
    parser = argparse.ArgumentParser(
        description="Train/test HARP with a future-failure resiliency objective."
    )

    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--model_type", choices=["temporal", "baseline"], default="temporal")
    parser.add_argument("--samples_dir", type=str, default="dynamic_abilene_h6_1000_samples")
    parser.add_argument("--topo", type=str, default="dynamic_abilene")
    parser.add_argument("--test_cluster", type=int, default=0)
    parser.add_argument("--failure_id", type=str, default="None")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--train_start_idx", type=int, default=0)
    parser.add_argument("--train_end_idx", type=int, default=800)
    parser.add_argument("--val_start_idx", type=int, default=800)
    parser.add_argument("--val_end_idx", type=int, default=1000)
    parser.add_argument("--test_start_idx", type=int, default=800)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    parser.add_argument("--num_paths_per_pair", type=int, default=4)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=0)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_mlp1_hidden_layers", type=int, default=1)
    parser.add_argument("--num_mlp2_hidden_layers", type=int, default=1)
    parser.add_argument("--num_for_loops", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--checkpoint", type=int, default=0)

    parser.add_argument("--current_weight", type=float, default=1.0)
    parser.add_argument("--resilience_weight", type=float, default=0.25)
    parser.add_argument("--worst_case_weight", type=float, default=0.5)
    parser.add_argument("--failure_capacity_fraction", type=float, default=0.25)
    parser.add_argument("--risk_prior", type=float, default=1.0)
    parser.add_argument("--risk_recency_power", type=float, default=2.0)
    parser.add_argument(
        "--scenario_top_k",
        type=int,
        default=0,
        help="Use top-k risky final edges. 0 means all final edges.",
    )

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)

    return parser.parse_args()


def build_props(args, device):
    parser_args = [
        "--topo",
        args.topo,
        "--mode",
        args.mode,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch_size",
        "1",
        "--num_paths_per_pair",
        str(args.num_paths_per_pair),
        "--num_transformer_layers",
        str(args.num_transformer_layers),
        "--num_heads",
        str(args.num_heads),
        "--num_gnn_layers",
        str(args.num_gnn_layers),
        "--num_mlp1_hidden_layers",
        str(args.num_mlp1_hidden_layers),
        "--num_mlp2_hidden_layers",
        str(args.num_mlp2_hidden_layers),
        "--num_for_loops",
        str(args.num_for_loops),
        "--dropout",
        str(args.dropout),
        "--framework",
        "harp",
        "--pred",
        "0",
        "--dynamic",
        "1",
        "--dtype",
        args.dtype,
        "--checkpoint",
        str(args.checkpoint),
    ]

    props = parse_args(parser_args)
    props.device = device
    props.return_details = True

    if args.dtype.lower() == "float32":
        props.dtype = torch.float32
    elif args.dtype.lower() == "float16":
        props.dtype = torch.bfloat16
    else:
        raise ValueError("Only float32 and float16 are allowed")

    return props


def default_model_path(args):
    if args.model_path is not None:
        return args.model_path

    return (
        f"HARP_resilient_{args.model_type}_{args.topo}_pred_False_"
        f"{args.num_paths_per_pair}sp.pkl"
    )


def final_timestep_view(sample):
    final = {
        "node_features": sample["node_features"][:, -1, :, :],
        "edge_index": sample["edge_index"][-1],
        "capacities": sample["capacities"][-1],
        "padded_edge_ids_per_path": sample["padded_edge_ids_per_path"][-1],
        "paths_to_edges": sample["paths_to_edges"][-1],
        "tm": sample["tm"][:, -1, :, :],
        "tm_pred": sample["tm_pred"][:, -1, :, :],
        "opt": sample["opt"],
        "metadata": sample.get("metadata", {}),
    }

    if final["capacities"].dim() == 1:
        final["capacities"] = final["capacities"].unsqueeze(0)

    return final


def build_model(args, props):
    if args.model_type == "temporal":
        return HARP(props)

    return BaselineHARP(props)


def run_model(model, props, sample, model_type):
    if model_type == "temporal":
        return model(
            props,
            sample["node_features"],
            sample["edge_index"],
            sample["capacities"],
            sample["padded_edge_ids_per_path"],
            sample["tm"],
            sample["tm"],
            sample["paths_to_edges"],
            None,
            None,
        )

    final_sample = final_timestep_view(sample)
    return model(
        props,
        final_sample["node_features"],
        final_sample["edge_index"],
        final_sample["capacities"],
        final_sample["padded_edge_ids_per_path"],
        final_sample["tm"],
        final_sample["tm_pred"],
        final_sample["paths_to_edges"],
    )


def compute_resilience_metrics(details, sample, args):
    return resilience_loss_from_details(
        details=details,
        sample=sample,
        current_weight=args.current_weight,
        resilience_weight=args.resilience_weight,
        worst_case_weight=args.worst_case_weight,
        failure_capacity_fraction=args.failure_capacity_fraction,
        risk_prior=args.risk_prior,
        risk_recency_power=args.risk_recency_power,
        scenario_top_k=args.scenario_top_k,
    )


def make_loader(samples_dir, start_idx, end_idx, shuffle):
    dataset = DynamicAbileneDataset(
        samples_dir,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=dynamic_collate,
    )


def train_one_epoch(model, props, args, dataloader, optimizer, epoch, n_epochs):
    model.train()
    combined_values = []
    skipped = 0

    with tqdm(dataloader) as tepoch:
        tepoch.set_description(f"Resilient Epoch {epoch + 1}/{n_epochs}")

        for sample in tepoch:
            sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

            bad_param = find_first_nonfinite_parameter(model)
            if bad_param is not None:
                raise RuntimeError(f"Non-finite model parameter before forward: {bad_param}")

            optimizer.zero_grad(set_to_none=True)
            details = run_model(model, props, sample, args.model_type)

            try:
                metrics = compute_resilience_metrics(details, sample, args)
            except RuntimeError as exc:
                skipped += 1
                print(f"[WARN] {exc}. Skipping resilient batch.")
                continue

            metrics["loss"].backward()

            bad_grad = find_first_nonfinite_gradient(model)
            if bad_grad is not None:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    "[WARN] Non-finite gradient detected for "
                    f"{bad_grad}. Skipping optimizer step."
                )
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bad_param = find_first_nonfinite_parameter(model)
            if bad_param is not None:
                raise RuntimeError(f"Optimizer step produced non-finite parameter: {bad_param}")

            combined_values.append(metrics["combined_value"])
            tepoch.set_postfix(
                combined=sum(combined_values) / len(combined_values),
                current=metrics["current_norm"],
                expected=metrics["expected_failure_norm"],
                worst=metrics["worst_failure_norm"],
                skipped=skipped,
            )

    if not combined_values:
        return float("nan")

    return sum(combined_values) / len(combined_values)


def evaluate(model, props, args, dataloader, description):
    model.eval()
    skipped = 0
    combined_values = []
    current_values = []
    expected_failure_values = []
    worst_failure_values = []

    with torch.no_grad():
        with tqdm(dataloader) as vals:
            vals.set_description(description)

            for sample in vals:
                sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

                bad_param = find_first_nonfinite_parameter(model)
                if bad_param is not None:
                    raise RuntimeError(f"Non-finite model parameter during eval: {bad_param}")

                details = run_model(model, props, sample, args.model_type)

                try:
                    metrics = compute_resilience_metrics(details, sample, args)
                except RuntimeError as exc:
                    skipped += 1
                    print(f"[WARN] {exc}. Skipping resilient eval batch.")
                    continue

                combined_values.append(metrics["combined_value"])
                current_values.append(metrics["current_norm"])
                expected_failure_values.append(metrics["expected_failure_norm"])
                worst_failure_values.append(metrics["worst_failure_norm"])

                vals.set_postfix(
                    combined=sum(combined_values) / len(combined_values),
                    current=metrics["current_norm"],
                    expected=metrics["expected_failure_norm"],
                    worst=metrics["worst_failure_norm"],
                    scenarios=metrics["num_scenarios"],
                    skipped=skipped,
                )

    if not combined_values:
        return {
            "combined_avg": float("nan"),
            "combined": [],
            "current": [],
            "expected_failure": [],
            "worst_failure": [],
            "skipped": skipped,
        }

    return {
        "combined_avg": sum(combined_values) / len(combined_values),
        "combined": combined_values,
        "current": current_values,
        "expected_failure": expected_failure_values,
        "worst_failure": worst_failure_values,
        "skipped": skipped,
    }


def write_resilience_outputs(results, args):
    if args.results_dir is None:
        results_dir = f"results/{args.topo}/{args.num_paths_per_pair}sp/{args.test_cluster}"
    else:
        results_dir = args.results_dir

    os.makedirs(results_dir, exist_ok=True)

    prefix = f"harp_resilient_{args.model_type}_dynamic_failure_id_{args.failure_id}"

    distributions = {
        "combined": results["combined"],
        "current": results["current"],
        "expected_failure": results["expected_failure"],
        "worst_failure": results["worst_failure"],
    }

    for name, values in distributions.items():
        values_path = f"{results_dir}/{prefix}_{name}_values.txt"
        stats_path = f"{results_dir}/{prefix}_{name}_stats.txt"

        with open(values_path, "w") as f:
            for value in values:
                f.write(str(value) + "\n")

        write_distribution_stats(values, results["skipped"], stats_path)
        print(f"Wrote {name} values: {values_path}")
        print(f"Wrote {name} stats: {stats_path}")


def main():
    args = parse_resilience_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(args, device)
    model_path = default_model_path(args)

    print("Running future-failure resilient HARP")
    print(f"Mode: {args.mode}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    print(f"Samples dir: {args.samples_dir}")
    print(f"Model path: {model_path}")
    print(
        "Objective: "
        f"{args.current_weight} * current_norm + "
        f"{args.resilience_weight} * "
        f"((1 - {args.worst_case_weight}) * expected_failure_norm + "
        f"{args.worst_case_weight} * worst_failure_norm)"
    )
    print(f"Failure capacity fraction: {args.failure_capacity_fraction}")
    print(f"Risk prior: {args.risk_prior}")
    print(f"Risk recency power: {args.risk_recency_power}")
    print(f"Scenario top-k: {args.scenario_top_k if args.scenario_top_k > 0 else 'all'}")

    if args.mode == "train":
        train_dl = make_loader(
            args.samples_dir,
            args.train_start_idx,
            args.train_end_idx,
            shuffle=True,
        )
        val_dl = make_loader(
            args.samples_dir,
            args.val_start_idx,
            args.val_end_idx,
            shuffle=False,
        )

        model = build_model(args, props).to(device=device, dtype=props.dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_avg = train_one_epoch(
                model=model,
                props=props,
                args=args,
                dataloader=train_dl,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=args.epochs,
            )

            val_results = evaluate(
                model=model,
                props=props,
                args=args,
                dataloader=val_dl,
                description="Resilient Validation",
            )

            current_avg = (
                sum(val_results["current"]) / len(val_results["current"])
                if val_results["current"]
                else float("nan")
            )
            expected_avg = (
                sum(val_results["expected_failure"]) / len(val_results["expected_failure"])
                if val_results["expected_failure"]
                else float("nan")
            )
            worst_avg = (
                sum(val_results["worst_failure"]) / len(val_results["worst_failure"])
                if val_results["worst_failure"]
                else float("nan")
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train combined avg: {train_avg:.6f} | "
                f"Val combined avg: {val_results['combined_avg']:.6f} | "
                f"Val current avg: {current_avg:.6f} | "
                f"Val expected failure avg: {expected_avg:.6f} | "
                f"Val worst failure avg: {worst_avg:.6f} | "
                f"Skipped val: {val_results['skipped']}"
            )

            torch.save(model, model_path)

        return

    test_dl = make_loader(
        args.samples_dir,
        args.test_start_idx,
        args.test_end_idx,
        shuffle=False,
    )

    model = torch.load(model_path, map_location=device)
    model = model.to(device=device, dtype=props.dtype)

    test_results = evaluate(
        model=model,
        props=props,
        args=args,
        dataloader=test_dl,
        description="Resilient Test",
    )

    print(f"Resilient Test Error:\nCombined avg: {test_results['combined_avg']:>8f}\n")
    write_resilience_outputs(test_results, args)


if __name__ == "__main__":
    main()
