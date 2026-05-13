import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from frameworks.harp_baseline_system import BaselineHARP
from utils.args_parser import parse_args
from utils.training_utils import (
    DynamicAbileneDataset,
    dynamic_collate,
    find_first_nonfinite_gradient,
    find_first_nonfinite_parameter,
    move_dynamic_sample_to_device,
)


def parse_baseline_args():
    parser = argparse.ArgumentParser(
        description="Train/test snapshot-only HARP on final timesteps of dynamic samples."
    )

    parser.add_argument("--mode", choices=["train", "test"], required=True)
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

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Checkpoint path. Defaults to HARP_baseline_<topo>_pred_False_<K>sp.pkl.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory for test stats. Defaults to results/<topo>/<K>sp/<cluster>.",
    )

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
        "harp_baseline",
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
        f"HARP_baseline_{args.topo}_pred_False_"
        f"{args.num_paths_per_pair}sp.pkl"
    )


def final_timestep_view(sample):
    """
    Convert one dynamic temporal sample into the static HARP instance at t=-1.
    """
    final = {
        "node_features": sample["node_features"][:, -1, :, :],
        "edge_index": sample["edge_index"][-1],
        "capacities": sample["capacities"][-1],
        "padded_edge_ids_per_path": sample["padded_edge_ids_per_path"][-1],
        "paths_to_edges": sample["paths_to_edges"][-1],
        "tm": sample["tm"][:, -1, :, :],
        "tm_pred": sample["tm_pred"][:, -1, :, :],
        "opt": sample["opt"],
    }

    if final["capacities"].dim() == 1:
        final["capacities"] = final["capacities"].unsqueeze(0)

    return final


def baseline_mlu_loss(predicted, final_sample):
    if not torch.isfinite(predicted).all():
        raise RuntimeError("Non-finite predicted tensor reached baseline_mlu_loss")

    predicted = predicted.clamp_min(0.0)
    max_cong = predicted.max()

    opt = final_sample["opt"]
    if not torch.is_tensor(opt):
        opt = torch.tensor(opt, device=predicted.device, dtype=predicted.dtype)
    else:
        opt = opt.to(device=predicted.device, dtype=predicted.dtype)

    opt = opt.reshape(()).clamp_min(1e-12)
    loss = max_cong / opt

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite baseline loss computed")

    return (
        loss,
        float(loss.detach().cpu()),
        float(max_cong.detach().cpu()),
        float(final_sample["tm"].sum().detach().cpu()),
        float(predicted.sum().detach().cpu()),
    )


def run_model(model, props, final_sample):
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


def run_epoch(model, props, dataloader, optimizer, epoch, n_epochs):
    model.train()
    loss_values = []
    skipped = 0

    with tqdm(dataloader) as tepoch:
        tepoch.set_description(f"Baseline Epoch {epoch + 1}/{n_epochs}")

        for sample in tepoch:
            sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)
            final_sample = final_timestep_view(sample)

            bad_param = find_first_nonfinite_parameter(model)
            if bad_param is not None:
                raise RuntimeError(f"Non-finite model parameter before forward: {bad_param}")

            optimizer.zero_grad(set_to_none=True)
            predicted = run_model(model, props, final_sample)

            try:
                loss, loss_value, raw_mlu_value, tm_sum_value, total_util_value = baseline_mlu_loss(
                    predicted,
                    final_sample,
                )
            except RuntimeError as exc:
                skipped += 1
                print(f"[WARN] {exc}. Skipping baseline batch.")
                continue

            loss.backward()

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

            loss_values.append(loss_value)
            tepoch.set_postfix(
                loss=sum(loss_values) / len(loss_values),
                raw_mlu=raw_mlu_value,
                tm_sum=tm_sum_value,
                total_util=total_util_value,
                skipped=skipped,
            )

    if not loss_values:
        return float("nan")

    return sum(loss_values) / len(loss_values)


def evaluate(model, props, dataloader, description):
    model.eval()
    loss_values = []
    skipped = 0

    with torch.no_grad():
        with tqdm(dataloader) as vals:
            vals.set_description(description)

            for sample in vals:
                sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)
                final_sample = final_timestep_view(sample)

                bad_param = find_first_nonfinite_parameter(model)
                if bad_param is not None:
                    raise RuntimeError(f"Non-finite model parameter during eval: {bad_param}")

                predicted = run_model(model, props, final_sample)

                try:
                    loss, loss_value, raw_mlu_value, tm_sum_value, total_util_value = baseline_mlu_loss(
                        predicted,
                        final_sample,
                    )
                except RuntimeError as exc:
                    skipped += 1
                    print(f"[WARN] {exc}. Skipping baseline eval batch.")
                    continue

                loss_values.append(loss_value)
                vals.set_postfix(
                    loss=sum(loss_values) / len(loss_values),
                    raw_mlu=raw_mlu_value,
                    tm_sum=tm_sum_value,
                    total_util=total_util_value,
                    skipped=skipped,
                )

    if not loss_values:
        return float("nan"), [], skipped

    return sum(loss_values) / len(loss_values), loss_values, skipped


def write_stats(values, skipped, values_path, stats_path):
    os.makedirs(os.path.dirname(values_path), exist_ok=True)

    with open(values_path, "w") as f:
        for value in values:
            f.write(str(value) + "\n")

    dists = [float(v) for v in values]
    dists.sort()

    with open(stats_path, "w") as f:
        f.write("Average: " + str(statistics.mean(dists)) + "\n")
        f.write("Median: " + str(dists[int(len(dists) * 0.5)]) + "\n")
        f.write("25TH: " + str(dists[int(len(dists) * 0.25)]) + "\n")
        f.write("75TH: " + str(dists[int(len(dists) * 0.75)]) + "\n")
        f.write("90TH: " + str(dists[int(len(dists) * 0.90)]) + "\n")
        f.write("95TH: " + str(dists[int(len(dists) * 0.95)]) + "\n")
        f.write("99TH: " + str(dists[int(len(dists) * 0.99)]) + "\n")
        f.write("100TH: " + str(dists[int(len(dists) - 1)]) + "\n")
        f.write("Skipped: " + str(skipped) + "\n")


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


def main():
    args = parse_baseline_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(args, device)
    model_path = default_model_path(args)

    print("Running snapshot-only HARP baseline")
    print(f"Device: {device}")
    print(f"Samples dir: {args.samples_dir}")
    print(f"Model path: {model_path}")

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

        model = BaselineHARP(props).to(device=device, dtype=props.dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_avg = run_epoch(
                model=model,
                props=props,
                dataloader=train_dl,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=args.epochs,
            )

            val_avg, _, skipped = evaluate(
                model=model,
                props=props,
                dataloader=val_dl,
                description="Baseline Validation",
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Baseline train avg: {train_avg:.6f} | "
                f"Baseline val avg: {val_avg:.6f} | "
                f"Skipped val: {skipped}"
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

    test_avg, values, skipped = evaluate(
        model=model,
        props=props,
        dataloader=test_dl,
        description="Baseline Test",
    )

    print(f"Baseline Test Error:\nAvg loss: {test_avg:>8f}\n")

    if args.results_dir is None:
        results_dir = (
            f"results/{args.topo}/{args.num_paths_per_pair}sp/{args.test_cluster}"
        )
    else:
        results_dir = args.results_dir

    values_path = (
        f"{results_dir}/harp_baseline_dynamic_values_failure_id_{args.failure_id}.txt"
    )
    stats_path = (
        f"{results_dir}/harp_baseline_dynamic_stats_failure_id_{args.failure_id}.txt"
    )

    write_stats(values, skipped, values_path, stats_path)
    print(f"Wrote values: {values_path}")
    print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
