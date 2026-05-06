import sys
import os
import copy
import glob
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

cwd = os.getcwd()
sys.path.append(cwd + "/utils")

from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
from utils.training_utils import (
    create_dataloaders,
    validate,
    loss_mlu,
    move_to_device,
    train,
    DynamicAbileneDataset,
    dynamic_collate,
    move_dynamic_sample_to_device,
    train_dynamic,
    validate_dynamic,
    test_dynamic,
)
from frameworks.harp_system import HARP


def _strip_dynamic_args(argv):
    """
    The original HARP args_parser.py does not know about our new dynamic-sample
    arguments yet. This helper removes those args before calling parse_args(),
    then returns a Namespace with the dynamic args so we can attach them to props.

    This lets you run the dynamic pipeline by changing only run_harp.py and
    training_utils.py. Later, you can move these args into utils/args_parser.py.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    parser.add_argument("--dynamic_samples_dir", type=str, default=None)
    parser.add_argument("--dynamic_train_start_idx", type=int, default=0)
    parser.add_argument("--dynamic_train_end_idx", type=int, default=None)
    parser.add_argument("--dynamic_val_start_idx", type=int, default=None)
    parser.add_argument("--dynamic_val_end_idx", type=int, default=None)
    parser.add_argument("--dynamic_test_start_idx", type=int, default=0)
    parser.add_argument("--dynamic_test_end_idx", type=int, default=None)

    # If true and sample["opt"] exists, train/validate/test using normalized MLU.
    # If false or opt is absent, train/validate/test using raw MLU = predicted.max().
    parser.add_argument("--use_dynamic_opt", type=int, default=1)

    dynamic_args, remaining_argv = parser.parse_known_args(argv)
    return dynamic_args, remaining_argv


dynamic_args, remaining_argv = _strip_dynamic_args(sys.argv[1:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

props = parse_args(remaining_argv)
props.device = device
props_geant = copy.deepcopy(props)

# Attach dynamic args to props so training_utils can use them.
props.dynamic_samples_dir = dynamic_args.dynamic_samples_dir
props.dynamic_train_start_idx = dynamic_args.dynamic_train_start_idx
props.dynamic_train_end_idx = dynamic_args.dynamic_train_end_idx
props.dynamic_val_start_idx = dynamic_args.dynamic_val_start_idx
props.dynamic_val_end_idx = dynamic_args.dynamic_val_end_idx
props.dynamic_test_start_idx = dynamic_args.dynamic_test_start_idx
props.dynamic_test_end_idx = dynamic_args.dynamic_test_end_idx
props.use_dynamic_opt = bool(dynamic_args.use_dynamic_opt)

if props.dtype.lower() == "float32":
    props.dtype = torch.float32
elif props.dtype.lower() == "float16":
    # Original HARP code maps "float16" to bfloat16.
    props.dtype = torch.bfloat16
else:
    print("Only float32 and float16 are allowed")
    exit(1)

batch_size = props.batch_size
n_epochs = props.epochs


if props.mode.lower() == "train":
    model = HARP(props)
    model = model.to(device=device, dtype=props.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=props.lr)

    # ----------------------------------------------------------------------
    # New dynamic-Abilene path.
    # ----------------------------------------------------------------------
    if props.dynamic_samples_dir is not None:
        if props.batch_size != 1:
            raise ValueError(
                "Dynamic topology samples currently require --batch_size 1. "
                "Different samples can have different edge counts/path lengths, "
                "so normal PyTorch batching is not safe yet."
            )

        train_dataset = DynamicAbileneDataset(
            props.dynamic_samples_dir,
            start_idx=props.dynamic_train_start_idx,
            end_idx=props.dynamic_train_end_idx,
        )

        all_dynamic_files = sorted(
            glob.glob(str(Path(props.dynamic_samples_dir) / "sample_*.pt"))
        )

        if props.dynamic_val_start_idx is None:
            val_start = props.dynamic_train_end_idx
            if val_start is None:
                val_start = max(0, int(0.8 * len(all_dynamic_files)))
        else:
            val_start = props.dynamic_val_start_idx

        if props.dynamic_val_end_idx is None:
            val_end = len(all_dynamic_files)
        else:
            val_end = props.dynamic_val_end_idx

        val_dataset = DynamicAbileneDataset(
            props.dynamic_samples_dir,
            start_idx=val_start,
            end_idx=val_end,
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=dynamic_collate,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dynamic_collate,
        )

        print("Running dynamic Abilene training")
        print(f"Dynamic samples dir: {props.dynamic_samples_dir}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Using opt when available: {props.use_dynamic_opt}")

        for epoch in range(n_epochs):
            model.train()
            train_avg = train_dynamic(
                model=model,
                props=props,
                train_dl=train_dl,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
            )

            model.eval()
            val_avg = validate_dynamic(
                model=model,
                props=props,
                val_dl=val_dl,
            )

            print(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Dynamic train avg: {train_avg:.6f} | "
                f"Dynamic val avg: {val_avg:.6f}"
            )

            torch.save(
                model,
                f"HARP_dynamic_{props.topo}_pred_{props.pred}_{props.num_paths_per_pair}sp.pkl",
            )

        exit(0)

    # ----------------------------------------------------------------------
    # Original static HARP path.
    # ----------------------------------------------------------------------
    if props.meta_learning:
        print("Meta-Training Hattrick")
        rau = 7
        props_geant.topo = "geant"
        props_geant.num_paths_per_pair = 8
        props_geant.train_clusters = [0]
        props_geant.train_start_indices = [0]
        props_geant.train_end_indices = [6464]
        props_geant.epochs = 10
        props_geant.batch_size = 32
        props_geant.dynamic = 0
        props_geant.num_for_loops = rau
        props_geant.device = device
        props_geant.dtype = props.dtype

        geant_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        geant_ds_list, geant_dl_list = create_dataloaders(
            props_geant,
            props_geant.batch_size,
            training=True,
            shuffle=True,
        )

        for i in range(props_geant.epochs):
            train(
                model,
                props_geant,
                geant_ds_list,
                geant_dl_list,
                geant_optimizer,
                i,
                props_geant.epochs,
            )

        print("Meta-Training Done")

    ds_list, dl_list = create_dataloaders(
        props,
        batch_size,
        training=True,
        shuffle=True,
    )

    val_ds_list, val_dl_list = create_dataloaders(
        props,
        1,
        training=False,
        shuffle=False,
    )

    for epoch in range(n_epochs):
        model.train()
        train(model, props, ds_list, dl_list, optimizer, epoch, n_epochs)

        model.eval()
        for i in range(len(val_ds_list)):
            val_dataset = val_ds_list[i]
            val_dl = val_dl_list[i]

            val_dataset.pte = val_dataset.pte.to(
                device=props.device,
                dtype=props.dtype,
            )
            val_dataset.padded_edge_ids_per_path = (
                val_dataset.padded_edge_ids_per_path.to(device=props.device)
            )

            move_to_device(val_dataset.edge_ids_dict_tensor, props.device)
            move_to_device(val_dataset.original_pos_edge_ids_dict_tensor, props.device)

            val_norm_mlu = validate(model, props, val_dataset, val_dl)
            val_avg_loss = sum(val_norm_mlu) / len(val_norm_mlu)
            print(f"Validation Avg loss: {round(val_avg_loss, 5)}")

            val_dataset.pte = val_dataset.pte.to(device="cpu")
            val_dataset.padded_edge_ids_per_path = (
                val_dataset.padded_edge_ids_per_path.to(device="cpu")
            )
            move_to_device(val_dataset.edge_ids_dict_tensor, "cpu")
            move_to_device(val_dataset.original_pos_edge_ids_dict_tensor, "cpu")

        torch.save(
            model,
            f"HARP_{props.topo}_pred_{props.pred}_{props.num_paths_per_pair}sp.pkl",
        )


elif props.mode.lower() == "test":
    # ----------------------------------------------------------------------
    # New dynamic-Abilene test path.
    # ----------------------------------------------------------------------
    if props.dynamic_samples_dir is not None:
        test_dataset = DynamicAbileneDataset(
            props.dynamic_samples_dir,
            start_idx=props.dynamic_test_start_idx,
            end_idx=props.dynamic_test_end_idx,
        )

        test_dl = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=dynamic_collate,
        )

        model_path = (
            f"HARP_dynamic_{props.topo}_pred_{props.pred}_"
            f"{props.num_paths_per_pair}sp.pkl"
        )

        model = torch.load(model_path, map_location=device)
        model = model.to(device=device, dtype=props.dtype)
        model.eval()

        os.makedirs(
            f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}",
            exist_ok=True,
        )

        values_path = (
            f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}/"
            f"harp_dynamic_values_failure_id_{props.failure_id}.txt"
        )

        stats_path = (
            f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}/"
            f"harp_dynamic_stats_failure_id_{props.failure_id}.txt"
        )

        test_dynamic(
            model=model,
            props=props,
            test_dl=test_dl,
            values_path=values_path,
            stats_path=stats_path,
        )

        exit(0)

    # ----------------------------------------------------------------------
    # Original static HARP test path.
    # ----------------------------------------------------------------------
    cluster = props.test_cluster
    start = props.test_start_idx
    end = props.test_end_idx

    test_dataset = DM_Dataset_within_Cluster(props, cluster, start, end)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_dataset.pte = test_dataset.pte.to(props.device, dtype=props.dtype)
    test_dataset.padded_edge_ids_per_path = (
        test_dataset.padded_edge_ids_per_path.to(device)
    )

    move_to_device(test_dataset.edge_ids_dict_tensor, props.device)
    move_to_device(test_dataset.original_pos_edge_ids_dict_tensor, props.device)

    model = torch.load(
        f"HARP_{props.topo}_pred_{props.pred}_{props.num_paths_per_pair}sp.pkl",
        map_location=device,
    )
    model = model.to(dtype=props.dtype)
    model.eval()

    with torch.no_grad():
        with tqdm(test_dl) as tests:
            test_losses = []

            file = open(
                f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}/"
                f"harp_values_{cluster}_failure_id_{props.failure_id}.txt",
                "w",
            )

            for inputs in tests:
                node_features, capacities, tms, tms_pred, opt = inputs

                node_features = node_features.to(device=device, dtype=props.dtype)
                capacities = capacities.to(device=device, dtype=props.dtype)
                tms = tms.to(device=device, dtype=props.dtype)
                opt = opt.to(device=device, dtype=props.dtype)

                if props.pred:
                    tms_pred = tms_pred.to(device=device, dtype=props.dtype)

                    predicted = model(
                        props,
                        node_features,
                        test_dataset.edge_index,
                        capacities,
                        test_dataset.padded_edge_ids_per_path,
                        tms,
                        tms_pred,
                        test_dataset.pte,
                        test_dataset.edge_ids_dict_tensor,
                        test_dataset.original_pos_edge_ids_dict_tensor,
                    )
                else:
                    predicted = model(
                        props,
                        node_features,
                        test_dataset.edge_index,
                        capacities,
                        test_dataset.padded_edge_ids_per_path,
                        tms,
                        tms,
                        test_dataset.pte,
                        test_dataset.edge_ids_dict_tensor,
                        test_dataset.original_pos_edge_ids_dict_tensor,
                    )

                test_loss, test_loss_value = loss_mlu(predicted, opt)
                test_losses.append(test_loss_value)
                file.write(str(test_loss_value) + "\n")

            avg_loss = sum(test_losses) / len(test_losses)
            print(f"Test Error: \nAvg loss: {avg_loss:>8f} \n")
            file.close()

            with open(
                f"results/{props.topo}/{props.num_paths_per_pair}sp/{props.test_cluster}/"
                f"harp_stats_{cluster}_failure_id_{props.failure_id}.txt",
                "w",
            ) as f:
                import statistics

                dists = [float(v) for v in test_losses]
                dists.sort()

                f.write("Average: " + str(statistics.mean(dists)) + "\n")
                f.write("Median: " + str(dists[int(len(dists) * 0.5)]) + "\n")
                f.write("25TH: " + str(dists[int(len(dists) * 0.25)]) + "\n")
                f.write("75TH: " + str(dists[int(len(dists) * 0.75)]) + "\n")
                f.write("90TH: " + str(dists[int(len(dists) * 0.90)]) + "\n")
                f.write("95TH: " + str(dists[int(len(dists) * 0.95)]) + "\n")
                f.write("99TH: " + str(dists[int(len(dists) * 0.99)]) + "\n")
                f.write("100TH: " + str(dists[int(len(dists) - 1)]) + "\n")
