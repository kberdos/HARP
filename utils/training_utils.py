import glob
import statistics
from pathlib import Path

import torch
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


def dynamic_mlu_loss(predicted, sample, use_opt=True):
    """
    Dynamic-sample loss.

    If sample["opt"] exists and use_opt=True:
        optimize normalized MLU = model_mlu / optimal_mlu.

    If no opt exists:
        optimize raw MLU, plus an anti-collapse term so the model cannot
        make the loss look good by producing all-zero utilization while the
        traffic matrix is nonzero.

    Important:
        This function intentionally does NOT silently convert NaN/Inf predicted
        tensors into zero. If predicted is non-finite, training should skip or
        crash loudly rather than hide the bug.
    """
    if not torch.isfinite(predicted).all():
        raise RuntimeError("Non-finite predicted tensor reached dynamic_mlu_loss")

    # Utilization should be nonnegative. Clamp tiny negative values that can
    # come from numerical weirdness, but do not hide NaN/Inf above.
    predicted = predicted.clamp_min(0.0)

    max_cong = predicted.max()
    total_util = predicted.sum()

    if "tm" in sample and torch.is_tensor(sample["tm"]):
        tm_final_sum = sample["tm"][:, -1].sum()
    else:
        tm_final_sum = torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype)

    opt = sample.get("opt", None)

    if use_opt and opt is not None:
        if not torch.is_tensor(opt):
            opt = torch.tensor(opt, device=predicted.device, dtype=predicted.dtype)
        else:
            opt = opt.to(device=predicted.device, dtype=predicted.dtype)

        opt = opt.reshape(()).clamp_min(1e-12)
        base_loss = max_cong / opt
    else:
        base_loss = max_cong

    # Temporary no-Gurobi guardrail:
    # If there is real demand, total utilization should not be exactly zero.
    # This does not make the fallback objective perfect; it only prevents the
    # degenerate all-zero solution while we add true Gurobi opt values later.
    has_traffic = tm_final_sum > 1e-8

    zero_collapse_penalty = torch.where(
        has_traffic,
        torch.relu(torch.tensor(1e-4, device=predicted.device, dtype=predicted.dtype) - total_util) * 1000.0,
        torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype),
    )

    # Smooth anti-collapse term. Small enough that MLU is still the main goal.
    anti_collapse_term = torch.where(
        has_traffic,
        -0.001 * torch.log(total_util.clamp_min(1e-6)),
        torch.tensor(0.0, device=predicted.device, dtype=predicted.dtype),
    )

    loss = base_loss + anti_collapse_term + zero_collapse_penalty

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite dynamic loss computed")

    loss_value = float(loss.detach().cpu())
    raw_mlu_value = float(max_cong.detach().cpu())
    total_util_value = float(total_util.detach().cpu())
    tm_sum_value = float(tm_final_sum.detach().cpu())

    return loss, loss_value, raw_mlu_value, total_util_value, tm_sum_value


class DynamicAbileneDataset(Dataset):
    """
    Loads generated dynamic Abilene .pt samples.

    Expected sample keys:
        node_features: [1, T, N, 2]
        edge_index: list length T, each [2, E_t]
        capacities: list length T, each [1, E_t] or [E_t]
        padded_edge_ids_per_path: list length T, each [P, L_t]
        paths_to_edges: list length T, each sparse [P, E_t]
        tm: [1, T, P, 1]
        tm_pred: [1, T, P, 1]
        metadata: optional dict
        opt: optional scalar optimal MLU for the final timestep
    """

    def __init__(self, samples_dir, start_idx=0, end_idx=None):
        self.samples_dir = Path(samples_dir)
        self.files = sorted(glob.glob(str(self.samples_dir / "sample_*.pt")))

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
        return torch.load(self.files[idx], map_location="cpu")


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
            if key in ["node_features", "tm", "tm_pred", "opt"]:
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
    """
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


def train_dynamic(model, props, train_dl, optimizer, epoch, n_epochs):
    loss_values = []
    skipped = 0

    with tqdm(train_dl) as tepoch:
        tepoch.set_description(f"Dynamic Epoch {epoch + 1}/{n_epochs}")

        for sample in tepoch:
            sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

            optimizer.zero_grad(set_to_none=True)

            predicted = run_model_on_dynamic_sample(model, props, sample)

            # Do not let bad outputs get silently converted to zero.
            if not torch.isfinite(predicted).all():
                skipped += 1
                print("[WARN] Non-finite predicted output detected. Skipping batch.")
                continue

            try:
                loss, loss_value, raw_mlu_value, total_util_value, tm_sum_value = dynamic_mlu_loss(
                    predicted,
                    sample,
                    use_opt=getattr(props, "use_dynamic_opt", True),
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

            loss_values.append(loss_value)
            avg_loss = sum(loss_values) / len(loss_values)
            tepoch.set_postfix(
                loss=avg_loss,
                raw_mlu=raw_mlu_value,
                total_util=total_util_value,
                tm_sum=tm_sum_value,
                skipped=skipped,
            )

    if len(loss_values) == 0:
        return float("nan")

    return sum(loss_values) / len(loss_values)


def validate_dynamic(model, props, val_dl):
    loss_values = []
    skipped = 0

    with torch.no_grad():
        with tqdm(val_dl) as vals:
            vals.set_description("Dynamic Validation")

            for sample in vals:
                sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

                predicted = run_model_on_dynamic_sample(model, props, sample)

                if not torch.isfinite(predicted).all():
                    skipped += 1
                    print("[WARN] Non-finite predicted output detected during validation. Skipping batch.")
                    continue

                try:
                    loss, loss_value, raw_mlu_value, total_util_value, tm_sum_value = dynamic_mlu_loss(
                        predicted,
                        sample,
                        use_opt=getattr(props, "use_dynamic_opt", True),
                    )
                except RuntimeError as exc:
                    skipped += 1
                    print(f"[WARN] {exc}. Skipping validation batch.")
                    continue

                if not torch.isfinite(loss):
                    skipped += 1
                    print("[WARN] Non-finite dynamic validation loss detected. Skipping batch.")
                    continue

                loss_values.append(loss_value)
                avg_loss = sum(loss_values) / len(loss_values)
                vals.set_postfix(
                    loss=avg_loss,
                    raw_mlu=raw_mlu_value,
                    total_util=total_util_value,
                    tm_sum=tm_sum_value,
                    skipped=skipped,
                )

    if len(loss_values) == 0:
        return float("nan")

    return sum(loss_values) / len(loss_values)


def test_dynamic(model, props, test_dl, values_path, stats_path):
    loss_values = []
    skipped = 0

    with torch.no_grad():
        with tqdm(test_dl) as tests:
            tests.set_description("Dynamic Test")

            with open(values_path, "w") as values_file:
                for sample in tests:
                    sample = move_dynamic_sample_to_device(sample, props.device, props.dtype)

                    predicted = run_model_on_dynamic_sample(model, props, sample)

                    if not torch.isfinite(predicted).all():
                        skipped += 1
                        print("[WARN] Non-finite predicted output detected during test. Skipping batch.")
                        continue

                    try:
                        loss, loss_value, raw_mlu_value, total_util_value, tm_sum_value = dynamic_mlu_loss(
                            predicted,
                            sample,
                            use_opt=getattr(props, "use_dynamic_opt", True),
                        )
                    except RuntimeError as exc:
                        skipped += 1
                        print(f"[WARN] {exc}. Skipping test batch.")
                        continue

                    if not torch.isfinite(loss):
                        skipped += 1
                        print("[WARN] Non-finite dynamic test loss detected. Skipping batch.")
                        continue

                    loss_values.append(loss_value)
                    values_file.write(str(loss_value) + "\n")
                    avg_loss = sum(loss_values) / len(loss_values)
                    tests.set_postfix(
                        loss=avg_loss,
                        raw_mlu=raw_mlu_value,
                        total_util=total_util_value,
                        tm_sum=tm_sum_value,
                        skipped=skipped,
                    )

    if len(loss_values) == 0:
        avg_loss = float("nan")
        print("Dynamic Test Error:\nAvg loss: nan\n")
        return avg_loss

    avg_loss = sum(loss_values) / len(loss_values)
    print(f"Dynamic Test Error:\nAvg loss: {avg_loss:>8f}\n")

    dists = [float(v) for v in loss_values]
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
