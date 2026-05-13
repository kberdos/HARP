from contextlib import nullcontext

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from frameworks.harp_system import GNN, TransformerModel, epsilon

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def sdpa_math_context():
        return sdpa_kernel([SDPBackend.MATH])

except ModuleNotFoundError:
    def sdpa_math_context():
        return nullcontext()


class BaselineHARP(nn.Module):
    """
    Snapshot-only HARP baseline for dynamic Abilene samples.

    This intentionally omits the temporal path encoder. It follows the original
    HARP structure from the pre-temporal code path: GNN edge embeddings, a Set
    Transformer over each path's edge set, MLP split logits, and RAU-style
    refinements. The runner feeds only the final timestep from each dynamic
    sample, so this baseline uses the same data/labels as GRATE without history.
    """

    def __init__(self, props):
        super().__init__()

        self.num_gnn_layers = props.num_gnn_layers
        self.num_transformer_layers = props.num_transformer_layers
        self.dropout = props.dropout
        self.num_mlp1_hidden_layers = props.num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = props.num_mlp2_hidden_layers
        self.device = props.device

        self.gnn = GNN(2, self.num_gnn_layers)
        self.input_dim = self.gnn.output_dim + 1

        self.cls_token = nn.Parameter(torch.Tensor(1, self.input_dim))
        nn.init.kaiming_normal_(self.cls_token, nonlinearity="relu")

        if props.num_heads == 0:
            num_heads = self.input_dim // 4
        else:
            num_heads = props.num_heads

        self.transformer = TransformerModel(
            in_dim=self.input_dim,
            nhead=num_heads,
            dim_feedforward=self.input_dim,
            nlayers=self.num_transformer_layers,
            dropout=self.dropout,
            activation="gelu",
        )

        self.mlp_1_dim = self.input_dim + 1
        self.mlp1 = nn.ModuleList()
        self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))
        for _ in range(self.num_mlp1_hidden_layers):
            self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))
        self.mlp1.append(nn.Linear(self.mlp_1_dim, 1))

        self.mlp_2_dim = self.input_dim + 3
        self.mlp2 = nn.ModuleList()
        self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        for _ in range(self.num_mlp2_hidden_layers):
            self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        self.mlp2.append(nn.Linear(self.mlp_2_dim, 1))

    def forward(
        self,
        props,
        node_features,
        edge_index,
        capacities,
        padded_edge_ids_per_path,
        tm,
        tm_pred,
        paths_to_edges,
    ):
        num_for_loops = props.num_for_loops
        num_paths_per_pair = props.num_paths_per_pair
        batch_size = tm.shape[0]

        if capacities.dim() == 1:
            capacities = capacities.unsqueeze(0)

        if props.checkpoint:
            edge_embeddings_with_caps = checkpoint(
                self.gnn,
                node_features,
                edge_index,
                capacities,
                use_reentrant=False,
            )
        else:
            edge_embeddings_with_caps = self.gnn(
                node_features,
                edge_index,
                capacities,
            )

        (
            path_embeddings,
            path_edge_embeddings,
        ) = self.compute_path_embeddings(
            edge_embeddings_with_caps=edge_embeddings_with_caps,
            padded_edge_ids_per_path=padded_edge_ids_per_path,
            props=props,
        )

        tm = self.normalize_tm(tm)
        tm_pred = self.normalize_tm(tm_pred)

        mlp1_inputs = torch.cat((path_embeddings, tm_pred), dim=-1)
        gammas = self.forward_pass_mlp(
            mlp1_inputs,
            self.mlp1,
            self.num_mlp1_hidden_layers,
        )

        paths_to_edges = paths_to_edges.coalesce()
        total_number_of_paths = paths_to_edges.shape[0]

        for i in range(num_for_loops):
            if i > 0:
                gammas = new_gammas

            edges_util = self.compute_edge_utils(
                gammas=gammas,
                paths_to_edges=paths_to_edges,
                tm=tm_pred,
                capacities=capacities,
                props=props,
                batch_size=batch_size,
                num_paths_per_pair=num_paths_per_pair,
                add_epsilon=True,
            )

            mlu = self.compute_mlu(
                edges_util=edges_util,
                batch_size=batch_size,
                total_number_of_paths=total_number_of_paths,
                subtract_epsilon=True,
            )

            (
                bottleneck_path_edge_embeddings,
                max_utilization_per_path,
            ) = self.compute_bottleneck_link_mlu_per_path(
                edge_utils=edges_util,
                padded_edge_ids_per_path=padded_edge_ids_per_path,
                path_edge_embeddings=path_edge_embeddings,
                batch_size=batch_size,
                total_number_of_paths=total_number_of_paths,
            )

            mlp2_inputs = torch.cat(
                (
                    bottleneck_path_edge_embeddings,
                    max_utilization_per_path,
                    mlu,
                    tm_pred,
                ),
                dim=-1,
            )

            delta_gammas = self.forward_pass_mlp(
                mlp2_inputs,
                self.mlp2,
                self.num_mlp2_hidden_layers,
            )

            gammas = gammas.reshape(batch_size, -1, 1)
            new_gammas = delta_gammas + gammas

        if num_for_loops == 0:
            new_gammas = gammas

        return self.compute_edge_utils(
            gammas=new_gammas,
            paths_to_edges=paths_to_edges,
            tm=tm,
            capacities=capacities,
            props=props,
            batch_size=batch_size,
            num_paths_per_pair=num_paths_per_pair,
            add_epsilon=False,
        )

    def normalize_tm(self, tm: Tensor):
        if tm.dim() == 2:
            return tm.unsqueeze(0)

        if tm.dim() == 3:
            return tm

        raise ValueError(f"Baseline HARP expects final-timestep TM, got {tuple(tm.shape)}")

    def compute_path_embeddings(
        self,
        edge_embeddings_with_caps: Tensor,
        padded_edge_ids_per_path: Tensor,
        props,
    ):
        safe_edge_ids = padded_edge_ids_per_path.clamp(min=0)
        path_mask = padded_edge_ids_per_path.eq(-1)

        batch_size, _, feat_dim = edge_embeddings_with_caps.shape
        total_number_of_paths, max_path_length = safe_edge_ids.shape

        gather_indices = safe_edge_ids.view(
            1,
            total_number_of_paths,
            max_path_length,
            1,
        ).expand(
            batch_size,
            -1,
            -1,
            feat_dim,
        )

        expanded_edges = edge_embeddings_with_caps.unsqueeze(1).expand(
            -1,
            total_number_of_paths,
            -1,
            -1,
        )

        path_edge_inputs = torch.gather(
            expanded_edges,
            dim=2,
            index=gather_indices,
        )

        path_edge_inputs = path_edge_inputs.masked_fill(
            path_mask.view(1, total_number_of_paths, max_path_length, 1),
            0.0,
        )

        cls_token = self.cls_token.view(1, 1, 1, self.input_dim).expand(
            batch_size,
            total_number_of_paths,
            -1,
            -1,
        )

        transformer_inputs = torch.cat((cls_token, path_edge_inputs), dim=2)
        transformer_inputs = transformer_inputs.reshape(
            batch_size * total_number_of_paths,
            max_path_length + 1,
            self.input_dim,
        )

        cls_mask = torch.zeros(
            total_number_of_paths,
            1,
            device=path_mask.device,
            dtype=torch.bool,
        )
        full_path_mask = torch.cat((cls_mask, path_mask), dim=1)
        full_path_mask = full_path_mask.view(1, total_number_of_paths, -1).expand(
            batch_size,
            -1,
            -1,
        )
        full_path_mask = full_path_mask.reshape(
            batch_size * total_number_of_paths,
            max_path_length + 1,
        )

        with sdpa_math_context():
            transformer_output = self.transformer(
                transformer_inputs,
                src_key_padding_mask=full_path_mask,
            )

        transformer_output = transformer_output.reshape(
            batch_size,
            total_number_of_paths,
            max_path_length + 1,
            self.input_dim,
        )

        return transformer_output[:, :, 0, :], transformer_output[:, :, 1:, :]

    def compute_mlu(
        self,
        edges_util,
        batch_size,
        total_number_of_paths,
        subtract_epsilon=True,
    ):
        mlu, _ = torch.max(edges_util, dim=-1)
        if subtract_epsilon:
            mlu = mlu - epsilon
        return mlu.view(batch_size, 1, 1).expand(-1, total_number_of_paths, -1)

    def compute_bottleneck_link_mlu_per_path(
        self,
        edge_utils,
        padded_edge_ids_per_path,
        path_edge_embeddings,
        batch_size,
        total_number_of_paths,
    ):
        device = edge_utils.device
        padded_edge_ids_per_path = padded_edge_ids_per_path.to(device=device)

        P, L = padded_edge_ids_per_path.shape
        B, E = edge_utils.shape

        if P != total_number_of_paths:
            raise ValueError(
                f"Path count mismatch: padded paths has {P}, expected {total_number_of_paths}"
            )

        valid_mask = padded_edge_ids_per_path.ge(0)
        safe_edge_ids = padded_edge_ids_per_path.clamp(min=0)

        gather_ids = safe_edge_ids.view(1, P, L).expand(B, -1, -1)
        expanded_edge_utils = edge_utils.view(B, 1, E).expand(-1, P, -1)
        gathered_utils = torch.gather(expanded_edge_utils, dim=2, index=gather_ids)
        gathered_utils = gathered_utils.masked_fill(
            ~valid_mask.view(1, P, L),
            float("-inf"),
        )

        max_utilization_per_path, bottleneck_positions = gathered_utils.max(dim=2)

        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, P)
        path_indices = torch.arange(P, device=device).view(1, P).expand(B, P)
        bottleneck_path_edge_embeddings = path_edge_embeddings[
            batch_indices,
            path_indices,
            bottleneck_positions,
        ]

        return (
            bottleneck_path_edge_embeddings,
            (max_utilization_per_path - epsilon).unsqueeze(-1),
        )

    def forward_pass_mlp(self, inputs, mlp: nn.ModuleList, num_hidden_layers):
        for index, layer in enumerate(mlp):
            if index == 0:
                out = F.leaky_relu(layer(inputs), 0.02)
            elif index == (num_hidden_layers + 1):
                out = layer(out)
            else:
                out = F.leaky_relu(layer(out), 0.02)
        return out

    def compute_split_ratios(self, gammas, batch_size, num_paths_per_pair):
        gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)
        split_ratios = torch.softmax(gammas, dim=-1)
        return split_ratios.reshape(batch_size, -1)

    def compute_edge_utils(
        self,
        gammas,
        paths_to_edges,
        tm,
        capacities,
        props,
        batch_size,
        num_paths_per_pair,
        add_epsilon=True,
    ):
        split_ratios = self.compute_split_ratios(
            gammas,
            batch_size,
            num_paths_per_pair,
        )

        data_on_tunnels = split_ratios * tm.squeeze(-1)

        data_on_links = torch.sparse.mm(
            paths_to_edges.to(dtype=torch.float32).t(),
            data_on_tunnels.to(dtype=torch.float32).t(),
        ).t()

        if props.dtype == torch.bfloat16:
            data_on_links = data_on_links.to(dtype=torch.bfloat16)

        capacities = capacities.clamp_min(1e-6)

        if add_epsilon:
            return data_on_links / capacities + epsilon

        return data_on_links / capacities
