from contextlib import nullcontext

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend

    def sdpa_math_context():
        return sdpa_kernel([SDPBackend.MATH])

except ModuleNotFoundError:
    def sdpa_math_context():
        return nullcontext()

import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # , GINConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch_scatter
import os
from torch.utils.checkpoint import checkpoint

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

epsilon = 1e-4


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        nhead: int,
        dim_feedforward: int,
        nlayers: int,
        dropout: float = 0.0,
        activation="gelu",
    ):
        super().__init__()

        encoder_layers = TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.in_dim = in_dim

    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: [batch_size, seq_len, in_dim]
            src_key_padding_mask: [batch_size, seq_len], where True marks
                padded tokens to ignore.

        Returns:
            output: [batch_size, seq_len, in_dim]
        """

        output = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )
        return output


class TunnelEncoder(nn.Module):
    def __init__(self, input_dim: int, props):
        super().__init__()
        self.input_dim = input_dim
        self.num_transformer_layers = props.num_transformer_layers
        self.dropout = props.dropout
        self.num_mlp1_hidden_layers = props.num_mlp1_hidden_layers
        self.use_temporal_cls = getattr(props, "use_temporal_cls", True)
        self.use_temporal_residual = getattr(props, "use_temporal_residual", True)
        self.recency_alpha = getattr(props, "temporal_recency_alpha", 0.1)

        self.cls_token = nn.Parameter(torch.Tensor(1, input_dim))
        nn.init.kaiming_normal_(self.cls_token, nonlinearity="relu")

        self.temporal_cls = nn.Parameter(torch.Tensor(1, 1, input_dim))
        nn.init.kaiming_normal_(self.temporal_cls, nonlinearity="relu")

        if props.num_heads == 0:
            num_heads = input_dim // 4
        else:
            num_heads = props.num_heads

        self.set_transformer = TransformerModel(
            in_dim=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim,
            nlayers=self.num_transformer_layers,
            dropout=self.dropout,
            activation="gelu",
        )

        self.temporal_positional_encoding = SinusoidalPositionalEncoding(input_dim)

        temporal_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=4,
            dim_feedforward=input_dim * 2,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_transformer = TransformerEncoder(temporal_layer, num_layers=1)

        self.temporal_input_projection = nn.Linear(input_dim + 1, input_dim)
        self.temporal_dropout = nn.Dropout(p=props.dropout)

        self.mlp_1_dim = input_dim + 1
        self.mlp1 = nn.ModuleList()
        self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))

        for _ in range(self.num_mlp1_hidden_layers):
            self.mlp1.append(nn.Linear(self.mlp_1_dim, self.mlp_1_dim))

        self.mlp1.append(nn.Linear(self.mlp_1_dim, 1))

    def forward_pass_mlp(self, inputs, mlp: nn.ModuleList, num_hidden_layers):
        for index, layer in enumerate(mlp):
            if index == 0:
                gammas_1 = F.leaky_relu(layer(inputs), 0.02)
            elif index == (num_hidden_layers + 1):
                gammas_1 = layer(gammas_1)
            else:
                gammas_1 = F.leaky_relu(layer(gammas_1), 0.02)

        return gammas_1

    def _normalize_tm_pred(self, tm_pred: Tensor):
        """
        Converts tm_pred into [B, T, P, 1].

        Accepts:
            [P, 1]
            [B, P, 1]
            [B, T, P, 1]
        """

        if tm_pred.dim() == 2:
            return tm_pred.unsqueeze(0).unsqueeze(1)

        if tm_pred.dim() == 3:
            return tm_pred.unsqueeze(1)

        if tm_pred.dim() == 4:
            return tm_pred

        raise ValueError(f"Unsupported tm_pred shape: {tuple(tm_pred.shape)}")

    def generate_causal_mask(self, K: int, device):
        return torch.triu(
            torch.full((K, K), float("-inf"), device=device),
            diagonal=1,
        )

    def _build_recency_weights(self, num_timesteps: int, device, dtype):
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

        positions = torch.arange(num_timesteps, device=device, dtype=dtype)

        if num_timesteps == 1:
            normalized_positions = torch.zeros_like(positions)
        else:
            normalized_positions = positions / max(num_timesteps - 1, 1)

        return 1.0 + self.recency_alpha * normalized_positions.pow(2)

    def forward(
        self,
        path_edge_inputs: Tensor,
        tm_pred: Tensor,
        path_padding_mask: Tensor = None,
    ):
        """
        Args:
            path_edge_inputs:
                [B, P, L, D] or [B, P, T, L, D]

            tm_pred:
                [P, 1] or [B, P, 1] or [B, T, P, 1]

            path_padding_mask:
                [B, P, L] or [B, P, T, L]

        Returns:
            gammas:
                [B, P, 1]

            final_path_edge_embeddings:
                [B, P, L, D]

            final_path_embeddings:
                [B, P, D]
        """

        if path_edge_inputs.dim() == 4:
            path_edge_inputs = path_edge_inputs.unsqueeze(2)

            if path_padding_mask is not None:
                path_padding_mask = path_padding_mask.unsqueeze(2)

        elif path_edge_inputs.dim() != 5:
            raise ValueError(
                f"Unsupported path edge input shape: {tuple(path_edge_inputs.shape)}"
            )

        (
            batch_size,
            total_number_of_paths,
            num_timesteps,
            max_path_length,
            _,
        ) = path_edge_inputs.shape

        flat_inputs = path_edge_inputs.contiguous().reshape(
            batch_size * total_number_of_paths * num_timesteps,
            max_path_length,
            self.input_dim,
        )

        cls_token = self.cls_token.view(1, 1, self.input_dim).expand(
            flat_inputs.shape[0],
            -1,
            -1,
        )

        flat_inputs = torch.cat((cls_token, flat_inputs), dim=1)

        src_key_padding_mask = None

        if path_padding_mask is not None:
            flat_mask = path_padding_mask.contiguous().reshape(
                batch_size * total_number_of_paths * num_timesteps,
                max_path_length,
            )

            cls_mask = torch.zeros(
                flat_mask.shape[0],
                1,
                device=flat_mask.device,
                dtype=torch.bool,
            )

            src_key_padding_mask = torch.cat((cls_mask, flat_mask), dim=1)

        with sdpa_math_context():
            flat_outputs = self.set_transformer(
                flat_inputs,
                src_key_padding_mask=src_key_padding_mask,
            )
        flat_outputs = torch.nan_to_num(
            flat_outputs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        flat_outputs = flat_outputs.clamp(min=-1e6, max=1e6)

        set_outputs = flat_outputs.contiguous().reshape(
            batch_size,
            total_number_of_paths,
            num_timesteps,
            max_path_length + 1,
            self.input_dim,
        )

        timestep_embeddings = set_outputs[:, :, :, 0, :]
        last_embedding = timestep_embeddings[:, :, -1, :]

        # tm_pred: [B, T, P, 1] -> [B, P, T, 1]
        tm_pred = self._normalize_tm_pred(tm_pred)
        tm_pred = tm_pred.permute(0, 2, 1, 3).contiguous()

        assert tm_pred.shape[:3] == (
            batch_size,
            total_number_of_paths,
            num_timesteps,
        ), (
            f"tm_pred shape mismatch. Expected first dims "
            f"{(batch_size, total_number_of_paths, num_timesteps)}, "
            f"got {tuple(tm_pred.shape)}"
        )

        tm_pred = tm_pred / tm_pred.max(dim=2, keepdim=True).values.clamp_min(1e-6)

        temporal_inputs = torch.cat((timestep_embeddings, tm_pred), dim=-1)
        temporal_inputs = self.temporal_input_projection(temporal_inputs)

        assert temporal_inputs.shape[-1] == self.input_dim

        recency_weights = self._build_recency_weights(
            num_timesteps,
            temporal_inputs.device,
            temporal_inputs.dtype,
        )

        temporal_inputs = temporal_inputs * recency_weights.view(
            1,
            1,
            num_timesteps,
            1,
        )

        temporal_inputs = self.temporal_dropout(temporal_inputs)

        temporal_inputs = torch.nan_to_num(
            temporal_inputs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        temporal_inputs = temporal_inputs.clamp(min=-1e6, max=1e6)

        temporal_inputs = temporal_inputs.contiguous().reshape(
            batch_size * total_number_of_paths,
            num_timesteps,
            self.input_dim,
        )

        if self.use_temporal_cls:
            temporal_cls = self.temporal_cls.expand(
                temporal_inputs.shape[0],
                -1,
                -1,
            )
            temporal_inputs = torch.cat((temporal_cls, temporal_inputs), dim=1)

        temporal_inputs = self.temporal_positional_encoding(temporal_inputs)

        temporal_inputs = torch.nan_to_num(
            temporal_inputs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        temporal_inputs = temporal_inputs.clamp(min=-1e6, max=1e6)

        K = num_timesteps
        device = temporal_inputs.device

        base_mask = self.generate_causal_mask(K, device)

        if self.use_temporal_cls:
            full_mask = torch.zeros(K + 1, K + 1, device=device)
            full_mask[0, :] = 0
            full_mask[1:, 0] = 0
            full_mask[1:, 1:] = base_mask
            temporal_mask = full_mask
        else:
            temporal_mask = base_mask

        temporal_outputs = self.temporal_transformer(
            temporal_inputs,
            mask=temporal_mask,
        )

        temporal_outputs = torch.nan_to_num(
            temporal_outputs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        temporal_outputs = temporal_outputs.clamp(min=-1e6, max=1e6)

        temporal_outputs = temporal_outputs.contiguous().reshape(
            batch_size,
            total_number_of_paths,
            temporal_outputs.shape[1],
            self.input_dim,
        )

        if self.use_temporal_cls:
            final_path_embeddings = temporal_outputs[:, :, 0, :]
        else:
            final_path_embeddings = temporal_outputs[:, :, -1, :]

        if self.use_temporal_residual:
            final_path_embeddings = final_path_embeddings + last_embedding

        # Use final timestep path-edge embeddings for RAU bottleneck refinement.
        final_path_edge_embeddings = set_outputs[:, :, -1, 1:, :]

        final_tm_pred = tm_pred[:, :, -1, :]


        mlp_inputs = torch.cat((final_path_embeddings, final_tm_pred), dim=-1)

        mlp_inputs = torch.nan_to_num(
            mlp_inputs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        mlp_inputs = mlp_inputs.clamp(min=-1e6, max=1e6)

        gammas = self.forward_pass_mlp(
            mlp_inputs,
            self.mlp1,
            self.num_mlp1_hidden_layers,
        )

        gammas = torch.nan_to_num(
            gammas,
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )
        gammas = gammas.clamp(min=-50.0, max=50.0)

        return gammas, final_path_edge_embeddings, final_path_embeddings


class GNN(nn.Module):
    def __init__(self, num_features, num_gnn_layers):
        super(GNN, self).__init__()
        self.num_features = num_features

        self.gnns = nn.ModuleList()

        for i in range(num_gnn_layers):
            if i == 0:
                self.gnns.append(GCNConv(num_features, num_features + 1))
            elif i == 1:
                self.gnns.append(GCNConv(num_features + 1, num_features + 2))
            else:
                self.gnns.append(GCNConv(num_features + 2, num_features + 2))

        self.output_dim = num_gnn_layers * (self.num_features + 2) - 1

    def forward(self, node_features, edge_index, capacities):
        """
        Args:
            node_features:
                [B, N, F]

            edge_index:
                [2, E]

            capacities:
                [B, E]

        Returns:
            edge_embeddings:
                [B, E, output_dim]
        """

        batch_size = node_features.shape[0]
        ne_list = []

        for i in range(batch_size):
            nf = node_features[i]
            caps = capacities[i]

            sample_ne_list = []

            for j, gnn in enumerate(self.gnns):
                if j == 0:
                    ne = gnn(nf, edge_index=edge_index, edge_weight=caps)
                else:
                    ne = gnn(ne, edge_index=edge_index, edge_weight=caps)

                ne = F.leaky_relu(ne, 0.02)
                sample_ne_list.append(ne)

            if len(self.gnns) > 1:
                node_embeddings = torch.cat(sample_ne_list, dim=-1)
                ne_list.append(node_embeddings)
            else:
                ne_list.append(sample_ne_list[0])

        node_embeddings = torch.stack(ne_list).contiguous()

        edge_index_expanded = edge_index.t().expand(batch_size, -1, -1)

        batch_size, num_nodes, feature_size = node_embeddings.shape
        _, num_edges, _ = edge_index_expanded.shape

        batch_index = torch.arange(
            batch_size,
            device=node_embeddings.device,
        ).view(-1, 1, 1)

        batch_index = batch_index.expand(-1, num_edges, 2)

        edge_embeddings = node_embeddings[batch_index, edge_index_expanded]

        capacities = capacities.unsqueeze(-1)

        edge_embeddings = edge_embeddings.sum(dim=-2)
        edge_embeddings = torch.cat((edge_embeddings, capacities), dim=-1)

        return edge_embeddings


class HARP(nn.Module):
    def __init__(self, props):
        super(HARP, self).__init__()

        self.num_gnn_layers = props.num_gnn_layers
        self.num_transformer_layers = props.num_transformer_layers
        self.dropout = props.dropout
        self.num_mlp1_hidden_layers = props.num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = props.num_mlp2_hidden_layers
        self.device = props.device

        self.gnn = GNN(2, self.num_gnn_layers)

        self.input_dim = self.gnn.output_dim + 1
        self.tunnel_encoder = TunnelEncoder(self.input_dim, props)

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
        edge_ids_dict_tensor=None,
        original_pos_edge_ids_dict_tensor=None,
    ):
        """
        Supports two modes.

        Static / old mode:
            edge_index: [2, E]
            capacities: [B, E] or [B, T, E]
            padded_edge_ids_per_path: [P, L]
            paths_to_edges: sparse [P, E]

        Dynamic Option A mode:
            edge_index: list length T, each [2, E_t]
            capacities: list length T, each [B, E_t] or [E_t]
            padded_edge_ids_per_path: list length T, each [P, L_t]
            paths_to_edges: list length T, each sparse [P, E_t]

        Option A assumption:
            Source-destination pairs and K path slots stay fixed.
            Actual edge IDs and path edge sequences can change at each timestep.
        """

        num_for_loops = props.num_for_loops
        num_paths_per_pair = props.num_paths_per_pair

        dynamic_topology = self.is_dynamic_topology_input(
            edge_index=edge_index,
            capacities=capacities,
            padded_edge_ids_per_path=padded_edge_ids_per_path,
            paths_to_edges=paths_to_edges,
        )

        tm_history = self.normalize_tm_history(tm)
        tm_pred_history = self.normalize_tm_history(tm_pred)

        tm_final = tm_history[:, -1]
        tm_pred_final = tm_pred_history[:, -1]

        batch_size = tm_final.shape[0]

        if dynamic_topology:
            path_edge_inputs, path_padding_mask = self.compute_dynamic_path_edge_inputs(
                props=props,
                node_features=node_features,
                edge_indices_seq=edge_index,
                capacities_seq=capacities,
                padded_paths_seq=padded_edge_ids_per_path,
            )

            final_paths_to_edges = paths_to_edges[-1].coalesce()
            final_padded_paths = padded_edge_ids_per_path[-1]
            final_capacities = capacities[-1]
            final_edge_index = edge_index[-1]

            if final_capacities.dim() == 1:
                final_capacities = final_capacities.unsqueeze(0).expand(batch_size, -1)

        else:
            edge_embeddings_with_caps = self.compute_edge_embeddings(
                props=props,
                node_features=node_features,
                edge_index=edge_index,
                capacities=capacities,
            )

            path_edge_inputs, path_padding_mask = self.gather_path_edge_inputs(
                edge_embeddings_with_caps=edge_embeddings_with_caps,
                padded_edge_ids_per_path=padded_edge_ids_per_path,
            )

            final_paths_to_edges = paths_to_edges.coalesce()
            final_padded_paths = padded_edge_ids_per_path
            final_capacities = self.normalize_capacities(
                capacities=capacities,
                batch_size=batch_size,
                props=props,
            )
            final_edge_index = edge_index

        total_number_of_paths = final_paths_to_edges.shape[0]

        gammas, path_edge_embeddings, _ = self.tunnel_encoder(
            path_edge_inputs,
            tm_pred_history,
            path_padding_mask,
        )

        indices = final_paths_to_edges.indices()
        values = final_paths_to_edges.values()

        row_indices = indices[0]
        col_indices = indices[1]

        pte_info = [
            final_paths_to_edges,
            row_indices,
            col_indices,
            values,
        ]

        for i in range(num_for_loops):
            if i > 0:
                gammas = new_gammas

            edges_util = self.compute_edge_utils(
                gammas=gammas,
                paths_to_edges=final_paths_to_edges,
                tm=tm_pred_final,
                capacities=final_capacities,
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
                padded_edge_ids_per_path=final_padded_paths,
                path_edge_embeddings=path_edge_embeddings,
                batch_size=batch_size,
                total_number_of_paths=total_number_of_paths,
                pte_info=pte_info,
            )

            dnn_2_inputs = torch.cat(
            (
                bottleneck_path_edge_embeddings,
                max_utilization_per_path,
                mlu,
                tm_pred_final,
            ),
            dim=-1,
            )

            dnn_2_inputs = torch.nan_to_num(
                dnn_2_inputs,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
            dnn_2_inputs = dnn_2_inputs.clamp(min=-1e6, max=1e6)

            delta_gammas = self.forward_pass_mlp(
                dnn_2_inputs,
                self.mlp2,
                self.num_mlp2_hidden_layers,
            )

            delta_gammas = torch.nan_to_num(
                delta_gammas,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )
            delta_gammas = delta_gammas.clamp(min=-50.0, max=50.0)

            gammas = gammas.reshape(batch_size, -1, 1)

            gammas = torch.nan_to_num(
                gammas,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )
            gammas = gammas.clamp(min=-50.0, max=50.0)

            new_gammas = delta_gammas + gammas

            new_gammas = torch.nan_to_num(
                new_gammas,
                nan=0.0,
                posinf=50.0,
                neginf=-50.0,
            )
            new_gammas = new_gammas.clamp(min=-50.0, max=50.0)

        if num_for_loops == 0:
            new_gammas = gammas

        edges_util = self.compute_edge_utils(
            gammas=new_gammas,
            paths_to_edges=final_paths_to_edges,
            tm=tm_final,
            capacities=final_capacities,
            props=props,
            batch_size=batch_size,
            num_paths_per_pair=num_paths_per_pair,
            add_epsilon=False,
        )

        if getattr(props, "return_details", False):
            split_ratios = self.compute_split_ratios(
                new_gammas,
                batch_size,
                num_paths_per_pair,
            )

            data_on_tunnels = split_ratios * tm_final.squeeze(-1)
            data_on_links = torch.sparse.mm(
                final_paths_to_edges.to(dtype=torch.float32).t(),
                data_on_tunnels.to(dtype=torch.float32).t(),
            ).t()

            if props.dtype == torch.bfloat16:
                data_on_links = data_on_links.to(dtype=torch.bfloat16)

            return {
                "edges_util": edges_util,
                "gammas": new_gammas,
                "split_ratios": split_ratios,
                "data_on_links": data_on_links,
                "paths_to_edges": final_paths_to_edges,
                "capacities": final_capacities,
                "edge_index": final_edge_index,
                "tm": tm_final,
            }

        return edges_util

    def is_dynamic_topology_input(
        self,
        edge_index,
        capacities,
        padded_edge_ids_per_path,
        paths_to_edges,
    ):
        return (
            isinstance(edge_index, (list, tuple))
            or isinstance(capacities, (list, tuple))
            or isinstance(padded_edge_ids_per_path, (list, tuple))
            or isinstance(paths_to_edges, (list, tuple))
        )

    def normalize_tm_history(self, tm: Tensor):
        """
        Converts traffic input into [B, T, P, 1].

        Accepts:
            [P, 1]       -> [1, 1, P, 1]
            [B, P, 1]    -> [B, 1, P, 1]
            [B, T, P, 1] -> unchanged
        """

        if tm.dim() == 2:
            return tm.unsqueeze(0).unsqueeze(1)

        if tm.dim() == 3:
            return tm.unsqueeze(1)

        if tm.dim() == 4:
            return tm

        raise ValueError(f"Unsupported tm shape: {tuple(tm.shape)}")

    def normalize_tm(self, tm: Tensor):
        """
        Backward-compatible helper.
        Returns only final timestep.
        """

        return self.normalize_tm_history(tm)[:, -1]

    def normalize_capacities(self, capacities: Tensor, batch_size: int, props):
        """
        Static/fixed-edge capacity normalization.
        Dynamic list-based capacities are handled directly in forward().
        """

        if isinstance(capacities, (list, tuple)):
            capacities = capacities[-1]

        if capacities.dim() == 3:
            capacities = capacities[:, -1]

        if capacities.dim() == 1:
            capacities = capacities.unsqueeze(0)

        if (
            capacities.dim() == 2
            and not props.dynamic
            and batch_size > 1
            and capacities.shape[0] == 1
        ):
            capacities = capacities.expand(batch_size, -1)

        return capacities

    def compute_edge_embeddings(self, props, node_features, edge_index, capacities):
        """
        Static/fixed-edge mode.

        If node_features is [B, T, N, F], this still assumes the same edge_index
        for every timestep.

        True changing edge_index over time is handled by compute_dynamic_path_edge_inputs().
        """

        if node_features.dim() == 4:
            num_timesteps = node_features.shape[1]
            edge_embeddings_over_time = []

            for timestep in range(num_timesteps):
                nf = node_features[:, timestep, :, :]
                caps = capacities[:, timestep, :]

                if props.checkpoint:
                    edge_embeddings_t = checkpoint(
                        self.gnn,
                        nf,
                        edge_index,
                        caps,
                        use_reentrant=False,
                    )
                else:
                    edge_embeddings_t = self.gnn(nf, edge_index, caps)

                edge_embeddings_over_time.append(edge_embeddings_t)

            return torch.stack(edge_embeddings_over_time, dim=1)

        if props.checkpoint:
            return checkpoint(
                self.gnn,
                node_features,
                edge_index,
                capacities,
                use_reentrant=False,
            )

        return self.gnn(node_features, edge_index, capacities)

    def compute_dynamic_path_edge_inputs(
        self,
        props,
        node_features,
        edge_indices_seq,
        capacities_seq,
        padded_paths_seq,
    ):
        """
        Builds path-edge inputs for true changing-edge Option A.

        Inputs:
            node_features:
                [B, T, N, F]

            edge_indices_seq:
                list length T, each [2, E_t]

            capacities_seq:
                list length T, each [B, E_t] or [E_t]

            padded_paths_seq:
                list length T, each [P, L_t]

        Outputs:
            path_edge_inputs:
                [B, P, T, Lmax, D]

            path_padding_mask:
                [B, P, T, Lmax]

        Assumption:
            P is fixed across time.
            Actual edge IDs/path contents may change across time.
        """

        if node_features.dim() != 4:
            raise ValueError(
                "Dynamic topology mode expects node_features shape [B, T, N, F]. "
                f"Got {tuple(node_features.shape)}"
            )

        batch_size, num_timesteps, _, _ = node_features.shape

        if not isinstance(edge_indices_seq, (list, tuple)):
            raise ValueError(
                "Dynamic topology mode expects edge_index to be a list/tuple of length T."
            )

        if not isinstance(capacities_seq, (list, tuple)):
            raise ValueError(
                "Dynamic topology mode expects capacities to be a list/tuple of length T."
            )

        if not isinstance(padded_paths_seq, (list, tuple)):
            raise ValueError(
                "Dynamic topology mode expects padded_edge_ids_per_path "
                "to be a list/tuple of length T."
            )

        if not (
            len(edge_indices_seq)
            == len(capacities_seq)
            == len(padded_paths_seq)
            == num_timesteps
        ):
            raise ValueError(
                "Dynamic topology sequence lengths must match node_features.shape[1]. "
                f"Got T={num_timesteps}, "
                f"edge_indices={len(edge_indices_seq)}, "
                f"capacities={len(capacities_seq)}, "
                f"padded_paths={len(padded_paths_seq)}"
            )

        max_path_length = max(p.shape[1] for p in padded_paths_seq)

        path_inputs_over_time = []
        path_masks_over_time = []

        for timestep in range(num_timesteps):
            nf_t = node_features[:, timestep, :, :]
            edge_index_t = edge_indices_seq[timestep]
            caps_t = capacities_seq[timestep]
            padded_paths_t = padded_paths_seq[timestep]

            if caps_t.dim() == 1:
                caps_t = caps_t.unsqueeze(0).expand(batch_size, -1)

            if props.checkpoint:
                edge_embeddings_t = checkpoint(
                    self.gnn,
                    nf_t,
                    edge_index_t,
                    caps_t,
                    use_reentrant=False,
                )
            else:
                edge_embeddings_t = self.gnn(
                    nf_t,
                    edge_index_t,
                    caps_t,
                )

            path_edge_inputs_t, path_mask_t = self.gather_path_edge_inputs(
                edge_embeddings_t,
                padded_paths_t,
            )

            current_length = path_edge_inputs_t.shape[2]

            if current_length < max_path_length:
                pad_len = max_path_length - current_length

                path_edge_inputs_t = F.pad(
                    path_edge_inputs_t,
                    pad=(0, 0, 0, pad_len),
                    value=0.0,
                )

                path_mask_t = F.pad(
                    path_mask_t,
                    pad=(0, pad_len),
                    value=True,
                )

            path_inputs_over_time.append(path_edge_inputs_t)
            path_masks_over_time.append(path_mask_t)

        path_edge_inputs = torch.stack(path_inputs_over_time, dim=2).contiguous()
        path_padding_mask = torch.stack(path_masks_over_time, dim=2).contiguous()

        return path_edge_inputs, path_padding_mask

    def gather_path_edge_inputs(
        self,
        edge_embeddings_with_caps: Tensor,
        padded_edge_ids_per_path: Tensor,
    ):
        """
        Static:
            edge_embeddings_with_caps: [B, E, D]
            padded_edge_ids_per_path: [P, L]

        Temporal static:
            edge_embeddings_with_caps: [B, T, E, D]
            padded_edge_ids_per_path: [P, L]

        Returns:
            Static:
                [B, P, L, D], [B, P, L]

            Temporal static:
                [B, P, T, L, D], [B, P, T, L]
        """

        safe_edge_ids = padded_edge_ids_per_path.clamp(min=0)
        path_mask = padded_edge_ids_per_path.eq(-1)

        if edge_embeddings_with_caps.dim() == 3:
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
                2,
                gather_indices,
            )

            path_edge_inputs = path_edge_inputs.masked_fill(
                path_mask.view(1, total_number_of_paths, max_path_length, 1),
                0.0,
            )

            return (
                path_edge_inputs,
                path_mask.view(1, total_number_of_paths, max_path_length).expand(
                    batch_size,
                    -1,
                    -1,
                ),
            )

        batch_size, num_timesteps, _, feat_dim = edge_embeddings_with_caps.shape
        total_number_of_paths, max_path_length = safe_edge_ids.shape

        gather_indices = safe_edge_ids.view(
            1,
            1,
            total_number_of_paths,
            max_path_length,
            1,
        ).expand(
            batch_size,
            num_timesteps,
            -1,
            -1,
            feat_dim,
        )

        expanded_edges = edge_embeddings_with_caps.unsqueeze(2).expand(
            -1,
            -1,
            total_number_of_paths,
            -1,
            -1,
        )

        path_edge_inputs = torch.gather(
            expanded_edges,
            3,
            gather_indices,
        )

        path_edge_inputs = path_edge_inputs.masked_fill(
            path_mask.view(1, 1, total_number_of_paths, max_path_length, 1),
            0.0,
        )

        path_padding_mask = path_mask.view(
            1,
            1,
            total_number_of_paths,
            max_path_length,
        ).expand(
            batch_size,
            num_timesteps,
            -1,
            -1,
        )

        return (
            path_edge_inputs.permute(0, 2, 1, 3, 4).contiguous(),
            path_padding_mask.permute(0, 2, 1, 3).contiguous(),
        )

    def compute_mlu(
        self,
        edges_util,
        batch_size,
        total_number_of_paths,
        subtract_epsilon=True,
    ):
        """
        Compute per-batch MLU and broadcast over paths.

        Args:
            edges_util: [B, E]

        Returns:
            mlu: [B, P, 1]
        """

        mlu, _ = torch.max(edges_util, dim=-1)

        if subtract_epsilon:
            mlu = mlu - epsilon

        mlu = mlu.view(batch_size, 1, 1).expand(
            -1,
            total_number_of_paths,
            -1,
        )

        return mlu

    def compute_bottleneck_link_mlu_per_path(
        self,
        edge_utils,
        padded_edge_ids_per_path,
        path_edge_embeddings,
        batch_size,
        total_number_of_paths,
        pte_info,
    ):
        """
        Safer bottleneck-link lookup.

        Original HARP used torch_scatter over paths_to_edges, then tried to map
        the scatter argmax back into padded_edge_ids_per_path. That is brittle
        for dynamic topology because edge IDs/path padding can change per sample.

        This version directly uses padded_edge_ids_per_path:

            edge_utils: [B, E_final]
            padded_edge_ids_per_path: [P, L_final], padded with -1
            path_edge_embeddings: [B, P, Lmax, D]

        Returns:
            bottleneck_path_edge_embeddings: [B, P, D]
            max_utilization_per_path: [B, P, 1]
        """

        if padded_edge_ids_per_path.dim() != 2:
            raise ValueError(
                "padded_edge_ids_per_path must have shape [P, L]. "
                f"Got {tuple(padded_edge_ids_per_path.shape)}"
            )

        if edge_utils.dim() != 2:
            raise ValueError(
                "edge_utils must have shape [B, E]. "
                f"Got {tuple(edge_utils.shape)}"
            )

        if path_edge_embeddings.dim() != 4:
            raise ValueError(
                "path_edge_embeddings must have shape [B, P, L, D]. "
                f"Got {tuple(path_edge_embeddings.shape)}"
            )

        device = edge_utils.device

        padded_edge_ids_per_path = padded_edge_ids_per_path.to(device=device)

        P, L_final = padded_edge_ids_per_path.shape
        B, E_final = edge_utils.shape
        _, P_embed, L_embed, D = path_edge_embeddings.shape

        if P != total_number_of_paths:
            raise ValueError(
                f"Path count mismatch: padded paths has P={P}, "
                f"total_number_of_paths={total_number_of_paths}"
            )

        if P_embed != P:
            raise ValueError(
                f"Path embedding count mismatch: path_edge_embeddings has P={P_embed}, "
                f"padded paths has P={P}"
            )

        if L_final > L_embed:
            raise ValueError(
                f"Final padded path length L_final={L_final} is larger than "
                f"path_edge_embeddings length L_embed={L_embed}"
            )

        valid_mask = padded_edge_ids_per_path.ge(0)

        safe_edge_ids = padded_edge_ids_per_path.clamp(min=0)

        if safe_edge_ids.numel() > 0:
            max_edge_id = int(safe_edge_ids.max().detach().cpu())
            if max_edge_id >= E_final:
                raise ValueError(
                    f"padded_edge_ids_per_path contains edge id {max_edge_id}, "
                    f"but edge_utils only has {E_final} edges."
                )

        # Gather edge utilization for every edge position in every path.
        # edge_utils: [B, E]
        # safe_edge_ids: [P, L]
        # gathered_utils: [B, P, L]
        gather_ids = safe_edge_ids.view(1, P, L_final).expand(B, -1, -1)
        expanded_edge_utils = edge_utils.view(B, 1, E_final).expand(-1, P, -1)

        gathered_utils = torch.gather(
            expanded_edge_utils,
            dim=2,
            index=gather_ids,
        )

        # Ignore padded positions.
        gathered_utils = gathered_utils.masked_fill(
            ~valid_mask.view(1, P, L_final),
            float("-inf"),
        )

        max_utilization_per_path, bottleneck_positions = gathered_utils.max(dim=2)

        # Safety: if a path somehow has no valid edges, avoid indexing garbage.
        no_valid_path = ~valid_mask.any(dim=1)

        if no_valid_path.any():
            bottleneck_positions = bottleneck_positions.masked_fill(
                no_valid_path.view(1, P),
                0,
            )
            max_utilization_per_path = max_utilization_per_path.masked_fill(
                no_valid_path.view(1, P),
                0.0,
            )

        # Gather the embedding at the bottleneck position.
        # path_edge_embeddings: [B, P, L_embed, D]
        batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, P)
        path_indices = torch.arange(P, device=device).view(1, P).expand(B, P)

        bottleneck_path_edge_embeddings = path_edge_embeddings[
            batch_indices,
            path_indices,
            bottleneck_positions,
        ]

        max_utilization_per_path = max_utilization_per_path - epsilon

        return bottleneck_path_edge_embeddings, max_utilization_per_path.unsqueeze(-1)

    def forward_pass_mlp(self, inputs, mlp: nn.ModuleList, num_hidden_layers):
        for index, layer in enumerate(mlp):
            if index == 0:
                gammas_1 = layer(inputs)
                gammas_1 = F.leaky_relu(gammas_1, 0.02)

            elif index == (num_hidden_layers + 1):
                gammas_1 = layer(gammas_1)

            else:
                gammas_1 = layer(gammas_1)
                gammas_1 = F.leaky_relu(gammas_1, 0.02)

        return gammas_1

    def compute_split_ratios(self, gammas, batch_size, num_paths_per_pair):
        gammas = gammas.reshape(batch_size, -1, num_paths_per_pair)

        gammas = torch.nan_to_num(
            gammas,
            nan=0.0,
            posinf=50.0,
            neginf=-50.0,
        )
        gammas = gammas.clamp(min=-50.0, max=50.0)

        split_ratios = torch.softmax(gammas, dim=-1)

        split_ratios = torch.nan_to_num(
            split_ratios,
            nan=1.0 / num_paths_per_pair,
            posinf=1.0,
            neginf=0.0,
        )

        split_ratios = split_ratios.reshape(batch_size, -1)

        return split_ratios

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
        """
        Convert raw path scores to edge utilizations.

        Args:
            gammas:
                [B, P, 1]

            paths_to_edges:
                sparse [P, E]

            tm:
                [B, P, 1]

            capacities:
                [B, E]

        Returns:
            edges_util:
                [B, E]
        """

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
            edges_util = data_on_links / capacities + epsilon
        else:
            edges_util = data_on_links / capacities

        edges_util = torch.nan_to_num(
            edges_util,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )

        edges_util = edges_util.clamp(min=0.0, max=1e6)

        return edges_util
