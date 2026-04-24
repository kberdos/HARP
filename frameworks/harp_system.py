from torch.nn.attention import sdpa_kernel, SDPBackend
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv#, GINConv 
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
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)

# Set Transformer
class TransformerModel(nn.Module):
    def __init__(self, in_dim: int, nhead: int, dim_feedforward: int,
                 nlayers: int, dropout: float = 0.0, activation="gelu"):
        super().__init__()
        
        encoder_layers = TransformerEncoderLayer(d_model=in_dim, nhead=nhead,
                            dim_feedforward=dim_feedforward, dropout=dropout,
                        batch_first=True, activation=activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.in_dim = in_dim
        
    def forward(self, src: Tensor, src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_dim), 
                                representing the source sequences.
            src_key_padding_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len) 
                                                        indicating which positions should be ignored 
                                                        in the source sequence. Default is None.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input, after being passed through 
                        the Transformer encoder.
        """
        
        if src_key_padding_mask is not None:
            src_key_padding_mask = (~src_key_padding_mask)
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
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
        nn.init.kaiming_normal_(self.cls_token, nonlinearity='relu')
        self.temporal_cls = nn.Parameter(torch.Tensor(1, 1, input_dim))
        nn.init.kaiming_normal_(self.temporal_cls, nonlinearity='relu')

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
            diagonal=1
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

    def forward(self, path_edge_inputs: Tensor, tm_pred: Tensor, path_padding_mask: Tensor = None):
        """
        Args:
            path_edge_inputs: [B, P, L, D] or [B, P, T, L, D]
            tm_pred: [B, P, 1] or [B, T, P, 1]

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
            - gammas: [B, P, 1]
            - final_path_edge_embeddings: [B, P, L, D]
            - final_path_embeddings: [B, P, D]
        """
        if path_edge_inputs.dim() == 4:
            path_edge_inputs = path_edge_inputs.unsqueeze(2)
            if path_padding_mask is not None:
                path_padding_mask = path_padding_mask.unsqueeze(2)
        elif path_edge_inputs.dim() != 5:
            raise ValueError(f"Unsupported path edge input shape: {tuple(path_edge_inputs.shape)}")

        batch_size, total_number_of_paths, num_timesteps, max_path_length, _ = path_edge_inputs.shape

        flat_inputs = path_edge_inputs.contiguous().reshape(
            batch_size * total_number_of_paths * num_timesteps, max_path_length, self.input_dim
        )
        cls_token = self.cls_token.view(1, 1, self.input_dim).expand(flat_inputs.shape[0], -1, -1)
        flat_inputs = torch.cat((cls_token, flat_inputs), dim=1)
        src_key_padding_mask = None
        if path_padding_mask is not None:
            flat_mask = path_padding_mask.contiguous().reshape(
                batch_size * total_number_of_paths * num_timesteps, max_path_length
            )
            cls_mask = torch.zeros(flat_mask.shape[0], 1, device=flat_mask.device, dtype=torch.bool)
            src_key_padding_mask = torch.cat((cls_mask, flat_mask), dim=1)

        with sdpa_kernel([SDPBackend.MATH]):
            flat_outputs = self.set_transformer(flat_inputs, src_key_padding_mask=src_key_padding_mask)

        set_outputs = flat_outputs.contiguous().reshape(
            batch_size, total_number_of_paths, num_timesteps, max_path_length + 1, self.input_dim
        )
        timestep_embeddings = set_outputs[:, :, :, 0, :]
        last_embedding = timestep_embeddings[:, :, -1, :]

        # Inject temporal demand history before temporal aggregation:
        # tm_pred: [B, T, P, 1] -> [B, P, T, 1], then concatenate with per-timestep tunnel embeddings.
        tm_pred = self._normalize_tm_pred(tm_pred)
        tm_pred = tm_pred.permute(0, 2, 1, 3).contiguous()
        assert tm_pred.shape[:3] == (batch_size, total_number_of_paths, num_timesteps)
        tm_pred = tm_pred / (tm_pred.max(dim=2, keepdim=True).values.clamp_min(1e-6))
        temporal_inputs = torch.cat((timestep_embeddings, tm_pred), dim=-1)
        temporal_inputs = self.temporal_input_projection(temporal_inputs)
        assert temporal_inputs.shape[-1] == self.input_dim

        # Lightweight recency bias so newer timesteps receive slightly larger magnitude.
        recency_weights = self._build_recency_weights(
            num_timesteps, temporal_inputs.device, temporal_inputs.dtype
        )
        temporal_inputs = temporal_inputs * recency_weights.view(1, 1, num_timesteps, 1)
        temporal_inputs = self.temporal_dropout(temporal_inputs)
        temporal_inputs = temporal_inputs.contiguous().reshape(
            batch_size * total_number_of_paths, num_timesteps, self.input_dim
        )

        if self.use_temporal_cls:
            temporal_cls = self.temporal_cls.expand(temporal_inputs.shape[0], -1, -1)
            temporal_inputs = torch.cat((temporal_cls, temporal_inputs), dim=1)

        temporal_inputs = self.temporal_positional_encoding(temporal_inputs)

        # Apply causal masking over time so each timestep only attends to current/past history.
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

        temporal_outputs = self.temporal_transformer(temporal_inputs, mask=temporal_mask)
        temporal_outputs = temporal_outputs.contiguous().reshape(
            batch_size, total_number_of_paths, temporal_outputs.shape[1], self.input_dim
        )
        if self.use_temporal_cls:
            final_path_embeddings = temporal_outputs[:, :, 0, :]
        else:
            final_path_embeddings = temporal_outputs[:, :, -1, :]
        if self.use_temporal_residual:
            final_path_embeddings = final_path_embeddings + last_embedding
        final_path_edge_embeddings = set_outputs[:, :, -1, 1:, :]

        final_tm_pred = tm_pred[:, :, -1, :]
        mlp_inputs = torch.cat((final_path_embeddings, final_tm_pred), dim=-1)
        gammas = self.forward_pass_mlp(mlp_inputs, self.mlp1, self.num_mlp1_hidden_layers)

        return gammas, final_path_edge_embeddings, final_path_embeddings

# GNN of HARP
class GNN(nn.Module):
    def __init__(self, num_features, num_gnn_layers):
        super(GNN, self).__init__()
        self.num_features = num_features
                
        self.gnns = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                self.gnns.append(GCNConv(num_features, num_features+1))
            elif i == 1:
                self.gnns.append(GCNConv(num_features+1, num_features+2))
            else:
                self.gnns.append(GCNConv(num_features+2, num_features+2))
        self.output_dim = num_gnn_layers*(self.num_features+2) - 1
        
        
        
    def forward(self, node_features, edge_index, capacities):
        """
        Forward pass of the GNN model.

        Args:
            node_features (torch.Tensor): Node features for each graph in the batch, 
                                        shape (batch_size, num_nodes, num_features).
            edge_index (torch.Tensor): Edge indices defining the graph connectivity, 
                                    shape (2, num_edges).
            capacities (torch.Tensor): Edge capacities for each graph in the batch, 
                                    shape (batch_size, num_edges).

        Returns:
            torch.Tensor: Edge embeddings for each edge in the graph, with capacities included, 
                        shape (batch_size, num_edges, output_dim).

        Process:
            1. Iterate over the batch of node features and capacities.
            2. For each graph in the batch:
                a. Pass the node features through each GNN layer (GCNConv), applying Leaky ReLU activation.
                b. Collect the intermediate node embeddings after each GNN layer.
                c. Concatenate node embeddings from all GNN layers if more than one GNN layer is used.
            3. Stack the node embeddings for all graphs in the batch.
            4. Expand edge_index to match the batch size, so it can be used for batch processing.
            5. Use the expanded edge_index to extract edge embeddings from the node embeddings.
            6. Sum the node embeddings corresponding to each edge and concatenate them with the edge capacities.
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
                # ne = F.silu(ne)
                sample_ne_list.append(ne)
            if len(self.gnns) > 1:
                node_embeddings = torch.cat((sample_ne_list), dim=-1)
                ne_list.append(node_embeddings)
            else:
                ne_list = sample_ne_list
        node_embeddings = torch.stack(ne_list).contiguous()
        edge_index_expanded = edge_index.t().expand(batch_size, -1, -1)
        batch_size, num_nodes, feature_size = node_embeddings.shape
        _, num_edges, _ = edge_index_expanded.shape
        
        # Create a batch index
        batch_index = torch.arange(batch_size, device=node_embeddings.device).view(-1, 1, 1)
        batch_index = batch_index.expand(-1, num_edges, 2)  # Repeat the batch index for each edge
        edge_embeddings = node_embeddings[batch_index, edge_index_expanded]
        capacities = capacities.unsqueeze(-1)
        edge_embeddings = edge_embeddings.sum(dim=-2)
        edge_embeddings = torch.cat((edge_embeddings, capacities), dim=-1)
        
        return edge_embeddings

        
class HARP(nn.Module):
    
    def __init__(self, props):
        
        super(HARP, self).__init__()
        
        # Define the architecture of HARP
        self.num_gnn_layers = props.num_gnn_layers
        self.num_transformer_layers = props.num_transformer_layers
        self.dropout = props.dropout
        self.num_mlp1_hidden_layers = props.num_mlp1_hidden_layers
        self.num_mlp2_hidden_layers = props.num_mlp2_hidden_layers
        self.device = props.device
        
        # Define the GNN
        self.gnn = GNN(2, self.num_gnn_layers)

        self.input_dim = self.gnn.output_dim + 1
        self.tunnel_encoder = TunnelEncoder(self.input_dim, props)
        
        # Define the 2nd MLP (Recurrent Adjustment Unit - RAU)
        self.mlp_2_dim = self.input_dim + 3
        self.mlp2 = nn.ModuleList()
        self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        for i in range(self.num_mlp2_hidden_layers):
            self.mlp2.append(nn.Linear(self.mlp_2_dim, self.mlp_2_dim))
        self.mlp2.append(nn.Linear(self.mlp_2_dim, 1))
        
        
    def forward(self, props, node_features, edge_index, capacities, padded_edge_ids_per_path,
                tm, tm_pred, paths_to_edges, edge_ids_dict_tensor, original_pos_edge_ids_dict_tensor):
        """
            Process:
            1. Pass the node features, edge index, and capacities through the GNN to obtain edge embeddings.
            2. Expand the edge embeddings using the padded edge IDs per path.
            3. Add a CLS token to the edge embeddings and apply masking for attention.
            4. At this point, tunnels are described as a set of edges (edge embeddings)
            5. Pass the tunnels as sets of edges through the Set Transformer.
            6. Concatenate the transformer output for path embeddings (corresponds to the CLS token) with the predicted traffic matrix.
            7. Compute initial split ratios using the first MLP (mlp1).
            8. Perform iterative adjustments of split ratios using the second MLP (RAU) within the 
            specified number of for-loops. MLP2 takes as input (per tunnel):
                i) Demand of the pair that the tunnels is associated with
                ii) Network-wide MLU
                iii) Bottleneck link utilization in the tunnel
                iv) Tunnel embeddings conditioned on the bottleneck link as generated by the Set Transformer 
        """
        
        num_for_loops = props.num_for_loops
        num_paths_per_pair = props.num_paths_per_pair
        edge_embeddings_with_caps = self.compute_edge_embeddings(props, node_features, edge_index, capacities)
        tm = self.normalize_tm(tm)
        tm_pred = self.normalize_tm(tm_pred)
        capacities = self.normalize_capacities(capacities, tm.shape[0], props)
        batch_size = tm.shape[0]
        total_number_of_paths = paths_to_edges.shape[0]

        path_edge_inputs, path_padding_mask = self.gather_path_edge_inputs(edge_embeddings_with_caps, padded_edge_ids_per_path)
        gammas, path_edge_embeddings, _ = self.tunnel_encoder(path_edge_inputs, tm_pred, path_padding_mask)
                
        paths_to_edges = paths_to_edges.coalesce()
        indices = paths_to_edges.indices()
        values = paths_to_edges.values()
        row_indices = indices[0]
        col_indices = indices[1]
        pte_info = [paths_to_edges, row_indices, col_indices, values]
        
        for i in range(num_for_loops):
            if i > 0:
                gammas = new_gammas
            
            if props.checkpoint:
                edges_util = checkpoint(self.compute_edge_utils, gammas, paths_to_edges, tm_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True, use_reentrant=False)
            else:
                edges_util = self.compute_edge_utils(gammas, paths_to_edges, tm_pred, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True)
            
            if props.checkpoint:
                mlu = checkpoint(self.compute_mlu, edges_util, batch_size, total_number_of_paths, subtract_epsilon=True, use_reentrant=False)
            else:
                mlu = self.compute_mlu(edges_util, batch_size, total_number_of_paths, subtract_epsilon=True)

            if props.checkpoint:
                bottleneck_path_edge_embeddings, max_utilization_per_path = checkpoint(self.compute_bottleneck_link_mlu_per_path, edges_util, padded_edge_ids_per_path, path_edge_embeddings, batch_size, total_number_of_paths, pte_info, use_reentrant=False)
            else:
                bottleneck_path_edge_embeddings, max_utilization_per_path = self.compute_bottleneck_link_mlu_per_path(edges_util, padded_edge_ids_per_path, path_edge_embeddings, batch_size, total_number_of_paths, pte_info)

            dnn_2_inputs = torch.cat((bottleneck_path_edge_embeddings, 
                                      max_utilization_per_path,
                                      mlu,
                                      tm_pred), dim=-1).squeeze(0)
            
            if props.checkpoint:
                delta_gammas = checkpoint(self.forward_pass_mlp, dnn_2_inputs, self.mlp2, self.num_mlp2_hidden_layers, use_reentrant=False)
            else:
                delta_gammas = self.forward_pass_mlp(dnn_2_inputs, self.mlp2, self.num_mlp2_hidden_layers)
            
            gammas = gammas.reshape(batch_size, -1, 1)
            new_gammas = delta_gammas + gammas
        
        if num_for_loops == 0:
            new_gammas = gammas
        
        if props.checkpoint:
            edges_util = checkpoint(self.compute_edge_utils, new_gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=False, use_reentrant=False)
        else:
            edges_util = self.compute_edge_utils(new_gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=False)
        
        return edges_util

    def normalize_tm(self, tm: Tensor):
        if tm.dim() == 2:
            tm = tm.unsqueeze(0)
        elif tm.dim() == 4:
            tm = tm[:, -1]
        return tm

    def normalize_capacities(self, capacities: Tensor, batch_size: int, props):
        if capacities.dim() == 3:
            capacities = capacities[:, -1]
        if capacities.dim() == 2 and not props.dynamic and batch_size > 1 and capacities.shape[0] == 1:
            capacities = capacities.expand(batch_size, -1)
        return capacities

    def compute_edge_embeddings(self, props, node_features, edge_index, capacities):
        if node_features.dim() == 4:
            num_timesteps = node_features.shape[1]
            edge_embeddings_over_time = []
            for timestep in range(num_timesteps):
                nf = node_features[:, timestep, :, :]
                caps = capacities[:, timestep, :,]
                if props.checkpoint:
                    edge_embeddings_t = checkpoint(self.gnn, nf, edge_index, caps, use_reentrant=False)
                else:
                    edge_embeddings_t = self.gnn(nf, edge_index, caps)
                edge_embeddings_over_time.append(edge_embeddings_t)
            return torch.stack(edge_embeddings_over_time, dim=1)

        if props.checkpoint:
            return checkpoint(self.gnn, node_features, edge_index, capacities, use_reentrant=False)
        return self.gnn(node_features, edge_index, capacities)

    def gather_path_edge_inputs(self, edge_embeddings_with_caps: Tensor, padded_edge_ids_per_path: Tensor):
        safe_edge_ids = padded_edge_ids_per_path.clamp(min=0)
        path_mask = padded_edge_ids_per_path.eq(-1)

        if edge_embeddings_with_caps.dim() == 3:
            batch_size, _, feat_dim = edge_embeddings_with_caps.shape
            total_number_of_paths, max_path_length = safe_edge_ids.shape
            gather_indices = safe_edge_ids.view(1, total_number_of_paths, max_path_length, 1).expand(
                batch_size, -1, -1, feat_dim
            )
            expanded_edges = edge_embeddings_with_caps.unsqueeze(1).expand(-1, total_number_of_paths, -1, -1)
            path_edge_inputs = torch.gather(expanded_edges, 2, gather_indices)
            path_edge_inputs = path_edge_inputs.masked_fill(path_mask.view(1, total_number_of_paths, max_path_length, 1), 0.0)
            return path_edge_inputs, path_mask.view(1, total_number_of_paths, max_path_length).expand(batch_size, -1, -1)

        batch_size, num_timesteps, _, feat_dim = edge_embeddings_with_caps.shape
        total_number_of_paths, max_path_length = safe_edge_ids.shape
        gather_indices = safe_edge_ids.view(1, 1, total_number_of_paths, max_path_length, 1).expand(
            batch_size, num_timesteps, -1, -1, feat_dim
        )
        expanded_edges = edge_embeddings_with_caps.unsqueeze(2).expand(-1, -1, total_number_of_paths, -1, -1)
        path_edge_inputs = torch.gather(expanded_edges, 3, gather_indices)
        path_edge_inputs = path_edge_inputs.masked_fill(
            path_mask.view(1, 1, total_number_of_paths, max_path_length, 1), 0.0
        )
        path_padding_mask = path_mask.view(1, 1, total_number_of_paths, max_path_length).expand(
            batch_size, num_timesteps, -1, -1
        )
        return path_edge_inputs.permute(0, 2, 1, 3, 4).contiguous(), path_padding_mask.permute(0, 2, 1, 3).contiguous()

    def compute_mlu(self, edges_util, batch_size, total_number_of_paths, subtract_epsilon=True):
            """
            Compute per-batch Maximum Link Utilization (MLU) and broadcast over paths.

            - Reduces per-edge utilizations to a single MLU per batch element via max over edges.
            - Optionally subtracts a small epsilon for numerical stability.
            - Reshapes to [B, 1, 1] and expands to [B, P, 1] to align with path-wise tensors.

            Args:
                edges_util (Tensor): Per-edge utilizations [B, E].
                batch_size (int): Batch size B.
                total_number_of_paths (int): Number of paths P.
                subtract_epsilon (bool): Whether to subtract a small epsilon from the MLU.

            Returns:
                Tensor: Broadcast MLU of shape [B, P, 1].
            """
            
            mlu, mlu_indices = torch.max(edges_util, dim=-1)
            if subtract_epsilon:
                mlu = mlu -  epsilon
            mlu = mlu.view(batch_size, 1, 1).expand(-1, total_number_of_paths, -1)
            
            return mlu

    def compute_bottleneck_link_mlu_per_path(self, edge_utils, padded_edge_ids_per_path, path_edge_embeddings, batch_size, total_number_of_paths, pte_info):
        """
        For each path, locate its bottleneck edge, fetch that edge's embedding, and return the
        bottleneck utilization.

        Steps:
        - Use `compute_bottleneck_util_per_path` logic to find the per-path max edge utilization
          and indices of the edges achieving that max.
        - Match those edge indices against `padded_edge_ids_per_path` to recover positions of the
          bottleneck edge within each path's padded edge sequence.
        - Index into `path_edge_embeddings` to extract the corresponding per-path bottleneck
          edge embeddings.

        Args:
            edge_utils (Tensor): Per-edge utilization [B, E].
            padded_edge_ids_per_path (LongTensor): Padded edge ids per path [B, P, Lmax].
            path_edge_embeddings (Tensor): Per-path per-edge embeddings [B, P, Lmax, D].
            batch_size (int): B.
            total_number_of_paths (int): P.
            pte_info (Tuple): (paths_to_edges, row_indices, col_indices, values) describing sparse
                path→edge mapping.

        Returns:
            Tuple[Tensor, Tensor]:
            - bottleneck_path_edge_embeddings: Embedding of bottleneck edge per path [B, P, D].
            - max_utilization_per_path: Bottleneck utilization per path [B, P, 1].
        """
        
        paths_to_edges, row_indices, col_indices, values = pte_info
        max_utilization_per_path, max_indices = torch_scatter.scatter_max((edge_utils[:, col_indices] * values),
                                                                        row_indices, dim=1, dim_size=paths_to_edges.shape[0])
        max_utilization_per_path = max_utilization_per_path - epsilon
        try:
            max_indices = col_indices[max_indices]
        except:
            print("max_indices.shape:", max_indices.shape)
            print("max_indices.device:", max_indices.device)
            print("max_indices.dtype:", max_indices.dtype)
            print("max_indices contains NaN:", torch.isnan(max_indices).any().item())
            print("max_indices contains Inf:", torch.isinf(max_indices).any().item())
            print(max_indices.max())
            print(col_indices.max())
            print("Out of bound indexing!!")
            exit(1)
        
        max_indices_expanded = max_indices.unsqueeze(2).expand(-1, -1,  padded_edge_ids_per_path.size(1))
        matches = (max_indices_expanded == padded_edge_ids_per_path)
        
        try:
            positions = torch.where(matches)
        except Exception as e:
            print(e)
            print(edge_utils.max())
            print("edge_utils contains NaN:", torch.isnan(edge_utils).any().item())
            print("edge_utils contains Inf:", torch.isinf(edge_utils).any().item())
            print(edge_utils.max())
            print(max_indices_expanded.shape, padded_edge_ids_per_path.shape)
            print(max_indices_expanded.max())
            print(padded_edge_ids_per_path.max())
            print(matches.max())
            print("Out of bound indexing!!")
            exit(1)
        positions = torch.stack(positions, dim=-1)
        positions = positions.view(batch_size, total_number_of_paths, -1)
                            
        dim0_range = positions[:, :, 0].view(batch_size, total_number_of_paths, -1)
        dim1_range = positions[:, :, 1].view(batch_size, total_number_of_paths, -1)
        positions = positions[:, :, -1].view(batch_size, total_number_of_paths, -1)
        
        bottleneck_path_edge_embeddings = (path_edge_embeddings[dim0_range, dim1_range, positions]).squeeze(-2)
                
        return bottleneck_path_edge_embeddings, max_utilization_per_path.unsqueeze(-1)


    def forward_pass_mlp(self, inputs, mlp: nn.ModuleList, num_hidden_layers):
        """
        Apply a stack of Linear layers with LeakyReLU activations to produce raw path scores (gammas).

        - The first layer is applied to `inputs`, followed by LeakyReLU.
        - Each hidden layer (count = `num_hidden_layers`) is applied with LeakyReLU.
        - The final layer (index == num_hidden_layers + 1) is applied without activation.

        Args:
            inputs (Tensor): Input tensor for the MLP.
            mlp (nn.ModuleList): Sequence of Linear layers defining the MLP.
            num_hidden_layers (int): Number of hidden layers (excludes input and output layers).

        Returns:
            Tensor: Output tensor shaped by the last Linear layer's out_features.
        """
        
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
        split_ratios = torch.exp(torch.nn.functional.log_softmax(gammas, dim=-1))
        split_ratios = split_ratios.reshape(batch_size, -1)
        
        return split_ratios
        
    def compute_edge_utils(self, gammas, paths_to_edges, tm, capacities, props, batch_size, num_paths_per_pair, add_epsilon=True):
        """
        Convert raw path scores to traffic allocations and compute per-edge utilizations.

        Behavior:
        - Reshape `gammas` to [B, P, K], apply log_softmax then exp to get
          numerically stable split ratios per path; multiply by tm to get data on tunnels.
        - Aggregate tunnel traffic to links via sparse matmul (paths_to_edges^T · tunnels).
        - Divide by capacities to obtain edge utilizations; optionally add epsilon.

        Args:
            gammas (Tensor): Path scores.
            paths_to_edges (torch.sparse_coo_tensor): Sparse map [P, E] from paths to edges.
            tm (Tensor): Traffic matrix per path group [B, P, 1].
            capacities (Tensor): Link capacities [B, E].
            props: Config namespace (uses props.dtype for dtype control).
            batch_size (int): B.
            num_paths_per_pair (int): K, paths per source-destination pair.
            add_epsilon (bool): Add small epsilon to edge utilizations.

        Returns:
            Tuple[Tensor, Tensor]:
            - edges_util: Edge utilizations [B, E].
            - split_ratios: Per-path split ratios [B, P, 1].
        """
        split_ratios = self.compute_split_ratios(gammas, batch_size, num_paths_per_pair)
        data_on_tunnels = split_ratios*tm.squeeze(-1)
        
        # Actual matrix
        # with torch.autocast(device_type="cuda", dtype=torch.float32):
        data_on_links = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), data_on_tunnels.to(dtype=torch.float32).t()).t()
        
        if props.dtype == torch.bfloat16:
            data_on_links = data_on_links.to(dtype=torch.bfloat16)
        
        if add_epsilon:
            edges_util = data_on_links/capacities + epsilon
        else:
            edges_util = data_on_links/capacities
        
        return edges_util
