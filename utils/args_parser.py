def add_default_args(parser):
    # Topology arguments
    parser.add_argument("--topo", type=str, help="Name of the topology to be used.")
    parser.add_argument("--weight", type=str, default=None, help="Name of metric used to represent weights of the edges.")
    parser.add_argument("--metric", type=str, default="MLU", help="Only supports MLU for now.")

    # HARP arguments
    parser.add_argument("--mode", type=str, default="train", help="Mode of operation: train/test.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_paths_per_pair", type=int, default=8, help="Number of paths per source-destination pair.")
    parser.add_argument("--framework", type=str, default="harp", help="Framework to use.")
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=0)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_mlp1_hidden_layers", type=int, default=2)
    parser.add_argument("--num_mlp2_hidden_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--num_for_loops", type=int, default=3)
    parser.add_argument("--failure_id", type=int, default=None)
    parser.add_argument("--dynamic", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--checkpoint", type=int, default=0)
    parser.add_argument("--meta_learning", type=int, default=0)

    # Static/original HARP split arguments
    parser.add_argument("--train_start_indices", type=int, nargs="+", default=None)
    parser.add_argument("--train_end_indices", type=int, nargs="+", default=None)
    parser.add_argument("--val_start_indices", type=int, nargs="+", default=None)
    parser.add_argument("--val_end_indices", type=int, nargs="+", default=None)
    parser.add_argument("--test_start_idx", type=int, default=None)
    parser.add_argument("--test_end_idx", type=int, default=None)
    parser.add_argument("--train_clusters", type=int, nargs="+", default=None)
    parser.add_argument("--val_clusters", type=int, nargs="+", default=None)
    parser.add_argument("--test_cluster", type=int, default=None)

    # Gurobi optimal computation arguments
    parser.add_argument("--opt_start_idx", type=int, default=None)
    parser.add_argument("--opt_end_idx", type=int, default=None)

    # Prediction arguments
    parser.add_argument("--pred", type=int, default=0)
    parser.add_argument("--pred_type", type=str, default="esm")

    # Dynamic temporal-topology sample arguments
    parser.add_argument("--dynamic_samples_dir", type=str, default=None)
    parser.add_argument("--dynamic_train_start_idx", type=int, default=0)
    parser.add_argument("--dynamic_train_end_idx", type=int, default=None)
    parser.add_argument("--dynamic_val_start_idx", type=int, default=None)
    parser.add_argument("--dynamic_val_end_idx", type=int, default=None)
    parser.add_argument("--dynamic_test_start_idx", type=int, default=0)
    parser.add_argument("--dynamic_test_end_idx", type=int, default=None)

    # If dynamic samples include sample["opt"], this controls whether to use it.
    # For the no-Gurobi resilience objective, set this to 0.
    parser.add_argument("--use_dynamic_opt", type=int, default=1)

    # Optional Gurobi split imitation / distillation loss.
    # Requires sample["opt_splits"] from add_dynamic_gurobi_opt.py.
    # For the no-Gurobi resilience objective, keep this at 0.
    parser.add_argument("--split_loss_weight", type=float, default=0.0)

    # Resilience-objective arguments.
    parser.add_argument("--use_resilience_objective", type=int, default=0)
    parser.add_argument("--resilience_loss_weight", type=float, default=0.0)
    parser.add_argument("--soft_backup_penalty_weight", type=float, default=0.0)
    parser.add_argument("--candidate_local_rescale", type=int, default=1)
    parser.add_argument("--disconnected_penalty", type=float, default=100.0)

    # Temporal-HARP extra args used by harp_system.py
    parser.add_argument("--use_temporal_cls", type=int, default=1)
    parser.add_argument("--use_temporal_residual", type=int, default=1)
    parser.add_argument("--temporal_recency_alpha", type=float, default=0.1)

    # Internal/debug flag. Dynamic training_utils turns this on so HARP returns
    # both edge utilizations and predicted path split ratios.
    parser.add_argument("--return_splits", type=int, default=0)

    return parser


def parse_args(args):
    import argparse

    # This prevents --dynamic from being confused with --dynamic_samples_dir.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = add_default_args(parser)
    args_ = parser.parse_args(args)

    args_.dynamic = bool(args_.dynamic)
    args_.checkpoint = bool(args_.checkpoint)
    args_.meta_learning = bool(args_.meta_learning)
    args_.pred = bool(args_.pred)
    args_.use_dynamic_opt = bool(args_.use_dynamic_opt)
    args_.use_resilience_objective = bool(args_.use_resilience_objective)
    args_.candidate_local_rescale = bool(args_.candidate_local_rescale)
    args_.use_temporal_cls = bool(args_.use_temporal_cls)
    args_.use_temporal_residual = bool(args_.use_temporal_residual)
    args_.return_splits = bool(args_.return_splits)

    return args_
