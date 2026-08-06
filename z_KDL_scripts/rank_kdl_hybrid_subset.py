
import argparse

import pickle

import re

from collections import defaultdict

from pathlib import Path



import numpy as np

import torch





def load_traffic_series(traffic_dir=None, traffic_pkl=None):

    if traffic_pkl is not None and Path(traffic_pkl).exists():

        with open(traffic_pkl, "rb") as f:

            x = pickle.load(f)

        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:

            raise ValueError(f"traffic_pkl should be [T,P], got {x.shape}")

        return x



    traffic_dir = Path(traffic_dir)

    files = sorted(

        traffic_dir.glob("t*.pkl"),

        key=lambda p: int(re.findall(r"\d+", p.stem)[0]),

    )

    if not files:

        raise RuntimeError(f"No t*.pkl files found in {traffic_dir}")



    series = []

    for p in files:

        with open(p, "rb") as f:

            x = pickle.load(f)

        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 2 and x.shape[1] == 1:

            x = x[:, 0]

        if x.ndim != 1:

            raise ValueError(f"{p} has shape {x.shape}, expected flat [P]")

        series.append(x)



    return np.stack(series, axis=0)





def load_pairs(pairs_pkl):

    with open(pairs_pkl, "rb") as f:

        pairs = pickle.load(f)



    # Usually list[(s,d)].

    if isinstance(pairs, dict):

        # Try common layouts.

        if "pairs" in pairs:

            pairs = pairs["pairs"]

        elif "sd_pairs" in pairs:

            pairs = pairs["sd_pairs"]

        else:

            raise ValueError(f"Unsupported pairs dict keys: {pairs.keys()}")



    pairs = list(pairs)

    out = []

    for x in pairs:

        if isinstance(x, (list, tuple)) and len(x) >= 2:

            out.append((int(x[0]), int(x[1])))

        else:

            raise ValueError(f"Unsupported pair entry: {x}")



    return out





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_pkl", default="pairs/kdl/t1.pkl")

    ap.add_argument("--traffic_dir", default="traffic_matrices/kdl")

    ap.add_argument("--traffic_pkl", default=None)

    ap.add_argument("--ranking_window", type=int, default=200)

    ap.add_argument("--top_k_per_source", type=int, default=5)

    ap.add_argument("--global_top_n", type=int, default=5000)

    ap.add_argument("--try_global_ns", type=str, default="1000,5000,10000,20000,50000")

    ap.add_argument("--out_metadata", default="kdl_hybrid_subset_metadata.pt")

    args = ap.parse_args()



    pairs = load_pairs(args.pairs_pkl)

    traffic = load_traffic_series(args.traffic_dir, args.traffic_pkl)



    T, P = traffic.shape

    if len(pairs) != P:

        raise RuntimeError(f"pairs length {len(pairs)} does not match traffic P={P}")



    W = min(args.ranking_window, T)

    rank_traffic = traffic[:W]



    # Ranking score: average demand over first W snapshots.

    mean_rank = rank_traffic.mean(axis=0)



    # Volume accounting.

    rank_total_volume = rank_traffic.sum()

    full_total_volume = traffic.sum()



    # Top-k per source.

    by_source = defaultdict(list)

    for idx, (s, d) in enumerate(pairs):

        by_source[s].append((mean_rank[idx], idx, d))



    per_source_selected = set()

    for s, entries in by_source.items():

        entries.sort(reverse=True, key=lambda x: x[0])

        for _, idx, _ in entries[: args.top_k_per_source]:

            per_source_selected.add(idx)



    # Try multiple global N values so we can see where 90% happens.

    try_ns = [int(x) for x in args.try_global_ns.split(",") if x.strip()]

    if args.global_top_n not in try_ns:

        try_ns.append(args.global_top_n)

    try_ns = sorted(set(try_ns))



    order = np.argsort(-mean_rank)



    print("KDL hybrid subset ranking")

    print("-------------------------")

    print(f"traffic shape: {traffic.shape}")

    print(f"ranking window: first {W} snapshots")

    print(f"num SD pairs total: {P}")

    print(f"num sources: {len(by_source)}")

    print(f"top_k_per_source: {args.top_k_per_source}")

    print(f"per-source selected count: {len(per_source_selected)}")

    print()

    print("global_top_n | selected_count | retained_rank_window | retained_all_snapshots")

    print("-------------|----------------|----------------------|-----------------------")



    chosen_selected = None

    chosen_global_n = args.global_top_n



    for N in try_ns:

        global_selected = set(int(i) for i in order[: min(N, P)])

        selected = sorted(per_source_selected | global_selected)



        rank_selected_volume = rank_traffic[:, selected].sum()

        full_selected_volume = traffic[:, selected].sum()



        rank_frac = rank_selected_volume / max(rank_total_volume, 1e-12)

        full_frac = full_selected_volume / max(full_total_volume, 1e-12)



        print(f"{N:12d} | {len(selected):14d} | {100*rank_frac:20.3f}% | {100*full_frac:21.3f}%")



        if N == args.global_top_n:

            chosen_selected = selected



    selected = chosen_selected

    selected_pairs = [pairs[i] for i in selected]



    rank_selected_volume = rank_traffic[:, selected].sum()

    full_selected_volume = traffic[:, selected].sum()



    metadata = {

        "pairs_pkl": args.pairs_pkl,

        "traffic_dir": args.traffic_dir,

        "traffic_pkl": args.traffic_pkl,

        "ranking_window": W,

        "top_k_per_source": args.top_k_per_source,

        "global_top_n": chosen_global_n,

        "num_total_sd_pairs": P,

        "num_selected_sd_pairs": len(selected),

        "selected_path_indices": torch.tensor(selected, dtype=torch.long),

        "selected_sd_pairs": selected_pairs,

        "rank_window_retained_fraction": float(rank_selected_volume / max(rank_total_volume, 1e-12)),

        "all_snapshots_retained_fraction": float(full_selected_volume / max(full_total_volume, 1e-12)),

    }



    torch.save(metadata, args.out_metadata)



    print()

    print(f"Saved selected metadata for global_top_n={chosen_global_n} to {args.out_metadata}")

    print(f"Selected SD pairs: {len(selected)} / {P}")

    print(f"Retained traffic over ranking window: {100*metadata['rank_window_retained_fraction']:.3f}%")

    print(f"Retained traffic over all snapshots:   {100*metadata['all_snapshots_retained_fraction']:.3f}%")



    if metadata["rank_window_retained_fraction"] < 0.90:

        print()

        print("WARNING: selected set retains <90% over ranking window.")

        print("Consider increasing --global_top_n to 10000 or 20000.")





if __name__ == "__main__":

    main()

