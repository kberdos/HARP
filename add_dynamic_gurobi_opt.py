import argparse
import glob
import math
from pathlib import Path

import torch
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm


def load_sample(path):
    return torch.load(path, map_location="cpu")


def get_final_sample_tensors(sample):
    """
    Extract final-timestep TE instance from one dynamic sample.

    Expected:
        tm: [1, T, P, 1]
        capacities[-1]: [1, E] or [E]
        paths_to_edges[-1]: sparse [P, E]
    """
    tm = sample["tm"][:, -1]  # [1, P, 1]
    tm = tm.squeeze(0).squeeze(-1).to(dtype=torch.float64)  # [P]

    capacities = sample["capacities"][-1]
    if capacities.dim() == 2:
        capacities = capacities.squeeze(0)
    capacities = capacities.to(dtype=torch.float64)  # [E]

    pte = sample["paths_to_edges"][-1].coalesce()  # [P, E]

    return tm, capacities, pte


def solve_optimal_mlu(sample, k_paths, time_limit=None, verbose=False):
    """
    Solve the optimal path-splitting LP:

        minimize U

        For each source-destination pair i:
            sum_{k=1..K} x_{i,k} = 1

        For each edge e:
            sum_p pte[p,e] * demand[p] * x[p] <= U * capacity[e]

        x[p] >= 0

    Here demand[p] is repeated per path slot in the sample's tm.

    Returns:
        opt: scalar optimal MLU
        opt_splits: [1, P] tensor of optimal path split ratios
        status: Gurobi status
        runtime: Gurobi runtime in seconds
    """
    tm, capacities, pte = get_final_sample_tensors(sample)

    P = tm.numel()
    E = capacities.numel()

    if P % k_paths != 0:
        raise ValueError(f"P={P} is not divisible by k_paths={k_paths}")

    num_pairs = P // k_paths

    if pte.shape[0] != P:
        raise ValueError(f"paths_to_edges has P={pte.shape[0]}, but tm has P={P}")

    if pte.shape[1] != E:
        raise ValueError(f"paths_to_edges has E={pte.shape[1]}, but capacities has E={E}")

    if torch.any(capacities <= 0):
        raise ValueError("Capacities must be positive")

    # Convert sparse matrix into edge -> path incidence for faster constraint construction.
    indices = pte.indices()
    values = pte.values()

    edge_to_paths = [[] for _ in range(E)]
    for idx in range(indices.shape[1]):
        p = int(indices[0, idx].item())
        e = int(indices[1, idx].item())
        val = float(values[idx].item())
        if val != 0.0:
            edge_to_paths[e].append(p)

    m = gp.Model("dynamic_abilene_mlu")
    m.Params.OutputFlag = 1 if verbose else 0

    if time_limit is not None and time_limit > 0:
        m.Params.TimeLimit = float(time_limit)

    # x[p] = fraction of pair demand assigned to path p.
    x = m.addVars(P, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # U = maximum link utilization.
    U = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="U")

    # Per-pair splitting constraints.
    for pair_idx in range(num_pairs):
        start = pair_idx * k_paths
        end = start + k_paths
        m.addConstr(gp.quicksum(x[p] for p in range(start, end)) == 1.0)

    # Edge utilization constraints.
    for e in range(E):
        cap_e = float(capacities[e].item())
        expr = gp.LinExpr()

        for p in edge_to_paths[e]:
            demand_p = float(tm[p].item())
            if demand_p != 0.0:
                expr += demand_p * x[p]

        m.addConstr(expr <= U * cap_e)

    m.setObjective(U, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        raise RuntimeError(f"Gurobi did not find a usable solution. Status={m.Status}")

    if m.SolCount == 0:
        raise RuntimeError(f"Gurobi returned no solution. Status={m.Status}")

    opt = float(U.X)

    if not math.isfinite(opt):
        raise RuntimeError(f"Optimal MLU is not finite: {opt}")

    opt_splits = torch.tensor([x[p].X for p in range(P)], dtype=torch.float32).view(1, P)

    # Small numerical cleanup. Gurobi may return tiny negative/above-one values due to tolerances.
    opt_splits = opt_splits.clamp(min=0.0, max=1.0)

    # Renormalize each SD-pair group to sum to 1. This keeps the label compatible with softmax outputs.
    opt_splits_grouped = opt_splits.view(1, num_pairs, k_paths)
    denom = opt_splits_grouped.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    opt_splits = (opt_splits_grouped / denom).view(1, P)

    return opt, opt_splits, m.Status, float(m.Runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--k_paths", type=int, required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--time_limit", type=float, default=None)
    parser.add_argument("--verbose_gurobi", type=int, default=0)

    args = parser.parse_args()

    files = sorted(glob.glob(str(Path(args.samples_dir) / "sample_*.pt")))

    if args.end_idx is None:
        args.end_idx = len(files)

    files = files[args.start_idx:args.end_idx]

    if len(files) == 0:
        raise ValueError(
            f"No sample files found in {args.samples_dir} "
            f"for slice [{args.start_idx}:{args.end_idx}]"
        )

    solved = 0
    skipped = 0
    failed = 0

    for fp in tqdm(files, desc="Adding Gurobi opt + splits"):
        sample = load_sample(fp)

        if "opt" in sample and "opt_splits" in sample and not args.overwrite:
            skipped += 1
            continue

        try:
            opt, opt_splits, status, runtime = solve_optimal_mlu(
                sample=sample,
                k_paths=args.k_paths,
                time_limit=args.time_limit,
                verbose=bool(args.verbose_gurobi),
            )

            sample["opt"] = torch.tensor(opt, dtype=torch.float32)
            sample["opt_splits"] = opt_splits
            sample["opt_status"] = int(status)
            sample["opt_runtime"] = float(runtime)

            torch.save(sample, fp)
            solved += 1

        except Exception as e:
            failed += 1
            print(f"[FAILED] {fp}: {e}")

    print("Done.")
    print(f"Solved:  {solved}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
