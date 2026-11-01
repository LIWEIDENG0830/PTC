import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    files = list((root / "runs").glob("**/result.csv"))
    if not files:
        raise RuntimeError("No results found")
    runs = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    keys = ["dataset", "horizon", "k", "method", "student"]
    per_horizon = runs.groupby(keys, as_index=False).agg(
        seeds=("seed", "size"),
        test_mse=("test_mse", "mean"),
        test_mse_std=("test_mse", "std"),
        test_mae=("test_mae", "mean"),
        test_mae_std=("test_mae", "std"),
    )
    aggregate = per_horizon.groupby(["dataset", "k", "method", "student"], as_index=False).agg(
        horizons=("horizon", "size"),
        test_mse=("test_mse", "mean"),
        test_mae=("test_mae", "mean"),
    )
    runs.sort_values(keys + ["seed"]).to_csv(root / "all_runs.csv", index=False)
    per_horizon.to_csv(root / "per_horizon_summary.csv", index=False)
    aggregate.to_csv(root / "four_horizon_summary.csv", index=False)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
