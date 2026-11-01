import argparse
import subprocess
import sys
from pathlib import Path

from config import BUDGETS, DATASETS, HORIZONS, METHODS, SEEDS, STUDENTS


ROOT = Path(__file__).resolve().parent


def execute(arguments):
    subprocess.run([sys.executable, *map(str, arguments)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("check", "full"), default="check")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.scope == "check":
        datasets = ("ETTm2",)
        horizons = HORIZONS
        budgets = (10,)
        methods = METHODS
        students = ("DLinear",)
        seeds = SEEDS
    else:
        datasets = DATASETS
        horizons = HORIZONS
        budgets = BUDGETS
        methods = METHODS
        students = STUDENTS
        seeds = SEEDS

    for dataset in datasets:
        for horizon in horizons:
            config = f"{dataset}_H{horizon}"
            execute(
                [
                    "construct.py",
                    "--config",
                    config,
                    "--data-root",
                    args.data_root,
                    "--output-root",
                    output_root,
                    "--budgets",
                    *budgets,
                    "--device",
                    args.device,
                ]
            )

    for dataset in datasets:
        for horizon in horizons:
            config = f"{dataset}_H{horizon}"
            for budget in budgets:
                for method in methods:
                    artifact = output_root / "artifacts" / config / f"K{budget}" / f"{method}.pt"
                    for student in students:
                        for seed in seeds:
                            destination = output_root / "runs" / config / f"K{budget}" / method / student / f"seed{seed}"
                            if (destination / "COMPLETE").exists():
                                continue
                            execute(
                                [
                                    "train.py",
                                    "--config",
                                    config,
                                    "--artifact",
                                    artifact,
                                    "--student",
                                    student,
                                    "--seed",
                                    seed,
                                    "--data-root",
                                    args.data_root,
                                    "--output",
                                    destination,
                                    "--device",
                                    args.device,
                                ]
                            )

    execute(["summarize.py", "--result-root", output_root])


if __name__ == "__main__":
    main()
