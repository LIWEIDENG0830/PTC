import argparse
from pathlib import Path

import torch

from config import BUDGETS, METHODS, get_config
from data import load_splits
from ptc import build_artifact, fit_projection_and_scores, selection_orders
from utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(17)
    cfg = get_config(args.config)
    train = load_splits(args.data_root, cfg)["train"]
    batch_size = 8 if cfg.channels > 100 else 128
    theta, scores, ridge = fit_projection_and_scores(train, args.device, batch_size)
    orders = selection_orders(scores, max(args.budgets))
    root = Path(args.output_root)
    for k in args.budgets:
        for method in METHODS:
            artifact = build_artifact(train, theta, orders[method][:k], method)
            artifact.update(config=cfg.name, ridge=ridge, selection_seed=7 if method == "random" else 17)
            path = root / "artifacts" / cfg.name / f"K{k}" / f"{method}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, path)
    marker = root / "artifacts" / cfg.name / "COMPLETE"
    marker.write_text("ok\n")


if __name__ == "__main__":
    main()
