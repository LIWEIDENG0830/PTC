import argparse
import copy
import time
from pathlib import Path

import torch

from config import STUDENTS, get_config
from data import load_splits
from networks import create_model
from utils import evaluate, seed_everything, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--student", choices=STUDENTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / "COMPLETE").is_file():
        return
    seed_everything(args.seed)
    cfg = get_config(args.config)
    splits = load_splits(args.data_root, cfg)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    x = artifact["x"].float()
    y = artifact["y"].float()
    model, batch_size, learning_rate = create_model(cfg, args.student)
    model = model.to(args.device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=min(batch_size, len(x)),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    history = []
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for bx, by in loader:
            bx = bx.to(args.device)
            by = by.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(bx) - by).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bx)
            count += len(bx)
        if epoch == 1 or epoch % args.eval_every == 0:
            val_mse, val_mae = evaluate(model, splits["val"], batch_size, args.device)
            history.append({"epoch": epoch, "train_mse": total / count, "val_mse": val_mse, "val_mae": val_mae})
            if val_mse < best - 1e-7:
                best = val_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    test_mse, test_mae = evaluate(model, splits["test"], batch_size, args.device)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "config": cfg.name,
        "dataset": cfg.dataset,
        "horizon": cfg.pred_len,
        "k": len(x),
        "method": artifact["method"],
        "student": args.student,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mse": best,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "epochs_ran": epoch,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.0,
    }
    write_csv(output / "result.csv", [result])
    write_csv(output / "history.csv", history)
    (output / "COMPLETE").write_text("ok\n")


if __name__ == "__main__":
    main()
