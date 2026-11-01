import csv
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataset, batch_size, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            prediction = model(x.to(device)).cpu()
            difference = prediction - y
            squared_error += difference.square().sum().item()
            absolute_error += difference.abs().sum().item()
            count += difference.numel()
    return squared_error / count, absolute_error / count


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
