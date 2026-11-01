import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import locate_csv


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, values, begin, end, seq_len, pred_len):
        self.values = torch.from_numpy(values[begin:end])
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        start = int(index)
        middle = start + self.seq_len
        end = middle + self.pred_len
        return self.values[start:middle], self.values[middle:end]


def load_splits(data_root, cfg):
    frame = pd.read_csv(locate_csv(data_root, cfg))
    values = frame.iloc[:, 1:].to_numpy(np.float32)
    n = len(values)
    if cfg.data_type == "ETTm":
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
    elif cfg.data_type == "ETTh":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    else:
        train_end = int(n * 0.7)
        test_size = int(n * 0.2)
        val_end = n - test_size
        test_end = n
    scaler = StandardScaler().fit(values[:train_end])
    values = scaler.transform(values).astype(np.float32)
    boundaries = {
        "train": (0, train_end),
        "val": (train_end - cfg.seq_len, val_end),
        "test": (val_end - cfg.seq_len, test_end),
    }
    return {
        split: WindowDataset(values, begin, end, cfg.seq_len, cfg.pred_len)
        for split, (begin, end) in boundaries.items()
    }
