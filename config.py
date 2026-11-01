from pathlib import Path
from types import SimpleNamespace


DATASETS = {
    "ETTh1": (7, "ETTh", "h", ("ETT-small/ETTh1.csv", "ETTh1.csv")),
    "ETTh2": (7, "ETTh", "h", ("ETT-small/ETTh2.csv", "ETTh2.csv")),
    "ETTm1": (7, "ETTm", "t", ("ETT-small/ETTm1.csv", "ETTm1.csv")),
    "ETTm2": (7, "ETTm", "t", ("ETT-small/ETTm2.csv", "ETTm2.csv")),
    "Exchange": (8, "custom", "d", ("exchange_rate/exchange_rate.csv", "exchange_rate.csv")),
    "Weather": (21, "custom", "h", ("weather/weather.csv", "weather.csv")),
    "Electricity": (321, "custom", "h", ("electricity/electricity.csv", "electricity.csv")),
    "Traffic": (862, "custom", "h", ("traffic/traffic.csv", "traffic.csv")),
}
HORIZONS = (96, 192, 336, 720)
BUDGETS = (10, 25)
STUDENTS = ("DLinear", "iTransformer", "PatchTST")
METHODS = ("random", "ptc")
SEEDS = (0, 1, 2, 3, 4)


def get_config(name):
    dataset, horizon_text = name.rsplit("_H", 1)
    if dataset not in DATASETS or int(horizon_text) not in HORIZONS:
        raise ValueError(name)
    channels, data_type, freq, relpaths = DATASETS[dataset]
    return SimpleNamespace(
        name=name,
        dataset=dataset,
        pred_len=int(horizon_text),
        channels=channels,
        data_type=data_type,
        freq=freq,
        relpaths=relpaths,
        seq_len=96,
        label_len=48,
    )


def locate_csv(root, cfg):
    root = Path(root)
    for relpath in cfg.relpaths:
        path = root / relpath
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {cfg.dataset} below {root}")
