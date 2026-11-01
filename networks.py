import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent


def itransformer_args(cfg):
    settings = {
        "ETTh1": (2, 512, 512, 32, 1e-4),
        "ETTh2": (2, 128, 128, 32, 1e-4),
        "ETTm1": (2, 512, 512, 32, 1e-4),
        "ETTm2": (2, 128, 128, 32, 1e-4),
        "Exchange": (2, 512, 512, 32, 1e-4),
        "Weather": (3, 512, 512, 32, 1e-4),
        "Electricity": (3, 512, 512, 32, 1e-4),
        "Traffic": (4, 512, 512, 16, 1e-3),
    }
    e_layers, d_model, d_ff, batch_size, learning_rate = settings[cfg.dataset]
    return SimpleNamespace(
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        enc_in=cfg.channels,
        dec_in=cfg.channels,
        c_out=cfg.channels,
        individual=False,
        output_attention=False,
        use_norm=True,
        embed="timeF",
        freq=cfg.freq,
        dropout=0.1,
        class_strategy="projection",
        factor=1,
        d_model=d_model,
        n_heads=8,
        d_ff=d_ff,
        e_layers=e_layers,
        activation="gelu",
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def patchtst_args(cfg):
    small = cfg.dataset in {"ETTh1", "ETTh2", "Exchange"}
    if small:
        n_heads, d_model, d_ff, dropout, batch_size = 4, 16, 128, 0.3, 128
    else:
        n_heads, d_model, d_ff, dropout = 16, 128, 256, 0.2
        batch_size = {"Electricity": 32, "Traffic": 24}.get(cfg.dataset, 128)
    return SimpleNamespace(
        enc_in=cfg.channels,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        e_layers=3,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
        fc_dropout=dropout,
        head_dropout=0.0,
        individual=False,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        kernel_size=25,
        batch_size=batch_size,
        learning_rate=1e-4,
    )


class WrappedModel(nn.Module):
    def __init__(self, model, four_arguments):
        super().__init__()
        self.model = model
        self.four_arguments = four_arguments

    def forward(self, x):
        if self.four_arguments:
            return self.model(x, None, None, None)
        return self.model(x)


def create_model(cfg, name):
    if name == "DLinear":
        sys.path.insert(0, str(ROOT / "vendor"))
        from dlinear import Model
        model = Model(itransformer_args(cfg))
        batch_size = itransformer_args(cfg).batch_size
        learning_rate = itransformer_args(cfg).learning_rate
        if cfg.dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}:
            learning_rate = {96: 1e-3, 192: 1e-3, 336: 1e-2, 720: 1e-1}[cfg.pred_len]
        elif cfg.dataset == "Weather":
            batch_size, learning_rate = 16, 1e-4
        elif cfg.dataset == "Traffic":
            batch_size, learning_rate = 16, 5e-2
        else:
            batch_size, learning_rate = 32, 1e-3
        return WrappedModel(model, False), batch_size, learning_rate
    if name == "iTransformer":
        sys.path.insert(0, str(ROOT / "vendor" / "itransformer"))
        from model.iTransformer import Model
        args = itransformer_args(cfg)
        return WrappedModel(Model(args), True), args.batch_size, args.learning_rate
    if name == "PatchTST":
        sys.path.insert(0, str(ROOT / "vendor" / "patchtst"))
        from models.PatchTST import Model
        args = patchtst_args(cfg)
        return WrappedModel(Model(args), False), args.batch_size, args.learning_rate
    raise ValueError(name)
