"""
bigru_tcn_wrapper
=================

Implementation of the BiGRU-TCN hybrid model for volatility forecasting.
"""

__version__ = "1.1.0"

import math

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dl_base import BasePyTorchVolatilityModel
from src.models.tcn_wrapper import TCNBlock

__all__ = ["BiGRU_TCNModel"]


class BiGRU_TCNNetwork(nn.Module):
    """BiGRU-TCN network with a softplus head for positive variance output."""

    def __init__(
        self,
        bigru_hidden: int,
        tcn_channels: int,
        tcn_levels: int,
        kernel_size: int,
        num_convs: int,
        dense_units: int,
        num_dense_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bigru = nn.GRU(
            input_size=1,
            hidden_size=bigru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        tcn_layers: list[nn.Module] = []
        in_channels = bigru_hidden * 2
        for level_idx in range(tcn_levels):
            dilation = 2 ** level_idx
            tcn_layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=tcn_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    num_convs=num_convs,
                )
            )
            in_channels = tcn_channels
        self.tcn = nn.Sequential(*tcn_layers)

        head_layers: list[nn.Module] = []
        in_features = tcn_channels
        current_units = dense_units
        for layer_idx in range(num_dense_layers):
            head_layers.append(nn.Linear(in_features, current_units))
            head_layers.append(nn.ReLU())
            if dropout > 0.0:
                head_layers.append(nn.Dropout(dropout))
            in_features = current_units
            if layer_idx == 0:
                current_units = max(16, dense_units // 2)

        head_layers.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*head_layers)
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bigru_out, _ = self.bigru(x)
        tcn_in = bigru_out.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in)
        return F.softplus(self.head(tcn_out[:, :, -1])) + self.eps


class BiGRU_TCNModel(BasePyTorchVolatilityModel):
    """Hybrid BiGRU-TCN model wrapper for volatility forecasting."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        look_back = trial.suggest_categorical("look_back", [7, 14, 21, 28])
        kernel_size = trial.suggest_categorical("kernel_size", [2, 3])
        num_convs = trial.suggest_categorical("num_convs", [1, 2, 3])
        min_levels = math.ceil(
            math.log2(1 + (look_back - 1) / (num_convs * (kernel_size - 1)))
        )
        max_levels = min(min_levels + 2, 5)

        return {
            "look_back": look_back,
            "bigru_hidden": trial.suggest_categorical("bigru_hidden", [32, 64, 128]),
            "tcn_channels": trial.suggest_categorical("tcn_channels", [16, 32, 64]),
            "tcn_levels": trial.suggest_int("tcn_levels", min_levels, max_levels),
            "kernel_size": kernel_size,
            "num_convs": num_convs,
            "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128, 256]),
            "num_dense_layers": trial.suggest_categorical("num_dense_layers", [1, 2]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    def __init__(
        self,
        look_back: int,
        bigru_hidden: int,
        tcn_channels: int,
        tcn_levels: int,
        kernel_size: int,
        num_convs: int,
        dense_units: int,
        num_dense_layers: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        **kwargs,
    ) -> None:
        super().__init__(
            look_back=look_back,
            bigru_hidden=bigru_hidden,
            tcn_channels=tcn_channels,
            tcn_levels=tcn_levels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            dense_units=dense_units,
            num_dense_layers=num_dense_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.net = BiGRU_TCNNetwork(
            bigru_hidden=bigru_hidden,
            tcn_channels=tcn_channels,
            tcn_levels=tcn_levels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            dense_units=dense_units,
            num_dense_layers=num_dense_layers,
            dropout=dropout,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]