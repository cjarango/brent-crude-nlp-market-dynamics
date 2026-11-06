"""
gru_wrapper
===========

Implementation of the GRU model for volatility forecasting.
"""

__version__ = "1.4.0"

import optuna
import torch.nn as nn
import torch.nn.functional as F

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = ["GRUModel"]


class GRUNetwork(nn.Module):
    """GRU with a softplus head that guarantees positive variance output."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.eps = 1e-8

    def forward(self, x):
        out, _ = self.gru(x)
        return F.softplus(self.fc(out[:, -1, :])) + self.eps


class GRUModel(BasePyTorchVolatilityModel):
    """GRU model wrapper for volatility forecasting."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    def __init__(
        self, 
        look_back: int, 
        hidden_size: int, 
        num_layers: int, 
        dropout: float, 
        learning_rate: float,
        weight_decay: float,
        **kwargs
    ) -> None:
        effective_dropout = dropout if num_layers > 1 else 0.0
        
        super().__init__(
            look_back=look_back, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout=effective_dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
        
        self.net = GRUNetwork(
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=effective_dropout,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]