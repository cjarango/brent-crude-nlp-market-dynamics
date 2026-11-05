"""
tcn_wrapper
===========

Implementation of the Temporal Convolutional Network (TCN) for volatility forecasting.

Architecture based on:
    Yu, Y., Dai, D., Lin, Y., Zhang, X., Li, Z., & He, M. (2026).
    "Forecasting crude oil price volatility utilizing multivariate decomposition
    and deep learning approaches: Considering the role of influencing factors."
    Energy, 360, 141916.
    
    And the original TCN formulation by:
    Bai, S., Kolter, J. Z., & Koltun, V. (2018).
    "An empirical evaluation of generic convolutional and recurrent networks
    for sequence modeling." arXiv:1803.01271.
"""

__version__ = "1.1.0"  # Subimos versión por el cambio de hiperparámetro

import math

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = ["TCNModel"]


class CausalConv1d(nn.Module):
    """1D convolution with causal padding (left-only) and WeightNorm."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual block with N dilated causal convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, num_convs):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_convs):
            in_c = in_channels if i == 0 else out_channels
            self.convs.append(CausalConv1d(in_c, out_channels, kernel_size, dilation))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(dropout))
            
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) 
            if in_channels != out_channels else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = x
        for conv, relu, dropout in zip(self.convs, self.relus, self.dropouts):
            out = dropout(relu(conv(out)))
            
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNNetwork(nn.Module):
    """TCN with a softplus head that guarantees positive variance output."""
    def __init__(self, num_channels: list, kernel_size: int, dropout: float, num_convs: int):
        super().__init__()
        layers = []
        for i in range(len(num_channels) - 1):
            layers.append(
                TCNBlock(
                    in_channels=num_channels[i],
                    out_channels=num_channels[i+1],
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                    num_convs=num_convs
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.eps = 1e-8

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]
        return F.softplus(self.fc(out)) + self.eps


class TCNModel(BasePyTorchVolatilityModel):
    """TCN model wrapper for volatility forecasting."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        look_back = trial.suggest_categorical("look_back", [7, 14, 21, 28])
        kernel_size = trial.suggest_categorical("kernel_size", [2, 3])
        num_convs = trial.suggest_categorical("num_convs", [1, 2, 3])
        
        # Ensure the receptive field covers the entire look_back window
        # RF = 1 + num_convs * (kernel_size - 1) * (2^num_levels - 1) >= look_back
        min_levels = math.ceil(
            math.log2(1 + (look_back - 1) / (num_convs * (kernel_size - 1)))
        )
        
        return {
            "look_back": look_back,
            "num_levels": trial.suggest_int("num_levels", min_levels, min_levels + 2),
            "channels": trial.suggest_categorical("channels", [16, 32, 64]),
            "kernel_size": kernel_size,
            "num_convs": num_convs,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    def __init__(
        self, 
        look_back: int, 
        num_levels: int, 
        channels: int,
        kernel_size: int,
        num_convs: int,
        dropout: float, 
        learning_rate: float,
        weight_decay: float,
        **kwargs
    ) -> None:
        
        num_channels = [1] + [channels] * num_levels
        
        super().__init__(
            look_back=look_back, 
            num_levels=num_levels,
            channels=channels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
        
        self.net = TCNNetwork(
            num_channels=num_channels, 
            kernel_size=kernel_size, 
            dropout=dropout,
            num_convs=num_convs,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]