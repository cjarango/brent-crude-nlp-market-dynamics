"""
conv_transformer_wrapper
========================

Implementation of the Convolution Transformer model for volatility forecasting.
"""

__version__ = "1.0.0"

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = ["ConvTransformerModel"]

_ACTIVATIONS = {
    "relu": "relu",
    "gelu": "gelu",
}

_ACT_MODULES = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class CausalConv1d(nn.Module):
    """1D convolution padded exclusively on the left to preserve causality."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ConvTransformerNetwork(nn.Module):
    """Causal Convolutional Transformer encoder over return windows with a positive variance head."""

    def __init__(
        self,
        in_features: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        conv_kernel_size: int = 3,
        activation: str = "gelu",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation='{activation}'. Use one of {list(_ACTIVATIONS)}."
            )
        if conv_kernel_size < 1:
            raise ValueError(f"conv_kernel_size ({conv_kernel_size}) must be >= 1.")

        self.seq_len = seq_len
        self.eps = eps

        self.input_conv = CausalConv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=conv_kernel_size,
            dilation=1,
        )
        self.input_act = _ACT_MODULES[activation]()

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=_ACTIVATIONS[activation],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.seq_len:
            raise ValueError(
                f"Input seq_len ({seq_len}) exceeds configured seq_len ({self.seq_len})."
            )
        h = self.input_act(self.input_conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        h = h + self.pos_embedding[:, :seq_len, :]
        mask = self.causal_mask[:seq_len, :seq_len]
        h = self.encoder(h, mask=mask)
        out = self.norm(h[:, -1, :])
        return F.softplus(self.fc(out)) + self.eps


class ConvTransformerModel(BasePyTorchVolatilityModel):
    """Convolution Transformer wrapper compatible with BasePyTorchVolatilityModel and WalkForwardTrainer."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        d_model = trial.suggest_categorical("d_model", [32, 64])
        nhead = trial.suggest_categorical("nhead", [2, 4])
        return {
            "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dim_feedforward": d_model * 2,
            "conv_kernel_size": trial.suggest_categorical("conv_kernel_size", [2, 3, 5]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "max_epochs": 200,
            "patience": 20,
            "batch_size": 128,
            "retrain_freq": 1,
            "seed": 42,
        }

    def __init__(
        self,
        look_back: int = 21,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "gelu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        in_features: int = 1,
        **kwargs,
    ) -> None:
        ff_dim = dim_feedforward if dim_feedforward is not None else d_model * 2
        super().__init__(
            look_back=look_back,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=ff_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            in_features=in_features,
            **kwargs,
        )

        self.net = ConvTransformerNetwork(
            in_features=in_features,
            seq_len=look_back,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            activation=activation,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]