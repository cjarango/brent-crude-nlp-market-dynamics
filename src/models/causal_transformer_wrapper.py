import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = ["CausalTransformerModel"]

_ACTIVATIONS = {
    "relu": "relu",
    "gelu": "gelu"
}


class CausalTransformerNetwork(nn.Module):
    """Causal Transformer encoder over return windows with a positive variance head."""

    def __init__(
        self,
        in_features: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation='{activation}'. Use one of {list(_ACTIVATIONS)}.")

        self.seq_len = seq_len
        self.eps = eps

        self.input_proj = nn.Linear(in_features, d_model)
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
        # x: (batch, seq_len, in_features)
        seq_len = x.size(1)
        h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        mask = self.causal_mask[:seq_len, :seq_len]
        h = self.encoder(h, mask=mask)
        out = self.norm(h[:, -1, :])
        return F.softplus(self.fc(out)) + self.eps


class CausalTransformerModel(BasePyTorchVolatilityModel):
    """Causal Transformer wrapper compatible with BasePyTorchVolatilityModel and WalkForwardTrainer."""

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
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
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
            dropout=dropout,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            in_features=in_features,
            **kwargs,
        )

        self.net = CausalTransformerNetwork(
            in_features=in_features,
            seq_len=look_back,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]