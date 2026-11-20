"""
mswt_gru_mlp
============

Multi-Scale Wavelet Transform GRU with an MLP head over per-band states.
"""

__version__ = "1.2.0"

import numpy as np
import optuna
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = [
    "MSWTGRUMLPModel",
    "MSWTGRUMLPNetwork",
    "compute_causal_dwt_bands",
    "default_decomp_window",
    "min_window_length",
]

_DWT_MODE = "symmetric"


def min_window_length(wavelet: str, J: int) -> int:
    filter_len = pywt.Wavelet(wavelet).dec_len
    return (filter_len - 1) * (2 ** J)


def default_decomp_window(wavelet: str, J: int) -> int:
    """Computes a safe causal decomposition window automatically from wavelet and J."""
    min_len = min_window_length(wavelet, J)
    return max(min_len, 2 * min_len)


def _decompose_causal_prefix(prefix_np: np.ndarray, wavelet: str, J: int) -> np.ndarray:
    coeffs = pywt.wavedec(prefix_np, wavelet=wavelet, level=J, mode=_DWT_MODE)
    n = len(prefix_np)
    step_bands = np.empty(len(coeffs), dtype=np.float32)
    for k in range(len(coeffs)):
        isolated = [np.zeros_like(c) for c in coeffs]
        isolated[k] = coeffs[k]
        rec = pywt.waverec(isolated, wavelet=wavelet, mode=_DWT_MODE)
        step_bands[k] = rec[n - 1]
    return step_bands


def compute_causal_dwt_bands(
    series: np.ndarray,
    wavelet: str,
    J: int,
    decomp_window: int | None = None,
) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64).flatten()
    n_samples = len(arr)
    min_len = min_window_length(wavelet, J)
    if n_samples < min_len:
        raise ValueError(
            f"Series length ({n_samples}) is smaller than minimum required "
            f"for wavelet='{wavelet}' and J={J} ({min_len})."
        )

    eff_win = decomp_window if decomp_window is not None else default_decomp_window(wavelet, J)
    out = np.zeros((n_samples, J + 1), dtype=np.float32)
    for t in range(n_samples):
        start = max(0, t + 1 - eff_win)
        prefix = arr[start : t + 1]
        if len(prefix) < min_len:
            prefix = np.pad(prefix, (min_len - len(prefix), 0), mode="edge")
        out[t] = _decompose_causal_prefix(prefix, wavelet=wavelet, J=J)
    return out


def _build_activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'. Use 'gelu', 'silu', or 'relu'.")


class MSWTGRUMLPNetwork(nn.Module):
    """Per-band GRU network whose concatenated last states feed an MLP head."""

    def __init__(
        self,
        J: int,
        hidden_size: int,
        num_layers: int,
        gru_dropout: float,
        mlp_dropout: float,
        mlp_hidden_units: list[int],
        activation: str | list[str] = "gelu",
    ) -> None:
        super().__init__()
        effective_gru_dropout = gru_dropout if num_layers > 1 else 0.0

        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=effective_gru_dropout,
                )
                for _ in range(J + 1)
            ]
        )

        activations = (
            [activation] * len(mlp_hidden_units)
            if isinstance(activation, str)
            else activation
        )
        if len(activations) != len(mlp_hidden_units):
            raise ValueError("activation list must match mlp_hidden_units length.")

        layers: list[nn.Module] = []
        in_features = (J + 1) * hidden_size
        for units, act_name in zip(mlp_hidden_units, activations):
            layers.append(nn.Linear(in_features, units))
            layers.append(_build_activation(act_name))
            if mlp_dropout > 0.0:
                layers.append(nn.Dropout(mlp_dropout))
            in_features = units

        layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*layers)
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        band_states: list[torch.Tensor] = []
        for k, gru in enumerate(self.grus):
            out, _ = gru(x[:, :, k : k + 1])
            band_states.append(out[:, -1, :])

        concat_states = torch.cat(band_states, dim=-1)
        return F.softplus(self.mlp(concat_states)) + self.eps


class MSWTGRUMLPModel(BasePyTorchVolatilityModel):
    """MS-WT-GRU-MLP model wrapper for volatility forecasting."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        mlp_choice = trial.suggest_categorical(
            "mlp_arch", ["64", "128", "128_64"]
        )
        mlp_map: dict[str, list[int]] = {
            "64": [64],
            "128": [128],
            "128_64": [128, 64],
        }
        return {
            "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
            "wavelet": "db4",
            "J": trial.suggest_categorical("J", [1, 2, 3]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "mlp_hidden_units": mlp_map[mlp_choice],
            "activation": trial.suggest_categorical("activation", ["gelu", "silu", "relu"]),
            "gru_dropout": trial.suggest_float("gru_dropout", 0.0, 0.3),
            "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }

    def __init__(
        self,
        look_back: int,
        J: int,
        hidden_size: int,
        num_layers: int,
        gru_dropout: float,
        mlp_dropout: float,
        learning_rate: float,
        weight_decay: float,
        wavelet: str = "db4",
        decomp_window: int | None = None,
        mlp_hidden_units: list[int] | None = None,
        activation: str | list[str] = "gelu",
        **kwargs,
    ) -> None:
        min_len = min_window_length(wavelet, J)
        effective_decomp = (
            decomp_window if decomp_window is not None else default_decomp_window(wavelet, J)
        )
        if effective_decomp < min_len:
            raise ValueError(
                f"[{self.__class__.__name__}] decomp_window={effective_decomp} < {min_len} "
                f"for wavelet='{wavelet}' and J={J}."
            )

        effective_gru_dropout = gru_dropout if num_layers > 1 else 0.0
        resolved_units = mlp_hidden_units if mlp_hidden_units is not None else [64]

        super().__init__(
            look_back=look_back,
            decomp_window=effective_decomp,
            wavelet=wavelet,
            J=J,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gru_dropout=effective_gru_dropout,
            mlp_dropout=mlp_dropout,
            mlp_hidden_units=resolved_units,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.net = MSWTGRUMLPNetwork(
            J=J,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gru_dropout=effective_gru_dropout,
            mlp_dropout=mlp_dropout,
            mlp_hidden_units=resolved_units,
            activation=activation,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]