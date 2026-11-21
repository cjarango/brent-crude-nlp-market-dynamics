from __future__ import annotations

import math

import numpy as np
import optuna
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = [
    "BandCausalTCNModel",
    "BandCausalTCNNetwork",
    "CausalConv1d",
    "TCNBlock",
    "compute_causal_reconstructed_bands",
    "default_band_window",
    "min_band_window_length",
]

_DWT_MODE = "symmetric"


def min_band_window_length(wavelet: str, level: int) -> int:
    filter_len = pywt.Wavelet(wavelet).dec_len
    return max(1, (filter_len - 1) * (2**level))


def default_band_window(wavelet: str, level: int) -> int:
    min_len = min_band_window_length(wavelet, level)
    return max(min_len, 2 * min_len)


def _step_causal_band_rec(
    prefix: np.ndarray,
    wavelet: str = "sym4",
    level: int = 3,
) -> np.ndarray:
    """Computes reconstructed values at the final sample of `prefix` for each wavelet band."""
    arr = np.asarray(prefix, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(level + 1, dtype=np.float32)

    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level, mode=_DWT_MODE)
    band_vals = np.empty(len(coeffs), dtype=np.float32)

    for k in range(len(coeffs)):
        isolated = [np.zeros_like(c) for c in coeffs]
        isolated[k] = coeffs[k]
        rec = pywt.waverec(isolated, wavelet=wavelet, mode=_DWT_MODE)
        band_vals[k] = float(rec[n - 1])

    return band_vals


def compute_causal_reconstructed_bands(
    series: np.ndarray,
    wavelet: str = "sym4",
    level: int = 3,
    decomp_window: int | None = None,
) -> np.ndarray:
    """Computes strictly causal reconstructed bands of shape (len(series), level + 1)."""
    arr = np.asarray(series, dtype=np.float64).flatten()
    n_samples = len(arr)
    min_len = min_band_window_length(wavelet, level)
    if n_samples < min_len:
        raise ValueError(
            f"Series length ({n_samples}) < min_band_window_length({wavelet}, {level})={min_len}."
        )

    eff_win = decomp_window if decomp_window is not None else default_band_window(wavelet, level)
    out = np.zeros((n_samples, level + 1), dtype=np.float32)

    for t in range(n_samples):
        start = max(0, t + 1 - eff_win)
        window = arr[start : t + 1]
        if len(window) < min_len:
            window = np.pad(window, (min_len - len(window), 0), mode="edge")
        out[t] = _step_causal_band_rec(window, wavelet=wavelet, level=level)

    return out


class CausalConv1d(nn.Module):
    """1D convolution with causal padding (left-only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual block with N dilated causal convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        num_convs: int,
    ) -> None:
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
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for conv, relu, dropout in zip(self.convs, self.relus, self.dropouts):
            out = dropout(relu(conv(out)))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class BandCausalTCNNetwork(nn.Module):
    """Computes causal wavelet bands on input windows, applies scale-adaptive band filters,
    cross-band mixing, direct band residual, and residual dilated TCN blocks."""

    def __init__(
        self,
        num_bands: int,
        channels: int,
        num_levels: int,
        kernel_size: int,
        dropout: float,
        num_convs: int,
        wavelet: str = "sym4",
        level: int = 3,
        decomp_window: int | None = None,
        band_kernel_size: int = 1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.wavelet = wavelet
        self.level = level
        min_len = min_band_window_length(wavelet, level)
        self.min_len = min_len
        self.decomp_window = (
            decomp_window if decomp_window is not None else default_band_window(wavelet, level)
        )
        if self.decomp_window < min_len:
            raise ValueError(
                f"decomp_window={self.decomp_window} < {min_len} "
                f"for wavelet='{wavelet}' and level={level}."
            )

        base_k = max(1, int(band_kernel_size))
        self.band_kernel_sizes = [base_k + b for b in range(num_bands)]

        self.band_convs = nn.ModuleList(
            [
                CausalConv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k_band,
                    dilation=1,
                    groups=1,
                )
                for k_band in self.band_kernel_sizes
            ]
        )

        self.band_mix = nn.Conv1d(num_bands, channels, kernel_size=1)
        self.band_residual = nn.Conv1d(num_bands, channels, kernel_size=1)

        blocks = [
            TCNBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=2**i,
                dropout=dropout,
                num_convs=num_convs,
            )
            for i in range(num_levels)
        ]
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels, 1)
        self.eps = eps
        self._window_cache: dict[bytes, np.ndarray] = {}

    def clear_cache(self) -> None:
        self._window_cache.clear()

    def _window_to_causal_bands(self, seq_1d: np.ndarray) -> np.ndarray:
        seq_f32 = np.asarray(seq_1d, dtype=np.float32)
        key = seq_f32.tobytes()
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached

        n_samples = len(seq_f32)
        out = np.zeros((n_samples, self.num_bands), dtype=np.float32)
        for t in range(n_samples):
            start = max(0, t + 1 - self.decomp_window)
            window = seq_f32[start : t + 1]
            if len(window) < self.min_len:
                window = np.pad(window, (self.min_len - len(window), 0), mode="edge")
            out[t] = _step_causal_band_rec(window, wavelet=self.wavelet, level=self.level)

        self._window_cache[key] = out
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives from BasePyTorchVolatilityModel as (batch, seq_len, 1)
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        bands_list = [
            self._window_to_causal_bands(x_np[b, :, 0]) for b in range(batch_size)
        ]
        x_bands = torch.tensor(
            np.stack(bands_list, axis=0),
            dtype=x.dtype,
            device=x.device,
        ).permute(0, 2, 1)  # (batch, num_bands, seq_len)

        filtered_bands = [
            conv(x_bands[:, b : b + 1, :]) for b, conv in enumerate(self.band_convs)
        ]
        h = torch.cat(filtered_bands, dim=1)

        mixed = F.relu(self.band_mix(h))
        res = self.band_residual(x_bands)
        h_in = mixed + res

        out = self.tcn(h_in)
        last_step = out[:, :, -1]
        return F.softplus(self.fc(last_step)) + self.eps


class BandCausalTCNModel(BasePyTorchVolatilityModel):
    """Band Causal TCN model wrapper aligned with TCNModel and BasePyTorchVolatilityModel."""

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
            "num_levels": trial.suggest_int("num_levels", min_levels, max_levels),
            "channels": trial.suggest_categorical("channels", [32, 64, 128]),
            "kernel_size": kernel_size,
            "num_convs": num_convs,
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "J": trial.suggest_categorical("J", [1, 2, 3]),
            "wavelet": trial.suggest_categorical("wavelet", ["db4", "sym4"]),
            "band_kernel_size": trial.suggest_categorical("band_kernel_size", [1, 2]),
            "max_epochs": 200,
            "patience": 20,
            "retrain_freq": 1,
            "seed": 42,
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
        J: int = 2,
        wavelet: str = "sym4",
        decomp_window: int | None = None,
        band_kernel_size: int = 1,
        **kwargs,
    ) -> None:
        min_len = min_band_window_length(wavelet, J)
        effective_decomp = (
            decomp_window if decomp_window is not None else default_band_window(wavelet, J)
        )
        if effective_decomp < min_len:
            raise ValueError(
                f"[{self.__class__.__name__}] decomp_window={effective_decomp} < {min_len} "
                f"for wavelet='{wavelet}' and J={J}."
            )

        super().__init__(
            look_back=look_back,
            num_levels=num_levels,
            channels=channels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            J=J,
            wavelet=wavelet,
            decomp_window=effective_decomp,
            band_kernel_size=band_kernel_size,
            **kwargs,
        )

        self.net = BandCausalTCNNetwork(
            num_bands=J + 1,
            channels=channels,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_convs=num_convs,
            wavelet=wavelet,
            level=J,
            decomp_window=effective_decomp,
            band_kernel_size=band_kernel_size,
        ).to(self.device)

    @property
    def context_length(self) -> int:
        return self.params["look_back"]