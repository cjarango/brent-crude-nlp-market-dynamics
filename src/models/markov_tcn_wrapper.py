"""
markov_tcn_wrapper
==================

Implementation of the hybrid Markov-Switching Temporal Convolutional Network
(Markov-TCN) for volatility forecasting with causal regime probability channels.
"""

from __future__ import annotations

import math

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from src.models.dl_base import BasePyTorchVolatilityModel

__version__ = "1.5.0"

__all__ = [
    "CausalConv1d",
    "MarkovTCNModel",
    "MarkovTCNNetwork",
    "TCNBlock",
    "compute_causal_markov_series",
    "markov_tcn_search_space",
]


def _fit_markov_with_retry(
    z_fit: np.ndarray,
    k_regimes: int,
    ar_order: int,
    search_reps: int = 20,
    max_attempts: int = 5,
    seed: int = 42,
):
    """Fits a Markov autoregression model on returns using a positive scale ladder.

    Args:
        z_fit: Standardized 1D return series used for parameter estimation.
        k_regimes: Number of hidden Markov regimes.
        ar_order: Autoregressive order of the switching process.
        search_reps: Number of initial random parameter search repetitions.
        max_attempts: Maximum number of scaling retries upon convergence failure.
        seed: Random seed for reproducible initialization.

    Returns:
        A tuple ``(result, scale)`` where ``result`` is the fitted statsmodels
        result (or ``None`` if all attempts fail) and ``scale`` is the positive
        multiplier applied during estimation.
    """
    scale_ladder = [1.0, 10.0, 100.0, 1_000.0, 10_000.0]
    rng_state = np.random.get_state()
    np.random.seed(seed)

    try:
        for attempt in range(max_attempts):
            scale = scale_ladder[min(attempt, len(scale_ladder) - 1)]
            z_scaled = np.ascontiguousarray(z_fit * scale, dtype=np.float64)
            reps = max(5, search_reps // (1 + attempt // 2))
            em_iters = 20 if attempt == 0 else (5 if attempt < 3 else 0)

            try:
                mod = MarkovAutoregression(
                    endog=z_scaled,
                    k_regimes=k_regimes,
                    order=ar_order,
                    switching_ar=False,
                    switching_variance=True,
                )
                res = mod.fit(
                    search_reps=reps,
                    search_iter=10,
                    em_iter=em_iters,
                    maxiter=400,
                    disp=False,
                )
                probs = np.asarray(res.filtered_marginal_probabilities, dtype=np.float64)
                if probs.ndim == 2 and probs.shape[1] == k_regimes and np.all(np.isfinite(probs)):
                    return res, scale
            except Exception:  # noqa: BLE001, S112
                continue
    finally:
        np.random.set_state(rng_state)

    return None, 1.0


def _causal_fallback_probs(z_all: np.ndarray, k_regimes: int) -> np.ndarray:
    """Computes deterministic causal logistic probabilities from squared returns.

    Args:
        z_all: Standardized 1D return series.
        k_regimes: Number of target regimes.

    Returns:
        Array of shape ``(len(z_all), k_regimes - 1)`` with values in ``[0, 1]``.
    """
    z_sq = np.clip(z_all**2, 0.0, 25.0)
    if k_regimes == 2:
        p1 = 1.0 / (1.0 + np.exp(-1.5 * (z_sq - 1.0)))
        return p1.reshape(-1, 1).astype(np.float32)
    p1 = 1.0 / (1.0 + np.exp(-1.5 * (z_sq - 0.75)))
    p2 = 1.0 / (1.0 + np.exp(-1.5 * (z_sq - 2.5)))
    p_high = np.minimum(p1, p2)
    p_mid = np.clip(p1 - p_high, 0.0, 1.0)
    return np.column_stack([p_mid, p_high]).astype(np.float32)


def compute_causal_markov_series(
    returns: np.ndarray,
    ma_window: int,
    ar_order: int,
    k_regimes: int,
    fit_end_idx: int | None = None,
    fitted_state: tuple | None = None,
    search_reps: int = 20,
    max_attempts: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, tuple]:
    """Estimates or applies a causal Markov-switching filter on log returns.

    When ``fitted_state`` is ``None``, parameters are estimated strictly on
    ``returns[:fit_end_idx]`` and regimes are ordered by ascending conditional
    variance. When ``fitted_state`` is provided, the stored parameters are
    applied via forward causal filtering without re-estimation.

    Args:
        returns: Input 1D return series.
        ma_window: Causal trailing moving-average window length applied to returns.
        ar_order: Autoregressive order of the Markov model.
        k_regimes: Number of hidden Markov regimes.
        fit_end_idx: Exclusive end index for parameter estimation.
        fitted_state: Pre-estimated tuple ``(params, order_idx, (mu, sd), scale)``.
        search_reps: Initial parameter search repetitions for MLE.
        max_attempts: Maximum scaling retry attempts.
        seed: Random seed for estimation reproducibility.

    Returns:
        A tuple ``(probs, fitted_state)`` where ``probs`` has shape
        ``(len(returns), k_regimes - 1)`` containing causal filtered probabilities
        for regimes ``1 .. k_regimes - 1``.
    """
    arr = np.asarray(returns, dtype=np.float64).flatten()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if k_regimes < 2:
        return np.empty((len(arr), 0), dtype=np.float32), (None, None, (0.0, 1.0), 1.0)

    r_ma = (
        pd.Series(arr, dtype="float64")
        .rolling(max(1, int(ma_window)), min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )

    if fitted_state is None:
        end = len(r_ma) if fit_end_idx is None else int(fit_end_idx)
        end = max(10, min(end, len(r_ma)))
        r_fit = r_ma[:end]
        mu = float(np.mean(r_fit))
        sd = float(np.std(r_fit))
        sd = sd if sd > 1e-12 else 1.0
        z_fit = (r_fit - mu) / sd

        res_fit, used_scale = _fit_markov_with_retry(
            z_fit=z_fit,
            k_regimes=k_regimes,
            ar_order=ar_order,
            search_reps=search_reps,
            max_attempts=max_attempts,
            seed=seed,
        )

        if res_fit is None:
            fitted_state = (None, None, (mu, sd), 1.0)
            z_all = (r_ma - mu) / sd
            return _causal_fallback_probs(z_all, k_regimes), fitted_state

        params = np.asarray(res_fit.params, dtype=np.float64)
        p_fit_raw = np.asarray(res_fit.filtered_marginal_probabilities, dtype=np.float64)
        offset_fit = len(z_fit) - len(p_fit_raw)
        states_fit = np.argmax(p_fit_raw, axis=1)
        ref_fit = arr[offset_fit:end]
        variances = [
            float(np.var(ref_fit[states_fit == k])) if np.any(states_fit == k) else -np.inf
            for k in range(k_regimes)
        ]
        order_idx = np.argsort(variances)
        fitted_state = (params, order_idx, (mu, sd), used_scale)
    else:
        params, order_idx, (mu, sd), used_scale = fitted_state

    z_base = (r_ma - mu) / sd
    if params is None:
        return _causal_fallback_probs(z_base, k_regimes), fitted_state

    z_all = np.ascontiguousarray(z_base * used_scale, dtype=np.float64)
    try:
        mod_all = MarkovAutoregression(
            endog=z_all,
            k_regimes=k_regimes,
            order=ar_order,
            switching_ar=False,
            switching_variance=True,
        )
        res_all = mod_all.filter(params)
        p_raw = np.asarray(res_all.filtered_marginal_probabilities, dtype=np.float64)
        offset = len(z_all) - len(p_raw)

        p_sorted = p_raw[:, order_idx]
        if offset > 0:
            p_full = np.vstack([np.repeat(p_sorted[0:1], offset, axis=0), p_sorted])
        else:
            p_full = p_sorted

        if not np.all(np.isfinite(p_full)):
            return _causal_fallback_probs(z_base, k_regimes), fitted_state

        p_full = np.clip(p_full, 0.0, 1.0)
        return p_full[:, 1:].astype(np.float32), fitted_state
    except Exception:  # noqa: BLE001
        return _causal_fallback_probs(z_base, k_regimes), fitted_state


class CausalConv1d(nn.Module):
    """1D convolution with causal padding (left-only)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        """Initializes the causal 1D convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            dilation: Spacing between kernel elements.
        """
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies left-padded causal convolution.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
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
        """Initializes the TCN residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolutional kernel size.
            dilation: Dilation factor for the causal convolutions.
            dropout: Dropout probability applied after each activation.
            num_convs: Number of stacked causal convolutions inside the block.
        """
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
        """Applies residual dilated causal convolutions.

        Args:
            x: Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        out = x
        for conv, relu, dropout in zip(self.convs, self.relus, self.dropouts):
            out = dropout(relu(conv(out)))

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class MarkovTCNNetwork(nn.Module):
    """Multi-channel TCN with a softplus head that guarantees positive variance output."""

    def __init__(self, num_channels: list[int], kernel_size: int, dropout: float, num_convs: int) -> None:
        """Initializes the multi-channel TCN network.

        Args:
            num_channels: Channel progression list ``[in_channels, C, ..., C]``.
            kernel_size: Convolutional kernel size across all blocks.
            dropout: Dropout probability in residual blocks.
            num_convs: Number of causal convolutions per residual block.
        """
        super().__init__()
        layers = []
        for i in range(len(num_channels) - 1):
            layers.append(
                TCNBlock(
                    in_channels=num_channels[i],
                    out_channels=num_channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                    num_convs=num_convs,
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts one-step-ahead positive variance from multi-channel return windows.

        Args:
            x: Input tensor of shape ``(batch, seq_len, in_channels)``.

        Returns:
            Positive variance forecast tensor of shape ``(batch, 1)``.
        """
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]
        return F.softplus(self.fc(out)) + self.eps


class MarkovTCNModel(BasePyTorchVolatilityModel):
    """Markov-Switching TCN model wrapper aligned with TCNModel and BasePyTorchVolatilityModel."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        """Defines the Optuna hyperparameter search space.

        Args:
            trial: Active Optuna trial instance.

        Returns:
            Dictionary of sampled architecture, optimization, and regime parameters.
        """
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
            "ma_window": trial.suggest_categorical("ma_window", [5, 10, 22]),
            "ar_order": trial.suggest_int("ar_order", 1, 2),
            "k_regimes": trial.suggest_categorical("k_regimes", [2, 3]),
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
        ma_window: int = 22,
        ar_order: int = 2,
        k_regimes: int = 2,
        **kwargs,
    ) -> None:
        """Initializes the Markov-TCN model wrapper.

        Args:
            look_back: Number of historical steps per input window.
            num_levels: Number of dilated residual blocks.
            channels: Hidden channel width in each TCN block.
            kernel_size: Convolutional kernel size.
            num_convs: Number of convolutions per residual block.
            dropout: Dropout probability.
            learning_rate: Initial learning rate for Adam.
            weight_decay: L2 regularization penalty.
            ma_window: Causal moving-average window for regime estimation.
            ar_order: Autoregressive order of the Markov model.
            k_regimes: Number of hidden Markov regimes.
            **kwargs: Additional keyword arguments forwarded to the base class.
        """
        in_channels = 1 + max(0, k_regimes - 1)
        num_channels = [in_channels] + [channels] * num_levels

        super().__init__(
            look_back=look_back,
            num_levels=num_levels,
            channels=channels,
            kernel_size=kernel_size,
            num_convs=num_convs,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            ma_window=ma_window,
            ar_order=ar_order,
            k_regimes=k_regimes,
            **kwargs,
        )

        self.net = MarkovTCNNetwork(
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            num_convs=num_convs,
        ).to(self.device)
        self._ms_state: tuple | None = None

    @property
    def context_length(self) -> int:
        """Returns the required input sequence length."""
        return self.params["look_back"]

    def _augment_series(
        self, y_scaled: np.ndarray, fit_end_idx: int | None = None
    ) -> np.ndarray:
        """Computes causal Markov regime probabilities on the full continuous return series
        before windowing and batching.

        Args:
            y_scaled: Standardized 1D return series of shape ``(N,)``.
            fit_end_idx: Exclusive end index for parameter estimation. When provided,
                refreshes Markov parameters strictly on ``y_scaled[:fit_end_idx]``;
                when ``None``, filters forward using the current ``self._ms_state``.

        Returns:
            2D float32 array of shape ``(N, 1 + max(0, k_regimes - 1))``.
        """
        arr = np.asarray(y_scaled, dtype=np.float64).flatten()
        if self.params["k_regimes"] < 2:
            return arr.reshape(-1, 1).astype(np.float32)

        if fit_end_idx is not None or self._ms_state is None:
            reg_probs, self._ms_state = compute_causal_markov_series(
                returns=arr,
                ma_window=self.params["ma_window"],
                ar_order=self.params["ar_order"],
                k_regimes=self.params["k_regimes"],
                fit_end_idx=fit_end_idx if fit_end_idx is not None else len(arr),
                fitted_state=None,
                max_attempts=5,
                seed=self.seed,
            )
            self.diagnostics["markov_converged"] = self._ms_state[0] is not None
            self.diagnostics["markov_scale"] = float(self._ms_state[3])
        else:
            reg_probs, _ = compute_causal_markov_series(
                returns=arr,
                ma_window=self.params["ma_window"],
                ar_order=self.params["ar_order"],
                k_regimes=self.params["k_regimes"],
                fitted_state=self._ms_state,
                max_attempts=5,
                seed=self.seed,
            )

        return np.concatenate([arr.reshape(-1, 1), reg_probs], axis=1).astype(np.float32)


def markov_tcn_search_space(trial: optuna.Trial) -> dict:
    """Returns the hyperparameter search space for MarkovTCNModel.

    Args:
        trial: Active Optuna trial instance.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    return MarkovTCNModel.suggest_params(trial)