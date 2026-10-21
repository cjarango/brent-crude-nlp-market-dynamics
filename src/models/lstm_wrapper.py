"""
lstm_wrapper
============

Implementation of the LSTM model wrapper.

This module provides a standardized wrapper for a Long Short-Term Memory
(LSTM) network that forecasts one-step-ahead conditional variance using only
past log returns. The network outputs the log-variance and is trained by
minimizing the Gaussian quasi-likelihood (equivalent to QLIKE on r_t^2).

Dependencies
------------
numpy, torch
"""

__version__ = "1.0.0"

import copy
from typing import Any

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from src.models.base import BaseVolatilityModel
from torch import nn

__all__ = ["LSTMWrapper"]


class _LSTMNet(nn.Module):
    """LSTM that maps a window of features to a single log-variance."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _gaussian_nll(logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian quasi-likelihood: 0.5 * mean(log s2 + r2 / s2)."""
    logvar = torch.clamp(logvar, -12.0, 12.0)
    return 0.5 * torch.mean(logvar + target * torch.exp(-logvar))


class LSTMWrapper(BaseVolatilityModel):
    """Wrapper for an LSTM one-step-ahead variance forecaster."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initializes the LSTMWrapper with a dictionary of hyperparameters.

        Args:
            params (Dict[str, Any]): Keys (all optional, with defaults):
                'window' (int, 10), 'hidden_size' (int, 32), 'num_layers' (int, 1),
                'dropout' (float, 0.0), 'lr' (float, 1e-3), 'weight_decay' (float, 0.0),
                'batch_size' (int, 64), 'max_epochs' (int, 100), 'patience' (int, 10),
                'val_fraction' (float, 0.15), 'seed' (int, 42).

        Raises:
            ValueError: If any hyperparameter is out of its valid range.
        """
        super().__init__(**params)

        self.window = params.get("window", 10)
        self.hidden_size = params.get("hidden_size", 32)
        self.num_layers = params.get("num_layers", 1)
        self.dropout = params.get("dropout", 0.0)
        self.lr = params.get("lr", 1e-3)
        self.weight_decay = params.get("weight_decay", 0.0)
        self.batch_size = params.get("batch_size", 64)
        self.max_epochs = params.get("max_epochs", 300)
        self.patience = params.get("patience", 30)
        self.val_fraction = params.get("val_fraction", 0.15)
        self.seed = params.get("seed", 42)
        
        requested = params.get("device", "auto")
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)

        for name in ("window", "hidden_size", "num_layers", "batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Parameter '{name}' must be an integer >= 1. Got {value}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"Parameter 'dropout' must be in [0, 1). Got {self.dropout}.")
        if self.lr <= 0:
            raise ValueError(f"Parameter 'lr' must be > 0. Got {self.lr}.")
        if not 0.0 < self.val_fraction < 0.5:
            raise ValueError(f"Parameter 'val_fraction' must be in (0, 0.5). Got {self.val_fraction}.")

        self.net: _LSTMNet | None = None
        self.scale: float = 1.0
        self.y_train: np.ndarray = np.array([])

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_epoch: int = 0
        self.n_parameters: int = 0

        self.converged: bool = False
        self.is_valid: bool = False
        self.is_fitted: bool = False

    @property
    def context_length(self) -> int:
        """Returns the number of past observations needed as context."""
        return self.window

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _windows(self, r_scaled: np.ndarray) -> np.ndarray:
        """Builds the (N, window, 2) tensor of [r, |r|] rolling windows."""
        feats = np.stack([r_scaled, np.abs(r_scaled)], axis=1).astype(np.float32)
        wins = sliding_window_view(feats, self.window, axis=0)  # (n-w+1, 2, w)
        return np.ascontiguousarray(wins.transpose(0, 2, 1))

    def _predict_logvar(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            chunks = [
                self.net(X[i:i + 4096].to(self.device)).cpu()
                for i in range(0, len(X), 4096)
            ]
        return torch.cat(chunks)

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    def fit(self, y_train: np.ndarray) -> None:
        """Trains the LSTM on log returns with chronological early stopping.

        The last `val_fraction` of the training windows is held out (in time
        order) exclusively for early stopping. The walk-forward validation
        block is never seen here.

        Args:
            y_train (np.ndarray): Array of training log returns.
        """
        self.y_train = np.asarray(y_train, dtype="float64")
        n = len(self.y_train)
        if n <= self.window + 10:
            raise ValueError("Training sample too short for the requested window.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.scale = float(np.std(self.y_train))
        r = self.y_train / self.scale

        X = torch.from_numpy(self._windows(r)[:-1])                 # (n-w, w, 2)
        t = torch.from_numpy((r[self.window:] ** 2).astype(np.float32))  # (n-w,)

        n_val = max(1, int(len(X) * self.val_fraction))
        X_tr, t_tr = X[:-n_val].to(self.device), t[:-n_val].to(self.device)
        X_va, t_va = X[-n_val:], t[-n_val:]          # validación se queda en CPU

        self.net = _LSTMNet(2, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        self.n_parameters = sum(p.numel() for p in self.net.parameters())
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train_losses, self.val_losses = [], []
        best_val, best_state, best_epoch, wait = np.inf, None, 0, 0

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            perm = torch.randperm(len(X_tr), generator=gen).to(self.device)
            batch_losses = []
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                loss = _gaussian_nll(self.net(X_tr[idx]), t_tr[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                batch_losses.append(loss.detach())

            train_loss = float(torch.stack(batch_losses).mean())
            val_loss = float(_gaussian_nll(self._predict_logvar(X_va), t_va))
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
                break

            if val_loss < best_val - 1e-6:
                best_val, best_epoch, wait = val_loss, epoch, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.converged = best_state is not None and bool(np.isfinite(best_val))
        if self.converged:
            self.net.load_state_dict(best_state)
            self.best_epoch = best_epoch

        self.is_valid = self.converged
        self.is_fitted = True

    def forecast(self, y_target: np.ndarray, initial_context: np.ndarray = None) -> np.ndarray:
        """Generates one-step-ahead variance forecasts.

        Args:
            y_target (np.ndarray): The new data to forecast.
            initial_context (np.ndarray, optional): The last context_length returns from
                the training set. Defaults to None.

        Returns:
            np.ndarray: Array of predicted variances.

        Raises:
            RuntimeError: If the model is not fitted or invalid.
            ValueError: If initial_context or y_target are too short.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling forecast.")

        if not self.is_valid:
            raise RuntimeError("Cannot forecast with an invalid model.")

        if initial_context is not None:
            if len(initial_context) != self.context_length:
                raise ValueError(f"initial_context must have exactly {self.context_length} elements.")
            full_history = np.concatenate([initial_context, y_target])
        else:
            if len(y_target) <= self.context_length:
                raise ValueError(f"Input sequence must be longer than {self.context_length}.")
            full_history = np.asarray(y_target, dtype="float64")

        r = full_history / self.scale
        X = torch.from_numpy(self._windows(r)[:-1])
        logvar = torch.clamp(self._predict_logvar(X), -12.0, 12.0)
        variance = torch.exp(logvar).numpy().astype("float64") * self.scale ** 2

        return variance

    def get_diagnostics(self) -> dict:
        """Returns diagnostic data from the fitted model.

        'conditional_volatility' is aligned with y_train (the first `window`
        observations, which have no full context, are filled with the sample std).
        'std_resid' starts at observation `window` (no padding).
        """
        if not self.is_fitted:
            return {}

        if not self.converged:
            return {
                "converged": False,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }

        var = self.forecast(self.y_train)
        vol = np.sqrt(var)
        pad = np.full(self.window, self.scale)

        return {
            "conditional_volatility": np.concatenate([pad, vol]),
            "residuals": self.y_train[self.window:],
            "std_resid": self.y_train[self.window:] / vol,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "n_parameters": self.n_parameters,
            "converged": self.converged,
        }