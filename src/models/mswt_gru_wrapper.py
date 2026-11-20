"""
mswt_gru_mlp_wrapper
====================

Causal Multi-Scale Wavelet Transform GRU with an MLP head over per-band states.
"""

from __future__ import annotations

import copy

import numpy as np
import optuna
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = [
    "CausalBandWindowDataset",
    "MSWTGRUMLPModel",
    "MSWTGRUMLPNetwork",
    "compute_causal_bands",
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


def _step_causal_bands(prefix: np.ndarray, wavelet: str, J: int) -> np.ndarray:
    """Computes reconstructed band values at the final sample of `prefix`."""
    coeffs = pywt.wavedec(prefix, wavelet=wavelet, level=J, mode=_DWT_MODE)
    n = len(prefix)
    step_vals = np.empty(len(coeffs), dtype=np.float32)
    for k in range(len(coeffs)):
        isolated = [np.zeros_like(c) for c in coeffs]
        isolated[k] = coeffs[k]
        rec = pywt.waverec(isolated, wavelet=wavelet, mode=_DWT_MODE)
        step_vals[k] = rec[n - 1]
    return step_vals


def compute_causal_bands(
    series: np.ndarray,
    wavelet: str,
    J: int,
    decomp_window: int | None = None,
) -> np.ndarray:
    """Computes (len(series), J + 1) bands strictly causally using only index <= t."""
    arr = np.asarray(series, dtype=np.float64).flatten()
    n_samples = len(arr)
    min_len = min_window_length(wavelet, J)
    if n_samples < min_len:
        raise ValueError(
            f"Series length ({n_samples}) < min_window_length({wavelet}, {J})={min_len}."
        )

    eff_win = decomp_window if decomp_window is not None else default_decomp_window(wavelet, J)
    bands = np.zeros((n_samples, J + 1), dtype=np.float32)
    for t in range(n_samples):
        start = max(0, t + 1 - eff_win)
        window = arr[start : t + 1]
        if len(window) < min_len:
            window = np.pad(window, (min_len - len(window), 0), mode="edge")
        bands[t] = _step_causal_bands(window, wavelet=wavelet, J=J)
    return bands


compute_causal_dwt_bands = compute_causal_bands


class CausalBandWindowDataset(Dataset):
    """Builds (look_back, J + 1) lag windows ending at t to predict target[t + 1]."""

    def __init__(self, bands: np.ndarray, targets: np.ndarray, look_back: int) -> None:
        self.bands = torch.tensor(bands, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.look_back = look_back

    def __len__(self) -> int:
        return max(0, len(self.targets) - self.look_back)

    def __getitem__(self, idx: int):
        x_win = self.bands[idx : idx + self.look_back]
        y_tgt = self.targets[idx + self.look_back]
        return x_win, y_tgt


def _build_activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU()
    if act == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'. Use 'gelu', 'silu', or 'relu'.")


def _parse_mlp_units(units: str | list[int] | None) -> list[int]:
    if units is None:
        return [64]
    if isinstance(units, list):
        return [int(u) for u in units]
    if isinstance(units, str):
        return [int(part) for part in units.split("_") if part.strip()]
    raise ValueError(f"Unsupported mlp_hidden_units value: {units!r}")


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
    """Causal MS-WT-GRU with an MLP head compatible with WalkForwardTrainer."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        mlp_archs = [
            "32",
            "64",
            "128",
            "32_32",
            "32_64",
            "32_128",
            "64_32",
            "64_64",
            "64_128",
            "128_32",
            "128_64",
            "128_128",
        ]
        mlp_choice = trial.suggest_categorical("mlp_arch", mlp_archs)

        return {
            "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
            "wavelet": "db4",
            "J": trial.suggest_categorical("J", [1, 2, 3, 4]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "mlp_hidden_units": mlp_choice,
            "activation": trial.suggest_categorical("activation", ["gelu", "silu", "relu"]),
            "gru_dropout": trial.suggest_float("gru_dropout", 0.0, 0.3),
            "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "max_epochs": 200,
            "patience": 20,
            "retrain_freq": 1,
            "fine_tune_epochs": 10,
            "target_scale": 1e4,
            "seed": 42,
        }

    def __init__(
        self,
        look_back: int,
        hidden_size: int,
        num_layers: int,
        gru_dropout: float,
        mlp_dropout: float,
        J: int,
        learning_rate: float,
        weight_decay: float,
        wavelet: str = "db4",
        decomp_window: int | None = None,
        mlp_hidden_units: str | list[int] | None = None,
        activation: str | list[str] = "gelu",
        fine_tune_epochs: int | None = None,
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
        resolved_units = _parse_mlp_units(mlp_hidden_units)

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

        self.fine_tune_epochs = (
            fine_tune_epochs if fine_tune_epochs is not None else self.max_epochs
        )
        self._band_cache: dict[int, np.ndarray] = {}

    @property
    def context_length(self) -> int:
        return self.params["look_back"]

    def _causal_bands(self, series: np.ndarray) -> np.ndarray:
        return compute_causal_bands(
            series,
            wavelet=self.params["wavelet"],
            J=self.params["J"],
            decomp_window=self.params["decomp_window"],
        )

    def _cached_bands_for_prefix(self, arr: np.ndarray, offset: int = 0) -> np.ndarray:
        """Returns causal bands for `arr`, caching each step `offset + t`."""
        n_samples = len(arr)
        wavelet = self.params["wavelet"]
        J = self.params["J"]
        eff_win = self.params["decomp_window"]
        min_len = min_window_length(wavelet, J)
        if n_samples < min_len:
            raise ValueError(
                f"Series length ({n_samples}) < min_window_length({wavelet}, {J})={min_len}."
            )

        out = np.zeros((n_samples, J + 1), dtype=np.float32)
        for t in range(n_samples):
            global_t = offset + t
            if global_t in self._band_cache:
                out[t] = self._band_cache[global_t]
            else:
                start = max(0, t + 1 - eff_win)
                window = arr[start : t + 1]
                if len(window) < min_len:
                    window = np.pad(window, (min_len - len(window), 0), mode="edge")
                val = _step_causal_bands(window, wavelet=wavelet, J=J)
                self._band_cache[global_t] = val
                out[t] = val
        return out

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")

        self.history["train_loss"] = []
        self.history["val_loss"] = []
        self._band_cache.clear()

        split_idx = int(len(y_train) * 0.8)
        y_tr_raw = y_train[:split_idx]

        self.scaler.fit(y_tr_raw.reshape(-1, 1))

        y_all_scaled = self.scaler.transform(y_train.reshape(-1, 1)).flatten()
        self.y_train_history = y_all_scaled.copy()
        self.rs_train_history = rs_train.copy()

        bands_all = self._causal_bands(y_all_scaled)
        rs_all_scaled = rs_train * self.target_scale

        bands_tr = bands_all[:split_idx]
        rs_tr = rs_all_scaled[:split_idx]

        val_start_with_ctx = max(0, split_idx - self.context_length)
        bands_val = bands_all[val_start_with_ctx:]
        rs_val = rs_all_scaled[val_start_with_ctx:]

        train_dataset = CausalBandWindowDataset(bands_tr, rs_tr, self.context_length)
        val_dataset = CausalBandWindowDataset(bands_val, rs_val, self.context_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params.get("weight_decay", 0.0),
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_val_loss = float("inf")
        patience_counter = 0
        epochs_run = 0
        converged = False
        self.best_state_dict = None

        for _ in range(self.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.net(x_batch), y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("NaN/Inf loss detected")
                loss.backward()
                torch_utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= max(1, len(train_loader))

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    val_loss += self.criterion(self.net(x_batch), y_batch).item()
            val_loss /= max(1, len(val_loader))

            self.history["train_loss"].append(epoch_loss)
            self.history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
            epochs_run += 1

            if best_val_loss - val_loss > self.tolerance:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state_dict = copy.deepcopy(self.net.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    converged = True
                    break

        self.diagnostics["converged"] = converged
        self.diagnostics["epochs_run"] = epochs_run

        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)

        self.is_fitted = True
        self.is_valid = True

    def forecast(
        self,
        y_seq: np.ndarray,
        rs_seq: np.ndarray | None = None,
        save_dir: str | None = None,
        extra_metadata: dict | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        if rs_seq is None:
            raise ValueError("rs_seq must be provided.")

        self.net.eval()
        self.history.setdefault("ft_train_loss", [])
        self.history.setdefault("ft_val_loss", [])
        self._band_cache.clear()

        context_len = self.context_length
        predictions: list[float] = []

        history_y = list(self.y_train_history)
        history_rs = list(self.rs_train_history * self.target_scale)

        y_seq_scaled = self.scaler.transform(y_seq.reshape(-1, 1)).flatten()

        for i in range(len(y_seq)):
            arr_so_far = np.asarray(history_y, dtype=np.float64)
            bands_so_far = self._cached_bands_for_prefix(arr_so_far, offset=0)
            current_window = bands_so_far[-context_len:]

            x_tensor = (
                torch.tensor(current_window, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.net(x_tensor).item()
            predictions.append(pred)

            actual_y_scaled = float(y_seq_scaled[i])
            actual_rs = float(rs_seq[i] * self.target_scale)

            if debug:
                print(
                    f"[DEBUG] i={i:03d} | y_raw={y_seq[i]:+.6f} | "
                    f"y_scaled={actual_y_scaled:+.6f} | "
                    f"rs={actual_rs/self.target_scale:.6e} | "
                    f"pred={pred/self.target_scale:.6e}"
                )

            history_y.append(actual_y_scaled)
            history_rs.append(actual_rs)

            if self.retrain_freq > 0 and i % self.retrain_freq == 0 and self.fine_tune_epochs > 0:
                val_len = max(context_len + 1, min(30, len(history_y) - context_len - 1))
                if val_len > 0 and len(history_y) > val_len:
                    split_ft = len(history_y) - val_len
                    arr_all = np.asarray(history_y, dtype=np.float64)
                    rs_all = np.asarray(history_rs, dtype=np.float64)

                    # Compute causal bands over the full history before slicing chronologically.
                    bands_all = self._cached_bands_for_prefix(arr_all, offset=0)

                    bands_ft_train = bands_all[:split_ft]
                    rs_ft_train = rs_all[:split_ft]

                    val_start_ctx = max(0, split_ft - context_len)
                    bands_ft_val = bands_all[val_start_ctx:]
                    rs_ft_val = rs_all[val_start_ctx:]

                    ft_train_dataset = CausalBandWindowDataset(
                        bands_ft_train, rs_ft_train, context_len
                    )
                    ft_val_dataset = CausalBandWindowDataset(
                        bands_ft_val, rs_ft_val, context_len
                    )

                    if len(ft_train_dataset) > 0 and len(ft_val_dataset) > 0:
                        ft_train_loader = DataLoader(
                            ft_train_dataset, batch_size=self.batch_size, shuffle=False
                        )
                        ft_val_loader = DataLoader(
                            ft_val_dataset, batch_size=self.batch_size, shuffle=False
                        )

                        optimizer = optim.Adam(
                            self.net.parameters(),
                            lr=self.params["learning_rate"],
                            weight_decay=self.params.get("weight_decay", 0.0),
                        )
                        scheduler = ReduceLROnPlateau(
                            optimizer, mode="min", factor=0.5, patience=5
                        )

                        self.net.train()
                        best_ft_loss = float("inf")
                        patience_counter = 0
                        best_ft_state_dict = None

                        for _ in range(self.fine_tune_epochs):
                            self.net.train()
                            ft_epoch_loss = 0.0
                            for x_batch, y_batch in ft_train_loader:
                                x_batch = x_batch.to(self.device)
                                y_batch = y_batch.to(self.device)
                                optimizer.zero_grad()
                                loss = self.criterion(self.net(x_batch), y_batch)
                                if torch.isnan(loss) or torch.isinf(loss):
                                    raise RuntimeError("NaN/Inf loss during fine-tuning")
                                loss.backward()
                                torch_utils.clip_grad_norm_(
                                    self.net.parameters(), max_norm=1.0
                                )
                                optimizer.step()
                                ft_epoch_loss += loss.item()
                            ft_epoch_loss /= max(1, len(ft_train_loader))

                            self.net.eval()
                            ft_val_loss = 0.0
                            with torch.no_grad():
                                for x_batch, y_batch in ft_val_loader:
                                    x_batch = x_batch.to(self.device)
                                    y_batch = y_batch.to(self.device)
                                    ft_val_loss += self.criterion(
                                        self.net(x_batch), y_batch
                                    ).item()
                            ft_val_loss /= max(1, len(ft_val_loader))

                            self.history["ft_train_loss"].append(ft_epoch_loss)
                            self.history["ft_val_loss"].append(ft_val_loss)
                            scheduler.step(ft_val_loss)

                            if best_ft_loss - ft_val_loss > self.tolerance:
                                best_ft_loss = ft_val_loss
                                patience_counter = 0
                                best_ft_state_dict = copy.deepcopy(
                                    self.net.state_dict()
                                )
                            else:
                                patience_counter += 1
                                if patience_counter >= self.patience:
                                    break

                        if best_ft_state_dict is not None:
                            self.net.load_state_dict(best_ft_state_dict)
                        self.net.eval()

        predictions_arr = np.array(predictions) / self.target_scale

        if save_dir is not None:
            self._save_artifact(
                save_dir=save_dir,
                y_seq=y_seq,
                rs_seq=rs_seq,
                predictions=predictions_arr,
                extra_metadata=extra_metadata,
            )

        return predictions_arr