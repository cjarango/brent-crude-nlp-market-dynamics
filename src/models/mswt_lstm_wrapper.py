from __future__ import annotations

import copy

import numpy as np
import optuna
import torch
import torch.nn.utils as torch_utils
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.models.dl_base import BasePyTorchVolatilityModel
from src.models.original_model import (
    MSWTNetwork,
    _dwt_window,
    min_window_length,
    validate_even_look_back,
)

__all__ = ["MSWT_LSTMModel", "WaveletWindowDataset"]


class WaveletWindowDataset(Dataset):
    def __init__(self, bands: np.ndarray, rs: np.ndarray, look_back: int):
        self.bands = torch.tensor(bands, dtype=torch.float32)
        self.rs = torch.tensor(rs, dtype=torch.float32)
        self.look_back = look_back

    def __len__(self) -> int:
        return len(self.rs) - self.look_back

    def __getitem__(self, idx: int):
        return self.bands[idx], self.rs[idx + self.look_back]


def _precompute_bands(
    series: np.ndarray, look_back: int, wavelet: str, J: int
) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(series, look_back)
    return np.stack([_dwt_window(np.array(w), wavelet, J) for w in windows])


class MSWT_LSTMModel(BasePyTorchVolatilityModel):
    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        wavelet = "db4"
        look_back = trial.suggest_categorical("look_back", [14, 28, 56])

        max_J = 0
        for j in range(1, 6):
            if look_back >= min_window_length(wavelet, j):
                max_J = j
        if max_J < 1:
            raise optuna.TrialPruned(
                f"look_back={look_back} supports no valid J for wavelet='{wavelet}'."
            )
        J = trial.suggest_int("J", 1, max_J)

        return {
            "look_back": look_back,
            "wavelet": wavelet,
            "J": J,
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "gating_dim": trial.suggest_categorical("gating_dim", [32, 64, 128]),
            "gating_hidden_layers": trial.suggest_int("gating_hidden_layers", 2, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "max_epochs": 100,
            "patience": 15,
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
        dropout: float,
        J: int,
        wavelet: str,
        gating_dim: int,
        gating_hidden_layers: int,
        learning_rate: float,
        weight_decay: float,
        fine_tune_epochs: int | None = None,
        **kwargs,
    ) -> None:
        validate_even_look_back(look_back)
        required = min_window_length(wavelet, J)
        if look_back < required:
            raise ValueError(
                f"[{self.__class__.__name__}] look_back={look_back} < {required} "
                f"(minimum for wavelet='{wavelet}' with J={J})."
            )

        super().__init__(
            look_back=look_back,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            J=J,
            wavelet=wavelet,
            gating_dim=gating_dim,
            gating_hidden_layers=gating_hidden_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.net = MSWTNetwork(
            J=J,
            look_back=look_back,
            wavelet=wavelet,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            gating_dim=gating_dim,
            gating_hidden_layers=gating_hidden_layers,
            precomputed=True,
        ).to(self.device)

        self.fine_tune_epochs = (
            fine_tune_epochs if fine_tune_epochs is not None else self.max_epochs
        )

    @property
    def context_length(self) -> int:
        return self.params["look_back"]

    def _precompute(self, series: np.ndarray) -> np.ndarray:
        """Precomputa bandas DWT a partir de una serie **ya escalada**."""
        return _precompute_bands(
            series,
            self.context_length,
            self.params["wavelet"],
            self.params["J"],
        )

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")

        self.history["train_loss"] = []
        self.history["val_loss"] = []

        split_idx = int(len(y_train) * 0.8)
        y_tr_raw = y_train[:split_idx]
        y_val_raw = y_train[split_idx:]

        self.scaler.fit(y_tr_raw.reshape(-1, 1))

        y_tr = self.scaler.transform(y_tr_raw.reshape(-1, 1)).flatten()
        y_val = self.scaler.transform(y_val_raw.reshape(-1, 1)).flatten()

        self.y_train_history = self.scaler.transform(
            y_train.reshape(-1, 1)
        ).flatten()
        self.rs_train_history = rs_train.copy() 

        rs_tr = rs_train[:split_idx] * self.target_scale
        rs_val = rs_train[split_idx:] * self.target_scale

        bands_tr = self._precompute(y_tr)
        bands_val = self._precompute(y_val)

        train_dataset = WaveletWindowDataset(bands_tr, rs_tr, self.context_length)
        val_dataset = WaveletWindowDataset(bands_val, rs_val, self.context_length)

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

        for epoch in range(self.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.net(x_batch), y_batch)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("NaN/Inf loss detected")
                loss.backward()
                torch_utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    val_loss += self.criterion(self.net(x_batch), y_batch).item()
            val_loss /= len(val_loader)

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
        context_len = self.context_length
        predictions = []

        history_y = list(self.y_train_history)
        history_rs = list(self.rs_train_history * self.target_scale)

        y_seq_scaled = self.scaler.transform(y_seq.reshape(-1, 1)).flatten()

        full = np.concatenate([self.y_train_history, y_seq_scaled])
        full_bands = self._precompute(full)
        n_train = len(self.y_train_history)

        for i in range(len(y_seq)):
            idx = n_train + i - context_len
            x_tensor = (
                torch.tensor(full_bands[idx], dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.net(x_tensor).item()
            predictions.append(pred)

            actual_y_scaled = y_seq_scaled[i]
            actual_rs = rs_seq[i] * self.target_scale

            if debug:
                print(
                    f"[DEBUG] i={i:03d} | y_raw={y_seq[i]:+.6f} | "
                    f"y_scaled={actual_y_scaled:+.6f} | "
                    f"rs={actual_rs/self.target_scale:.6e} | "
                    f"pred={pred/self.target_scale:.6e}"
                )

            history_y.append(actual_y_scaled)
            history_rs.append(actual_rs)

            if i % self.retrain_freq == 0:
                val_len = max(context_len + 1, min(30, len(history_y) - context_len - 1))
                if val_len > 0 and len(history_y) > val_len:
                    y_ft_train = np.array(history_y[:-val_len])
                    rs_ft_train = np.array(history_rs[:-val_len])
                    y_ft_val = np.array(history_y[-val_len:])
                    rs_ft_val = np.array(history_rs[-val_len:])

                    # Bandas DWT sobre series ya escaladas
                    bands_ft_train = self._precompute(y_ft_train)
                    bands_ft_val = self._precompute(y_ft_val)

                    ft_train_dataset = WaveletWindowDataset(
                        bands_ft_train, rs_ft_train, context_len
                    )
                    ft_val_dataset = WaveletWindowDataset(
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
                        ft_converged = False
                        best_ft_state_dict = None

                        for epoch in range(self.fine_tune_epochs):
                            self.net.train()
                            ft_epoch_loss = 0.0
                            for x_batch, y_batch in ft_train_loader:
                                x_batch = x_batch.to(self.device)
                                y_batch = y_batch.to(self.device)
                                optimizer.zero_grad()
                                loss = self.criterion(self.net(x_batch), y_batch)
                                if torch.isnan(loss) or torch.isinf(loss):
                                    raise RuntimeError(
                                        "NaN/Inf loss during fine-tuning"
                                    )
                                loss.backward()
                                torch_utils.clip_grad_norm_(
                                    self.net.parameters(), max_norm=1.0
                                )
                                optimizer.step()
                                ft_epoch_loss += loss.item()
                            ft_epoch_loss /= len(ft_train_loader)

                            self.net.eval()
                            ft_val_loss = 0.0
                            with torch.no_grad():
                                for x_batch, y_batch in ft_val_loader:
                                    x_batch = x_batch.to(self.device)
                                    y_batch = y_batch.to(self.device)
                                    ft_val_loss += self.criterion(
                                        self.net(x_batch), y_batch
                                    ).item()
                            ft_val_loss /= len(ft_val_loader)

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
                                    ft_converged = True
                                    break

                        if best_ft_state_dict is not None:
                            self.net.load_state_dict(best_ft_state_dict)
                        self.net.eval()

        predictions = np.array(predictions) / self.target_scale

        if save_dir is not None:
            self._save_artifact(
                save_dir=save_dir,
                y_seq=y_seq,
                rs_seq=rs_seq,
                predictions=predictions,
                extra_metadata=extra_metadata,
            )

        return predictions