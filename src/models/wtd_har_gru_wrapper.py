"""
wtd_har_gru_wrapper
===================

Causal Wavelet Transform Denoising (WTD) + Heterogeneous Autoregressive (HAR)
via QR decomposition + GRU network for one-step-ahead volatility forecasting.
"""

from __future__ import annotations

import copy
import gc

import numpy as np
import optuna
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.models.dl_base import BasePyTorchVolatilityModel

__all__ = [
    "CausalWTDHARWindowDataset",
    "WTDHARGRUModel",
    "WTDHARGRUNetwork",
    "compute_causal_wtd_series",
    "default_wtd_window",
    "min_wtd_window_length",
]

_DWT_MODE = "symmetric"


def min_wtd_window_length(wavelet: str, level: int) -> int:
    filter_len = pywt.Wavelet(wavelet).dec_len
    return (filter_len - 1) * (2 ** level)


def default_wtd_window(wavelet: str, level: int) -> int:
    min_len = min_wtd_window_length(wavelet, level)
    return max(min_len, 2 * min_len)


def _step_causal_wtd(
    prefix: np.ndarray,
    wavelet: str,
    level: int,
    threshold_mode: str = "soft",
) -> float:
    """Computes strictly causal wavelet-denoised value at the final sample of `prefix`."""
    coeffs = pywt.wavedec(prefix, wavelet=wavelet, level=level, mode=_DWT_MODE)
    n = len(prefix)

    finest_detail = coeffs[-1]
    sigma = float(np.median(np.abs(finest_detail))) / 0.6745 if len(finest_detail) > 0 else 0.0
    if sigma > 0.0 and len(prefix) > 1:
        thresh = sigma * np.sqrt(2.0 * np.log(len(prefix)))
        denoised_coeffs = [coeffs[0]] + [
            pywt.threshold(c, value=thresh, mode=threshold_mode) for c in coeffs[1:]
        ]
    else:
        denoised_coeffs = coeffs

    rec = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode=_DWT_MODE)
    return float(rec[n - 1])


def compute_causal_wtd_series(
    series: np.ndarray,
    wavelet: str = "db4",
    level: int = 2,
    decomp_window: int | None = None,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """Computes a causal wavelet-denoised 1D series using only indices <= t."""
    arr = np.asarray(series, dtype=np.float64).flatten()
    n_samples = len(arr)
    min_len = min_wtd_window_length(wavelet, level)
    if n_samples < min_len:
        raise ValueError(
            f"Series length ({n_samples}) < min_wtd_window_length({wavelet}, {level})={min_len}."
        )

    eff_win = decomp_window if decomp_window is not None else default_wtd_window(wavelet, level)
    out = np.zeros(n_samples, dtype=np.float64)
    for t in range(n_samples):
        start = max(0, t + 1 - eff_win)
        window = arr[start : t + 1]
        if len(window) < min_len:
            window = np.pad(window, (min_len - len(window), 0), mode="edge")
        out[t] = _step_causal_wtd(
            window,
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
        )
    return out


class CausalWTDHARWindowDataset(Dataset):
    """Builds (look_back, num_features) windows ending at t to predict targets[t + 1]."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, look_back: int) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.look_back = look_back

    def __len__(self) -> int:
        return max(0, len(self.targets) - self.look_back)

    def __getitem__(self, idx: int):
        x_win = self.features[idx : idx + self.look_back]
        y_tgt = self.targets[idx + self.look_back]
        return x_win, y_tgt


class WTDHARGRUNetwork(nn.Module):
    """GRU with a softplus head that guarantees positive variance output."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return F.softplus(self.fc(out[:, -1, :])) + self.eps


class WTDHARGRUModel(BasePyTorchVolatilityModel):
    """Causal WTD-HAR-GRU hybrid with QR HAR estimation and one-step-ahead variance output."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
            "lag_weekly": trial.suggest_categorical("lag_weekly", [5, 7]),
            "lag_monthly": trial.suggest_categorical("lag_monthly", [22, 30]),
            "wavelet": "db4",
            "wtd_level": trial.suggest_categorical("wtd_level", [1, 2, 3]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
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
        lag_weekly: int = 5,
        lag_monthly: int = 22,
        wavelet: str = "db4",
        wtd_level: int = 2,
        decomp_window: int | None = None,
        threshold_mode: str = "soft",
        fine_tune_epochs: int | None = None,
        **kwargs,
    ) -> None:
        min_len = min_wtd_window_length(wavelet, wtd_level)
        effective_decomp = (
            decomp_window if decomp_window is not None else default_wtd_window(wavelet, wtd_level)
        )
        if effective_decomp < min_len:
            raise ValueError(
                f"[{self.__class__.__name__}] decomp_window={effective_decomp} < {min_len} "
                f"for wavelet='{wavelet}' and wtd_level={wtd_level}."
            )

        effective_dropout = dropout if num_layers > 1 else 0.0

        super().__init__(
            look_back=look_back,
            lag_weekly=lag_weekly,
            lag_monthly=lag_monthly,
            wavelet=wavelet,
            wtd_level=wtd_level,
            decomp_window=effective_decomp,
            threshold_mode=threshold_mode,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

        # Features per step t: [y_scaled_t, wtd_rv_t, har_pred_t, prior_har_resid_t_minus_1]
        self.net = WTDHARGRUNetwork(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
        ).to(self.device)

        self.fine_tune_epochs = (
            fine_tune_epochs if fine_tune_epochs is not None else self.max_epochs
        )
        self.feature_scaler = StandardScaler()
        self._feature_scaler_fitted: bool = False
        self.beta_qr: np.ndarray | None = None
        self.y_train_raw_history: np.ndarray = np.array([], dtype=np.float64)
        self._wtd_cache: dict[int, float] = {}

    @property
    def context_length(self) -> int:
        return self.params["look_back"]

    def _cached_wtd_for_prefix(self, arr: np.ndarray, offset: int = 0) -> np.ndarray:
        """Computes causal WTD values for `arr`, caching each step `offset + t`."""
        n_samples = len(arr)
        wavelet = self.params["wavelet"]
        level = self.params["wtd_level"]
        eff_win = self.params["decomp_window"]
        thresh_mode = self.params["threshold_mode"]
        min_len = min_wtd_window_length(wavelet, level)
        if n_samples < min_len:
            raise ValueError(
                f"Series length ({n_samples}) < min_wtd_window_length({wavelet}, {level})={min_len}."
            )

        out = np.zeros(n_samples, dtype=np.float64)
        for t in range(n_samples):
            global_t = offset + t
            if global_t in self._wtd_cache:
                out[t] = self._wtd_cache[global_t]
            else:
                start = max(0, t + 1 - eff_win)
                window = arr[start : t + 1]
                if len(window) < min_len:
                    window = np.pad(window, (min_len - len(window), 0), mode="edge")
                val = _step_causal_wtd(
                    window,
                    wavelet=wavelet,
                    level=level,
                    threshold_mode=thresh_mode,
                )
                self._wtd_cache[global_t] = val
                out[t] = val
        return out

    def _build_har_design(self, wtd_rv_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Builds causal HAR lag matrix X [1, RV_d, RV_w, RV_m] aligned from `lag_monthly`."""
        lag_w = self.params["lag_weekly"]
        lag_m = self.params["lag_monthly"]
        s = pd.Series(np.asarray(wtd_rv_scaled, dtype=np.float64))

        rv_d = s.shift(1)
        rv_w = s.shift(1).rolling(window=lag_w).mean()
        rv_m = s.shift(1).rolling(window=lag_m).mean()

        df = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}).iloc[lag_m:]
        x_mat = np.column_stack(
            [
                np.ones(len(df), dtype=np.float64),
                df["rv_d"].to_numpy(dtype=np.float64),
                df["rv_w"].to_numpy(dtype=np.float64),
                df["rv_m"].to_numpy(dtype=np.float64),
            ]
        )
        valid_indices = df.index.to_numpy(dtype=int)
        return x_mat, valid_indices

    @staticmethod
    def _fit_qr(x_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """Solves linear least squares via reduced QR decomposition with lstsq fallback."""
        if len(x_mat) == 0:
            return np.zeros(4, dtype=np.float64)
        q_mat, r_mat = np.linalg.qr(x_mat, mode="reduced")
        if np.abs(np.diag(r_mat)).min() < 1e-10:
            beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
            return beta
        return np.linalg.solve(r_mat, q_mat.T @ y_vec)

    def _build_causal_feature_matrix(
        self,
        y_raw: np.ndarray,
        y_scaled: np.ndarray,
        rs_scaled: np.ndarray,
        beta_qr: np.ndarray,
        scale_gru_inputs: bool = True,
    ) -> np.ndarray:
        """
        Constructs strictly causal (N, 4) features at each index t for predicting t + 1.

        Standardizes only the GRU input channels using `self.feature_scaler` when fitted;
        target variance remains in `rs_scaled` (`rs * target_scale`).
        """
        lag_m = self.params["lag_monthly"]
        n_samples = len(y_raw)

        rv_raw_scaled = (np.asarray(y_raw, dtype=np.float64) ** 2) * self.target_scale
        wtd_rv_scaled = self._cached_wtd_for_prefix(rv_raw_scaled, offset=0)

        har_pred = np.zeros(n_samples, dtype=np.float64)
        prior_resid = np.zeros(n_samples, dtype=np.float64)

        if n_samples > lag_m:
            x_mat, valid_idx = self._build_har_design(wtd_rv_scaled)
            preds_valid = x_mat @ beta_qr
            har_pred[valid_idx] = preds_valid

            resids_valid = np.asarray(rs_scaled[valid_idx], dtype=np.float64) - preds_valid
            for k, idx_t in enumerate(valid_idx):
                if idx_t + 1 < n_samples:
                    prior_resid[idx_t + 1] = resids_valid[k]

        raw_feats = np.column_stack(
            [
                np.asarray(y_scaled, dtype=np.float64),
                wtd_rv_scaled,
                har_pred,
                prior_resid,
            ]
        )

        if scale_gru_inputs and self._feature_scaler_fitted:
            return self.feature_scaler.transform(raw_feats)
        return raw_feats

    def causal_features(
        self,
        y_raw: np.ndarray,
        rs_raw: np.ndarray,
        scale_gru_inputs: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns GRU-ready causal features and variance targets scaled by `target_scale`."""
        y_arr = np.asarray(y_raw, dtype=np.float64)
        rs_scaled = np.asarray(rs_raw, dtype=np.float64) * self.target_scale
        y_scaled = self.scaler.transform(y_arr.reshape(-1, 1)).flatten()
        feats = self._build_causal_feature_matrix(
            y_raw=y_arr,
            y_scaled=y_scaled,
            rs_scaled=rs_scaled,
            beta_qr=self.beta_qr,
            scale_gru_inputs=scale_gru_inputs,
        )
        return feats, rs_scaled

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")

        self.history["train_loss"] = []
        self.history["val_loss"] = []
        self._wtd_cache.clear()
        self._feature_scaler_fitted = False

        split_idx = int(len(y_train) * 0.8)
        y_tr_raw = np.asarray(y_train[:split_idx], dtype=np.float64)
        rs_tr_raw = np.asarray(rs_train[:split_idx], dtype=np.float64)

        self.scaler.fit(y_tr_raw.reshape(-1, 1))
        y_all_scaled = self.scaler.transform(np.asarray(y_train, dtype=np.float64).reshape(-1, 1)).flatten()

        rv_tr_scaled = (y_tr_raw ** 2) * self.target_scale
        wtd_tr_scaled = self._cached_wtd_for_prefix(rv_tr_scaled, offset=0)
        lag_m = self.params["lag_monthly"]
        if len(rv_tr_scaled) > lag_m:
            x_tr, valid_tr = self._build_har_design(wtd_tr_scaled)
            y_target_tr = rs_tr_raw[valid_tr] * self.target_scale
            self.beta_qr = self._fit_qr(x_tr, y_target_tr)
        else:
            self.beta_qr = np.zeros(4, dtype=np.float64)

        self.y_train_raw_history = np.asarray(y_train, dtype=np.float64).copy()
        self.y_train_history = y_all_scaled.copy()
        self.rs_train_history = np.asarray(rs_train, dtype=np.float64).copy()

        rs_all_scaled = np.asarray(rs_train, dtype=np.float64) * self.target_scale
        unscaled_feats_all = self._build_causal_feature_matrix(
            y_raw=np.asarray(y_train, dtype=np.float64),
            y_scaled=y_all_scaled,
            rs_scaled=rs_all_scaled,
            beta_qr=self.beta_qr,
            scale_gru_inputs=False,
        )

        # Fit GRU input scaler strictly on the training split.
        self.feature_scaler.fit(unscaled_feats_all[:split_idx])
        self._feature_scaler_fitted = True
        feat_all = self.feature_scaler.transform(unscaled_feats_all)

        feat_tr = feat_all[:split_idx]
        rs_tr = rs_all_scaled[:split_idx]

        val_start_with_ctx = max(0, split_idx - self.context_length)
        feat_val = feat_all[val_start_with_ctx:]
        rs_val = rs_all_scaled[val_start_with_ctx:]

        train_dataset = CausalWTDHARWindowDataset(feat_tr, rs_tr, self.context_length)
        val_dataset = CausalWTDHARWindowDataset(feat_val, rs_val, self.context_length)

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

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        self._wtd_cache.clear()

        context_len = self.context_length
        predictions: list[float] = []

        history_y_raw = list(self.y_train_raw_history)
        history_y_scaled = list(self.y_train_history)
        history_rs = list(np.asarray(self.rs_train_history, dtype=np.float64) * self.target_scale)

        y_seq_scaled = self.scaler.transform(np.asarray(y_seq, dtype=np.float64).reshape(-1, 1)).flatten()

        for i in range(len(y_seq)):
            y_raw_arr = np.asarray(history_y_raw, dtype=np.float64)
            y_scaled_arr = np.asarray(history_y_scaled, dtype=np.float64)
            rs_arr = np.asarray(history_rs, dtype=np.float64)

            feat_so_far = self._build_causal_feature_matrix(
                y_raw=y_raw_arr,
                y_scaled=y_scaled_arr,
                rs_scaled=rs_arr,
                beta_qr=self.beta_qr,
                scale_gru_inputs=True,
            )
            current_window = feat_so_far[-context_len:]

            x_tensor = (
                torch.tensor(current_window, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.net(x_tensor).item()
            predictions.append(pred)

            actual_y_raw = float(y_seq[i])
            actual_y_scaled = float(y_seq_scaled[i])
            actual_rs = float(rs_seq[i]) * self.target_scale

            if debug:
                print(
                    f"[DEBUG] i={i:03d} | y_raw={actual_y_raw:+.6f} | "
                    f"y_scaled={actual_y_scaled:+.6f} | "
                    f"rs={actual_rs/self.target_scale:.6e} | "
                    f"pred={pred/self.target_scale:.6e}"
                )

            history_y_raw.append(actual_y_raw)
            history_y_scaled.append(actual_y_scaled)
            history_rs.append(actual_rs)

            if self.retrain_freq > 0 and i % self.retrain_freq == 0 and self.fine_tune_epochs > 0:
                val_len = max(context_len + 1, min(30, len(history_y_scaled) - context_len - 1))
                if val_len > 0 and len(history_y_scaled) > val_len:
                    split_ft = len(history_y_scaled) - val_len
                    y_raw_all = np.asarray(history_y_raw, dtype=np.float64)
                    y_scaled_all = np.asarray(history_y_scaled, dtype=np.float64)
                    rs_all = np.asarray(history_rs, dtype=np.float64)

                    rv_ft_tr_scaled = (y_raw_all[:split_ft] ** 2) * self.target_scale
                    wtd_ft_tr_scaled = self._cached_wtd_for_prefix(rv_ft_tr_scaled, offset=0)
                    lag_m = self.params["lag_monthly"]
                    if len(rv_ft_tr_scaled) > lag_m:
                        x_ft_tr, valid_ft_tr = self._build_har_design(wtd_ft_tr_scaled)
                        y_ft_target = rs_all[:split_ft][valid_ft_tr]
                        self.beta_qr = self._fit_qr(x_ft_tr, y_ft_target)

                    feat_all = self._build_causal_feature_matrix(
                        y_raw=y_raw_all,
                        y_scaled=y_scaled_all,
                        rs_scaled=rs_all,
                        beta_qr=self.beta_qr,
                        scale_gru_inputs=True,
                    )

                    feat_ft_train = feat_all[:split_ft]
                    rs_ft_train = rs_all[:split_ft]

                    val_start_ctx = max(0, split_ft - context_len)
                    feat_ft_val = feat_all[val_start_ctx:]
                    rs_ft_val = rs_all[val_start_ctx:]

                    ft_train_dataset = CausalWTDHARWindowDataset(
                        feat_ft_train, rs_ft_train, context_len
                    )
                    ft_val_dataset = CausalWTDHARWindowDataset(
                        feat_ft_val, rs_ft_val, context_len
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

            del x_tensor
            if i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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