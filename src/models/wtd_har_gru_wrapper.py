"""
wtd_har_gru_wrapper
===================

Implementation of the WTD-HAR-RV-GRU hybrid model based on Pan et al. (2025).
Includes strict walk-forward fine-tuning of the GRU at every step.
"""

import copy
import gc

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.models.dl_base import BasePyTorchVolatilityModel
from src.models.har_wtd_wrapper import HARWTDWrapper

__all__ = ["WTDHARGRUWrapper"]


class HARGRUDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, look_back: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.look_back = look_back

    def __len__(self):
        return len(self.y) - self.look_back

    def __getitem__(self, idx):
        window = self.X[idx : idx + self.look_back]
        target = self.y[idx + self.look_back]
        return window, target


class HARGRUNetwork(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.eps = 1e-8

    def forward(self, x):
        out, _ = self.gru(x)
        return F.softplus(self.fc(out[:, -1, :])) + self.eps


class WTDHARGRUWrapper(BasePyTorchVolatilityModel):
    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "lag_weekly": trial.suggest_categorical("lag_weekly", [5, 7]),
            "lag_monthly": trial.suggest_categorical("lag_monthly", [22, 30]),
            "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "max_epochs": 200,
            "patience": 20,
            "retrain_freq": 1,
            "target_scale": 1e4,
            "seed": 42,
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
        **kwargs
    ) -> None:
        effective_dropout = dropout if num_layers > 1 else 0.0
        
        super().__init__(
            look_back=look_back,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lag_weekly=lag_weekly,
            lag_monthly=lag_monthly,
            **kwargs
        )
        
        self.net = HARGRUNetwork(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
        ).to(self.device)
        
        self.har_model = HARWTDWrapper(lag_weekly=lag_weekly, lag_monthly=lag_monthly)
        
        self.feature_mean = None
        self.feature_std = None
        self.train_har_preds = None
        self.train_har_resids = None

    @property
    def context_length(self) -> int:
        return self.params["look_back"]

    def _build_3channel_data(self, history_y: np.ndarray, history_rs: np.ndarray):
        self.har_model.fit(history_y, history_rs)
        har_preds = self.har_model.forecast(history_y, history_rs)
        har_resids = har_preds - history_rs
        X = np.column_stack([history_y, har_preds, har_resids])
        return X, har_preds, har_resids

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")

        self.y_train_history = y_train.copy()
        self.rs_train_history = rs_train.copy()
        self.history["train_loss"] = []
        self.history["val_loss"] = []
        self.history["ft_train_loss"] = []
        self.history["ft_val_loss"] = []

        X, har_preds, har_resids = self._build_3channel_data(y_train, rs_train)
        self.train_har_preds = har_preds
        self.train_har_resids = har_resids

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        rs_train_scaled = rs_train * self.target_scale
        rs_train_split, rs_val_split = rs_train_scaled[:split_idx], rs_train_scaled[split_idx:]

        self.feature_mean = np.mean(X_train, axis=0)
        self.feature_std = np.std(X_train, axis=0) + 1e-8
        
        X_train = (X_train - self.feature_mean) / self.feature_std
        X_val = (X_val - self.feature_mean) / self.feature_std

        train_dataset = HARGRUDataset(X_train, rs_train_split, self.context_length)
        val_dataset = HARGRUDataset(X_val, rs_val_split, self.context_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.net.parameters(), lr=self.params["learning_rate"], weight_decay=self.params.get("weight_decay", 0.0))
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
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forecast(self, y_seq: np.ndarray, rs_seq: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if not self.is_fitted or not self.is_valid:
            raise RuntimeError("Model must be fitted before forecasting.")
        if rs_seq is None:
            raise ValueError("rs_seq must be provided.")

        predictions = []
        history_y = list(self.y_train_history)
        history_rs = list(self.rs_train_history)
        
        history_har_preds = list(self.train_har_preds)
        history_har_resids = list(self.train_har_resids)

        context_len = self.context_length

        for i in range(len(y_seq)):
            har_hist_y = np.array(history_y)
            har_hist_rs = np.array(history_rs)
            self.har_model.fit(har_hist_y, har_hist_rs)
            
            har_pred = self.har_model.forecast(np.array([y_seq[i]]), np.array([rs_seq[i]]))[0]
            har_resid = har_pred - rs_seq[i]
            
            history_har_preds.append(har_pred)
            history_har_resids.append(har_resid)

            window_y = np.array(history_y[-context_len:]) 
            window_har = np.array(history_har_preds[-context_len-1 : -1])
            window_resid = np.array(history_har_resids[-context_len-1 : -1])

            x_input = np.column_stack([window_y, window_har, window_resid])
            x_input = (x_input - self.feature_mean) / self.feature_std
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)

            self.net.eval()
            with torch.no_grad():
                pred = self.net(x_tensor).item()
            predictions.append(pred)

            history_y.append(y_seq[i])
            history_rs.append(rs_seq[i])

            if i % self.retrain_freq == 0:
                val_len = max(context_len + 1, min(30, len(history_y) - context_len - 1))
                
                if val_len > 0 and len(history_y) > val_len:
                    X_hist = np.column_stack([history_y, history_har_preds, history_har_resids])
                    
                    X_ft_train = X_hist[:-val_len]
                    X_ft_val = X_hist[-val_len:]
                    
                    rs_ft_train = np.array(history_rs[:-val_len]) * self.target_scale
                    rs_ft_val = np.array(history_rs[-val_len:]) * self.target_scale

                    X_ft_train = (X_ft_train - self.feature_mean) / self.feature_std
                    X_ft_val = (X_ft_val - self.feature_mean) / self.feature_std

                    ft_train_dataset = HARGRUDataset(X_ft_train, rs_ft_train, context_len)
                    ft_val_dataset = HARGRUDataset(X_ft_val, rs_ft_val, context_len)

                    if len(ft_train_dataset) > 0 and len(ft_val_dataset) > 0:
                        ft_train_loader = DataLoader(ft_train_dataset, batch_size=self.batch_size, shuffle=False)
                        ft_val_loader = DataLoader(ft_val_dataset, batch_size=self.batch_size, shuffle=False)

                        optimizer = optim.Adam(self.net.parameters(), lr=self.params["learning_rate"], weight_decay=self.params.get("weight_decay", 0.0))
                        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

                        self.net.train()
                        best_ft_loss = float("inf")
                        patience_counter = 0
                        ft_converged = False
                        best_ft_state_dict = None

                        for epoch in range(self.max_epochs):
                            self.net.train()
                            ft_epoch_loss = 0.0
                            for x_batch, y_batch in ft_train_loader:
                                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                                optimizer.zero_grad()
                                loss = self.criterion(self.net(x_batch), y_batch)
                                if torch.isnan(loss) or torch.isinf(loss):
                                    raise RuntimeError("NaN/Inf loss detected during Fine-Tuning")
                                loss.backward()
                                torch_utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                                optimizer.step()
                                ft_epoch_loss += loss.item()
                            ft_epoch_loss /= len(ft_train_loader)

                            self.net.eval()
                            ft_val_loss = 0.0
                            with torch.no_grad():
                                for x_batch, y_batch in ft_val_loader:
                                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                                    ft_val_loss += self.criterion(self.net(x_batch), y_batch).item()
                            ft_val_loss /= len(ft_val_loader)

                            self.history["ft_train_loss"].append(ft_epoch_loss)
                            self.history["ft_val_loss"].append(ft_val_loss)
                            scheduler.step(ft_val_loss)

                            if best_ft_loss - ft_val_loss > self.tolerance:
                                best_ft_loss = ft_val_loss
                                patience_counter = 0
                                best_ft_state_dict = copy.deepcopy(self.net.state_dict())
                            else:
                                patience_counter += 1
                                if patience_counter >= self.patience:
                                    ft_converged = True
                                    break

                        if best_ft_state_dict is not None:
                            self.net.load_state_dict(best_ft_state_dict)
                        self.net.eval()

            del x_tensor
            if i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        predictions = np.array(predictions) / self.target_scale
        return predictions