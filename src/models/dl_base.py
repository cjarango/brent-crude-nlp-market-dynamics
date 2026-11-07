"""
dl_base
=======

Base class for Deep Learning volatility models (LSTM, GRU, TCN, etc.).
"""

import copy
import json
from pathlib import Path

try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11
    from datetime import datetime, timezone

    UTC = timezone.utc

import numpy as np
import pandas as pd
import torch
import torch.nn.utils as torch_utils
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset import TimeSeriesDataset
from src.eval.losses import QLIKELoss
from src.models.base import BaseVolatilityModel
from src.utils.logger import setup_logger

__all__ = ["BasePyTorchVolatilityModel"]


class BasePyTorchVolatilityModel(BaseVolatilityModel):
    """Base class for all PyTorch-based volatility models."""

    def __init__(
        self,
        max_epochs: int = 500,
        patience: int = 20,
        tolerance: float = 1e-5,
        batch_size: int = 128,
        retrain_freq: int = 1,
        seed: int = 42,
        target_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.best_state_dict = None
        self.seed = seed
        self.target_scale = target_scale 
        
        self.scaler = StandardScaler()

        self.diagnostics = {"converged": False, "epochs_run": 0}

        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.retrain_freq = retrain_freq

        self.criterion = QLIKELoss(eps=1e-8)

        self.y_train_history = None
        self.rs_train_history = None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "ft_train_loss": [],
            "ft_val_loss": [],
        }

        self.logger = setup_logger(self.__class__.__name__)

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        """Trains the model with a strict chronological 80/20 internal split."""
        if rs_train is None:
            raise ValueError("rs_train (Rogers-Satchell targets) must be provided.")

        self.history["train_loss"] = []
        self.history["val_loss"] = []

        split_idx = int(len(y_train) * 0.8)
        
        y_tr_raw = y_train[:split_idx]
        y_val_raw = y_train[split_idx:]
        self.scaler.fit(y_tr_raw.reshape(-1, 1))
        
        y_tr = self.scaler.transform(y_tr_raw.reshape(-1, 1)).flatten()
        y_val = self.scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
        
        self.y_train_history = self.scaler.transform(y_train.reshape(-1, 1)).flatten()
        self.rs_train_history = rs_train.copy()

        rs_tr = rs_train[:split_idx] * self.target_scale
        rs_val = rs_train[split_idx:] * self.target_scale

        train_dataset = TimeSeriesDataset(y_tr, rs_tr, self.context_length)
        val_dataset = TimeSeriesDataset(y_val, rs_val, self.context_length)

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
                preds = self.net(x_batch)
                loss = self.criterion(preds, y_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error("Loss became NaN or Inf during training.")
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

        if not converged:
            self.logger.warning(
                f"[{self.__class__.__name__}] Training did NOT converge. "
                f"Reached {self.max_epochs} epochs without triggering Early Stopping."
            )

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
        """Generates rolling one-step predictions with expanding-window fine-tuning."""
        if rs_seq is None:
            raise ValueError("rs_seq (Rogers-Satchell targets) must be provided.")

        self.net.eval()
        context_len = self.context_length
        predictions = []

        history_y = list(self.y_train_history)
        history_rs = list(self.rs_train_history * self.target_scale)

        for i in range(len(y_seq)):
            current_window = np.array(history_y[-context_len:])
            x_tensor = (
                torch.tensor(current_window, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )

            with torch.no_grad():
                pred = self.net(x_tensor).item()
            predictions.append(pred)

            actual_y_raw = y_seq[i]
            
            actual_y = self.scaler.transform([[actual_y_raw]])[0, 0]
            actual_rs = rs_seq[i] * self.target_scale

            if debug:
                print(f"[DEBUG] pred i={i:03d} | y_raw={actual_y_raw:+.6f} | y_scaled={actual_y:+.6f} | rs={actual_rs/self.target_scale:.6e} | pred={pred/self.target_scale:.6e}")

            history_y.append(actual_y)
            history_rs.append(actual_rs)

            if i % self.retrain_freq == 0:
                val_len = max(context_len + 1, min(30, len(history_y) - context_len - 1))
                
                if val_len > 0 and len(history_y) > val_len:
                    y_ft_train = np.array(history_y[:-val_len])
                    rs_ft_train = np.array(history_rs[:-val_len])
                    y_ft_val = np.array(history_y[-val_len:])
                    rs_ft_val = np.array(history_rs[-val_len:])

                    ft_train_dataset = TimeSeriesDataset(y_ft_train, rs_ft_train, context_len)
                    ft_val_dataset = TimeSeriesDataset(y_ft_val, rs_ft_val, context_len)

                    if len(ft_train_dataset) > 0 and len(ft_val_dataset) > 0:
                        ft_train_loader = DataLoader(ft_train_dataset, batch_size=self.batch_size, shuffle=False)
                        ft_val_loader = DataLoader(ft_val_dataset, batch_size=self.batch_size, shuffle=False)

                        optimizer = optim.Adam(
                            self.net.parameters(),
                            lr=self.params["learning_rate"],
                            weight_decay=self.params.get("weight_decay", 0.0),
                        )
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

                        if not ft_converged:
                            self.logger.warning(
                                f"[{self.__class__.__name__}] Fine-Tuning at day {i} did NOT converge."
                            )

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

    def _save_artifact(
        self,
        save_dir: str,
        y_seq: np.ndarray,
        rs_seq: np.ndarray,
        predictions: np.ndarray,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Persists the model state and companion artifacts."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.net.state_dict(), save_path / "model_state_dict.pt")

        metadata = {
            "class": self.__class__.__name__,
            "torch_version": torch.__version__,
            "seed": self.seed,
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "device_at_save": str(self.device),
            "horizon": 1,
            "forecast_mode": "one-step",
            "context_length": int(self.context_length),
            "params": dict(self.params),
            "trainer_hps": {
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "tolerance": self.tolerance,
                "batch_size": self.batch_size,
                "retrain_freq": self.retrain_freq,
            },
            "fit_diagnostics": dict(self.diagnostics),
            "sizes": {
                "train_history": len(self.y_train_history),
                "test_input": len(y_seq),
                "predictions": len(predictions),
            },
            "preprocessing": {"target_scale": self.target_scale, "input_scaler": "StandardScaler"},
            "data_hash": None,
            "validation_metrics": {},
            "test_metrics": {},
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        (save_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        pd.DataFrame(
            {
                "epoch": range(len(self.history["train_loss"])),
                "train_loss": self.history["train_loss"],
                "val_loss": self.history["val_loss"],
            }
        ).to_csv(save_path / "fit_history.csv", index=False)

        if self.history.get("ft_train_loss"):
            pd.DataFrame(
                {
                    "step": range(len(self.history["ft_train_loss"])),
                    "ft_train_loss": self.history["ft_train_loss"],
                    "ft_val_loss": self.history["ft_val_loss"],
                }
            ).to_csv(save_path / "ft_history.csv", index=False)

        np.savez_compressed(
            save_path / "history_buffer.npz",
            history_y=np.asarray(self.y_train_history),
            history_rs=np.asarray(self.rs_train_history),
        )

        n_pred = len(predictions)
        y_true_aligned = y_seq[-n_pred:] if len(y_seq) >= n_pred else y_seq
        rs_true_aligned = rs_seq[-n_pred:] if len(rs_seq) >= n_pred else rs_seq
        pd.DataFrame(
            {
                "step": range(n_pred),
                "y_true": y_true_aligned,
                "y_pred": predictions,
                "rs_true": rs_true_aligned,
            }
        ).to_csv(save_path / "walk_forward_predictions.csv", index=False)

        self.logger.info(f"[{self.__class__.__name__}] Artifacts saved to {save_path.resolve()}")
        return save_path

    def get_diagnostics(self) -> dict:
        """Returns the diagnostics of the last fit call."""
        return self.diagnostics

    def get_history(self) -> dict:
        """Returns the training and validation loss curves."""
        return self.history