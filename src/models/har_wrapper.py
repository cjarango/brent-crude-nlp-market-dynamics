import json
from pathlib import Path
from typing import Any

try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone
    UTC = timezone.utc

import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from tqdm.auto import tqdm

from src.models.base import BaseVolatilityModel

__all__ = ["HARWrapper"]


class HARWrapper(BaseVolatilityModel):

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "lag_weekly": trial.suggest_categorical("lag_weekly", [5, 7]),
            "lag_monthly": trial.suggest_categorical("lag_monthly", [22, 30]),
            "seed": 42
        }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lag_weekly = kwargs.get('lag_weekly', 5)
        self.lag_monthly = kwargs.get('lag_monthly', 22)
        self.seed = kwargs.get('seed', 42)

        self.model_fit = None
        self.y_train_history: np.ndarray = np.array([])
        self.rs_train_history: np.ndarray = np.array([])
        
        self.converged: bool = True
        self.is_fitted: bool = False
        self.is_valid: bool = False

    @property
    def context_length(self) -> int:
        return self.lag_monthly

    def _build_features(self, y_series: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({'r': y_series})
        df['RV'] = df['r'] ** 2
        df['RV_d'] = df['RV'].shift(1)
        df['RV_w'] = df['RV'].shift(1).rolling(window=self.lag_weekly).mean()
        df['RV_m'] = df['RV'].shift(1).rolling(window=self.lag_monthly).mean()
        return df.dropna()

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")
        
        self.y_train_history = y_train.copy()
        self.rs_train_history = rs_train.copy()
        
        try:
            df_features = self._build_features(self.y_train_history)
            X = df_features[['RV_d', 'RV_w', 'RV_m']]
            y = rs_train[self.lag_monthly:]
            
            X = sm.add_constant(X)
            self.model_fit = sm.OLS(y, X).fit()
            
            self.is_valid = True
            self.is_fitted = True
        except Exception as e:  # noqa: BLE001
            self.is_fitted = False
            self.is_valid = False
            raise RuntimeError(f"HAR fitting failed: {e}")

    def forecast(self, y_seq: np.ndarray, rs_seq: np.ndarray | None = None, save_dir: str | None = None, extra_metadata: dict | None = None, debug: bool = False) -> np.ndarray:
        if not self.is_fitted or not self.is_valid:
            raise RuntimeError("Model must be fitted before forecasting.")
        if rs_seq is None:
            raise ValueError("rs_seq must be provided.")

        predictions = []
        history_y = list(self.y_train_history)
        history_rs = list(self.rs_train_history)

        for i in range(len(y_seq)):
            try:
                df_hist = self._build_features(np.array(history_y))
                X_pred = df_hist[['RV_d', 'RV_w', 'RV_m']].iloc[[-1]]
                X_pred = sm.add_constant(X_pred, has_constant='add')
                
                X_hist = df_hist[['RV_d', 'RV_w', 'RV_m']]
                y_hist = np.array(history_rs[self.lag_monthly:])
                X_hist = sm.add_constant(X_hist)
                
                local_fit = sm.OLS(y_hist, X_hist).fit()
                pred = local_fit.predict(X_pred).iloc[0]
                predictions.append(max(pred, 1e-8)) 
                
            except Exception as e:  # noqa: BLE001
                if len(predictions) > 0:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(np.var(history_y))
                if debug:
                    print(f"[DEBUG] OLS failed at step {i}: {e}")

            history_y.append(y_seq[i])
            history_rs.append(rs_seq[i])

        predictions = np.array(predictions)
        if save_dir is not None:
            self._save_artifact(save_dir, y_seq, rs_seq, predictions, extra_metadata)
        return predictions

    def _save_artifact(self, save_dir, y_seq, rs_seq, predictions, extra_metadata):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "class": self.__class__.__name__,
            "params": {"lag_weekly": self.lag_weekly, "lag_monthly": self.lag_monthly},
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "fit_diagnostics": self.get_diagnostics(),
        }
        if extra_metadata: 
            metadata.update(extra_metadata)
            
        (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pd.DataFrame({
            "step": range(len(predictions)),
            "y_true": y_seq[-len(predictions):] if len(y_seq) >= len(predictions) else y_seq,
            "y_pred": predictions,
            "rs_true": rs_seq[-len(predictions):] if rs_seq is not None and len(rs_seq) >= len(predictions) else rs_seq
        }).to_csv(save_path / "walk_forward_predictions.csv", index=False)

    def get_diagnostics(self) -> dict:
        if not self.is_fitted: 
            return {}
        return {
            "r_squared": self.model_fit.rsquared, 
            "converged": True, 
            "epochs_run": 1
        }

    def get_history(self) -> dict:
        return {"train_loss": [], "val_loss": []}

    def get_parameter_table(
        self, 
        B: int = 0, 
        decimals: int = 4, 
        block_size: int = 22,
        verbose: bool = False
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be successfully fitted to extract parameters.")

        params = self.model_fit.params
        std_err = self.model_fit.bse
        pvalues = self.model_fit.pvalues

        if B > 0:
            ci_lower, ci_upper = self._block_bootstrap_ci(B, block_size, verbose)
            lower_vals = [ci_lower[p] for p in params.index]
            upper_vals = [ci_upper[p] for p in params.index]
        else:
            conf_int = self.model_fit.conf_int()
            lower_vals = conf_int.iloc[:, 0].values
            upper_vals = conf_int.iloc[:, 1].values

        df = pd.DataFrame({
            "Parameter": params.index,
            "Coef": params.values,
            "Std. Error": std_err.values,
            "P-value": pvalues.values,
            "CI Lower": lower_vals,
            "CI Upper": upper_vals
        })

        return df.round(decimals)
    
    def _block_bootstrap_ci(self, B: int, block_size: int, verbose: bool) -> tuple[dict, dict]:
        y_hat = np.asarray(self.model_fit.fittedvalues)
        resid = np.asarray(self.model_fit.resid)
        X = np.asarray(self.model_fit.model.exog)
        params_names = self.model_fit.params.index
        
        n = len(resid)
        n_blocks = int(np.ceil(n / block_size))
        
        boot_params = {p: [] for p in params_names}
        
        for _ in tqdm(range(B), desc="Block Bootstrap", leave=False, disable=not verbose):
            starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
            boot_resid = np.concatenate([resid[s:s+block_size] for s in starts])[:n]
            
            y_boot = y_hat + boot_resid
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y_boot, rcond=None)
                for i, p in enumerate(params_names):
                    boot_params[p].append(beta[i])
            except Exception:  # noqa: BLE001, S112
                continue
                
        ci_lower = {}
        ci_upper = {}
        for p in params_names:
            vals = np.array(boot_params[p])
            if len(vals) > 0:
                ci_lower[p] = np.percentile(vals, 2.5)
                ci_upper[p] = np.percentile(vals, 97.5)
            else:
                ci_lower[p] = np.nan
                ci_upper[p] = np.nan
                
        return ci_lower, ci_upper