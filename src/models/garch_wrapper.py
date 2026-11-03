"""
garch_wrapper
=============

Implementation of the GARCH model wrapper.

This module provides a standardized wrapper for the Generalized 
Autoregressive Conditional Heteroskedasticity (GARCH) model using 
the `arch` library. It includes parameter estimation, forecasting, 
diagnostic extraction, and non-parametric block bootstrap.
"""

import json
from pathlib import Path
from typing import Any

try:
    from datetime import UTC, datetime
except ImportError:  # Python < 3.11
    from datetime import datetime, timezone
    UTC = timezone.utc

import numpy as np
import optuna
import pandas as pd
from arch import arch_model
from src.models.base import BaseVolatilityModel
from tqdm.auto import tqdm

__all__ = ["GARCHWrapper"]


class GARCHWrapper(BaseVolatilityModel):
    """Wrapper for the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        """Defines the hyperparameter search space for Optuna."""
        return {
            "p": trial.suggest_int("p", 1, 3),
            "q": trial.suggest_int("q", 1, 3),
            "dist": trial.suggest_categorical("dist", ["normal", "t", "ged"]),
            "seed": 42
        }

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the GARCHWrapper."""
        super().__init__(**kwargs)
        
        self.p = kwargs.get('p', 1)
        self.q = kwargs.get('q', 1)
        self.dist = kwargs.get('dist', 'normal')
        self.seed = kwargs.get('seed', 42)

        if not isinstance(self.p, int) or self.p < 1:
            raise ValueError(f"Parameter 'p' must be an integer >= 1. Got {self.p}.")
        if not isinstance(self.q, int) or self.q < 1:
            raise ValueError(f"Parameter 'q' must be an integer >= 1. Got {self.q}.")
            
        valid_dists = ['normal', 't', 'ged']
        if self.dist not in valid_dists:
            raise ValueError(f"Parameter 'dist' must be one of {valid_dists}. Got '{self.dist}'.")

        self.model_fit = None
        self.y_train_history: np.ndarray = np.array([])
        
        self.converged: bool = False
        self.is_stationary: bool = False
        self.is_fitted: bool = False
        self.is_valid: bool = False

    @property
    def context_length(self) -> int:
        """Returns the number of past observations needed as context."""
        return max(self.p, self.q)

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        """Fits the GARCH model using Maximum Likelihood Estimation (MLE)."""
        self.y_train_history = y_train.copy()
        
        scaled_returns = y_train * 100.0
        
        try:
            am = arch_model(
                scaled_returns, 
                mean='Zero', 
                vol='GARCH', 
                p=self.p, 
                q=self.q, 
                dist=self.dist
            )
            self.model_fit = am.fit(disp='off', show_warning=False)
            
            self.converged = bool(self.model_fit.convergence_flag == 0)
            
            if self.converged:
                alphas = [self.model_fit.params.get(f'alpha[{i}]', 0.0) for i in range(1, self.q + 1)]
                betas = [self.model_fit.params.get(f'beta[{j}]', 0.0) for j in range(1, self.p + 1)]
                sum_persistence = np.sum(alphas) + np.sum(betas)
                self.is_stationary = bool(sum_persistence < 1.0)
            else:
                self.is_stationary = False
                
            self.is_valid = self.converged and self.is_stationary
            self.is_fitted = True
            
        except Exception as e:  # noqa: BLE001
            self.is_fitted = False
            self.is_valid = False
            raise RuntimeError(f"GARCH fitting failed on training set: {e}")

    def forecast(
        self,
        y_seq: np.ndarray,
        rs_seq: np.ndarray | None = None,
        save_dir: str | None = None,
        extra_metadata: dict | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        """Generates rolling one-step predictions with re-estimation at EVERY step."""
        if not self.is_fitted or not self.is_valid:
            raise RuntimeError("Model must be successfully fitted before forecasting.")

        predictions = []
        history = list(self.y_train_history)

        for i in range(len(y_seq)):
            # CORRECCIÓN: Eliminado el if i < context_len. 
            # Ahora predecimos desde el primer día de validación usando el historial de train.
            scaled_history = np.array(history) * 100.0
            
            try:
                am_local = arch_model(
                    scaled_history, 
                    mean='Zero', 
                    vol='GARCH', 
                    p=self.p, 
                    q=self.q, 
                    dist=self.dist
                )
                local_fit = am_local.fit(disp='off', show_warning=False)
                
                forecast_res = local_fit.forecast(horizon=1, reindex=False)
                
                var_pred = forecast_res.variance.values[-1, 0] / (100.0 ** 2)
                predictions.append(var_pred)
                
            except Exception as e:  # noqa: BLE001
                if len(predictions) > 0:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(np.var(scaled_history) / (100.0 ** 2))
                
                if debug:
                    print(f"[DEBUG] MLE failed at step {i}: {e}")

            actual_y = y_seq[i]
            history.append(actual_y)

            if debug:
                print(f"[DEBUG] pred i={i:03d} | y={actual_y:+.6f} | pred={predictions[-1]:.6e}")

        predictions = np.array(predictions)

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
        rs_seq: np.ndarray | None,
        predictions: np.ndarray,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Persists the predictions and companion artifacts (no neural weights)."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "class": self.__class__.__name__,
            "seed": self.seed,
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "horizon": 1,
            "forecast_mode": "one-step",
            "context_length": int(self.context_length),
            "params": {"p": self.p, "q": self.q, "dist": self.dist},
            "fit_diagnostics": {"converged": self.converged, "is_stationary": self.is_stationary},
            "sizes": {
                "train_history": len(self.y_train_history),
                "test_input": len(y_seq),
                "predictions": len(predictions),
            },
            "preprocessing": {"scaling_factor": 100.0},
            "data_hash": None,
            "validation_metrics": {},
            "test_metrics": {},
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        (save_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        n_pred = len(predictions)
        y_true_aligned = y_seq[-n_pred:] if len(y_seq) >= n_pred else y_seq
        rs_true_aligned = rs_seq[-n_pred:] if rs_seq is not None and len(rs_seq) >= n_pred else (rs_seq if rs_seq is not None else None)
        
        df_dict = {
            "step": range(n_pred),
            "y_true": y_true_aligned,
            "y_pred": predictions,
        }
        if rs_true_aligned is not None:
            df_dict["rs_true"] = rs_true_aligned

        pd.DataFrame(df_dict).to_csv(save_path / "walk_forward_predictions.csv", index=False)

    def get_diagnostics(self) -> dict:
        """Returns diagnostic data from the fitted model (TRAINING SET ONLY)."""
        if not self.is_fitted:
            return {}
            
        return {
            "params": self.model_fit.params.to_dict() if self.converged else {},
            "loglikelihood": self.model_fit.loglikelihood if self.converged else np.nan,
            "converged": self.converged,
            "is_stationary": self.is_stationary,
            "epochs_run": 0,
        }

    def get_history(self) -> dict:
        """Returns an empty history, as GARCH does not use epochs."""
        return {"train_loss": [], "val_loss": []}

    def get_parameter_table(
        self, 
        B: int = 0, 
        decimals: int = 4, 
        block_size: int = 22,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Returns a DataFrame with coefficients, standard errors, p-values, and CIs."""
        if not self.is_fitted or not self.converged:
            raise RuntimeError("Model must be successfully fitted to extract parameters.")

        params = self.model_fit.params
        std_err = self.model_fit.std_err
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
        """Calculates confidence intervals using non-parametric block bootstrap."""
        n = len(self.y_train_history)
        n_blocks = int(np.ceil(n / block_size))
        
        std_resid = self.model_fit.std_resid
        cond_vol = self.model_fit.conditional_volatility
        
        boot_params = {p: [] for p in self.model_fit.params.index}
        
        for _ in tqdm(range(B), desc="Block Bootstrap", leave=False, disable=not verbose):
            block_starts = np.random.randint(0, n - block_size, n_blocks)
            boot_resid = np.concatenate([std_resid[s:s+block_size] for s in block_starts])[:n]
            
            y_boot = boot_resid * cond_vol
            
            try:
                am_boot = arch_model(y_boot, mean='Zero', vol='GARCH', p=self.p, q=self.q, dist=self.dist)
                res_boot = am_boot.fit(disp='off')
                for p in self.model_fit.params.index:
                    boot_params[p].append(res_boot.params[p])
            except Exception:  # noqa: BLE001, S112
                continue
                
        ci_lower = {}
        ci_upper = {}
        for p in self.model_fit.params.index:
            vals = np.array(boot_params[p])
            if len(vals) > 0:
                ci_lower[p] = np.percentile(vals, 2.5)
                ci_upper[p] = np.percentile(vals, 97.5)
            else:
                ci_lower[p] = np.nan
                ci_upper[p] = np.nan
                
        return ci_lower, ci_upper