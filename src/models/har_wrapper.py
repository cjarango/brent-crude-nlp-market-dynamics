from __future__ import annotations

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
from scipy import stats
from tqdm.auto import tqdm

from src.models.base import BaseVolatilityModel

__all__ = ["HARQRFitResult", "HARWrapper"]

_PARAM_NAMES = ["const", "RV_d", "RV_w", "RV_m"]


class HARQRFitResult:
    """Stores QR regression estimates and HAC/Newey-West inference for HARWrapper."""

    def __init__(
        self,
        params: np.ndarray,
        x_mat: np.ndarray,
        y_vec: np.ndarray,
        cov_type: str = "hac",
        hac_lags: int | None = None,
        param_names: list[str] | None = None,
    ) -> None:
        self.param_names = param_names if param_names is not None else list(_PARAM_NAMES)
        self.params = pd.Series(np.asarray(params, dtype=np.float64), index=self.param_names)
        self.x_mat = np.asarray(x_mat, dtype=np.float64)
        self.y_vec = np.asarray(y_vec, dtype=np.float64)
        self.cov_type = cov_type
        self.hac_lags = hac_lags

        self.fittedvalues = self.x_mat @ self.params.to_numpy(dtype=np.float64)
        self.resid = self.y_vec - self.fittedvalues

        n_obs, n_features = self.x_mat.shape
        dof = max(1, n_obs - n_features)
        self.dof = dof

        rss = float(np.sum(self.resid ** 2))
        ss_tot = float(np.sum((self.y_vec - np.mean(self.y_vec)) ** 2))
        self.rsquared = float(1.0 - rss / ss_tot) if ss_tot > 0.0 else 0.0

        _, r_mat = np.linalg.qr(self.x_mat, mode="reduced")

        if cov_type == "hac":
            cov_beta = self._newey_west_cov(
                x_mat=self.x_mat,
                resid=self.resid,
                r_mat=r_mat,
                max_lags=hac_lags,
            )
        elif cov_type == "qr":
            s2 = rss / dof
            try:
                r_inv = np.linalg.inv(r_mat)
                cov_beta = s2 * (r_inv @ r_inv.T)
            except np.linalg.LinAlgError:
                cov_beta = s2 * np.linalg.pinv(self.x_mat.T @ self.x_mat)
        else:
            raise ValueError(f"Unsupported cov_type='{cov_type}'. Use 'hac' or 'qr'.")

        self.cov_params = pd.DataFrame(cov_beta, index=self.param_names, columns=self.param_names)
        se_vec = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
        self.bse = pd.Series(se_vec, index=self.param_names)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.where(se_vec > 0.0, self.params.to_numpy() / se_vec, 0.0)
        self.tvalues = pd.Series(t_stats, index=self.param_names)
        p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)
        self.pvalues = pd.Series(np.clip(p_vals, 0.0, 1.0), index=self.param_names)

    @staticmethod
    def _newey_west_cov(
        x_mat: np.ndarray,
        resid: np.ndarray,
        r_mat: np.ndarray,
        max_lags: int | None = None,
        small_sample_adj: bool = True,
    ) -> np.ndarray:
        """Computes Newey-West HAC covariance (X'X)^(-1) S_HAC (X'X)^(-1) via R."""
        n_obs, n_feat = x_mat.shape
        if n_obs <= 1:
            return np.zeros((n_feat, n_feat), dtype=np.float64)

        if max_lags is None:
            max_lags = max(1, int(np.floor(4.0 * ((n_obs / 100.0) ** (2.0 / 9.0)))))
        max_lags = min(max(0, int(max_lags)), n_obs - 1)

        weighted_x = x_mat * resid[:, None]
        s_hac = weighted_x.T @ weighted_x

        for lag in range(1, max_lags + 1):
            weight = 1.0 - lag / (max_lags + 1.0)
            gamma_lag = weighted_x[lag:].T @ weighted_x[:-lag]
            s_hac += weight * (gamma_lag + gamma_lag.T)

        if small_sample_adj and n_obs > n_feat:
            s_hac *= n_obs / (n_obs - n_feat)

        try:
            r_inv = np.linalg.inv(r_mat)
            middle = r_inv.T @ s_hac @ r_inv
            return r_inv @ middle @ r_inv.T
        except np.linalg.LinAlgError:
            xtx_pinv = np.linalg.pinv(x_mat.T @ x_mat)
            return xtx_pinv @ s_hac @ xtx_pinv

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_pred, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr @ self.params.to_numpy(dtype=np.float64)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        q_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=self.dof))
        coef = self.params.to_numpy(dtype=np.float64)
        se = self.bse.to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {
                "lower": coef - q_crit * se,
                "upper": coef + q_crit * se,
            },
            index=self.param_names,
        )


class HARWrapper(BaseVolatilityModel):
    """Heterogeneous Autoregressive (HAR) volatility model fitted via QR decomposition."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "lag_weekly": trial.suggest_categorical("lag_weekly", [5, 7]),
            "lag_monthly": trial.suggest_categorical("lag_monthly", [22, 30]),
            "seed": 42,
        }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lag_weekly: int = int(kwargs.get("lag_weekly", 5))
        self.lag_monthly: int = int(kwargs.get("lag_monthly", 22))
        self.seed: int = int(kwargs.get("seed", 42))
        self.cov_type: str = str(kwargs.get("cov_type", "hac"))
        self.hac_lags: int | None = kwargs.get("hac_lags", None)

        self.model_fit: HARQRFitResult | None = None
        self.y_train_history: np.ndarray = np.array([], dtype=np.float64)
        self.rs_train_history: np.ndarray = np.array([], dtype=np.float64)

        self.converged: bool = True
        self.is_fitted: bool = False
        self.is_valid: bool = False

    @property
    def context_length(self) -> int:
        return self.lag_monthly

    @staticmethod
    def _solve_qr(x_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        """Solves linear least squares via reduced QR decomposition with lstsq fallback."""
        x_arr = np.asarray(x_mat, dtype=np.float64)
        y_arr = np.asarray(y_vec, dtype=np.float64).reshape(-1)
        q_mat, r_mat = np.linalg.qr(x_arr, mode="reduced")
        diag_r = np.abs(np.diag(r_mat))
        if len(diag_r) == 0 or diag_r.min() < 1e-10:
            beta, _, _, _ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
            return np.asarray(beta, dtype=np.float64)
        return np.linalg.solve(r_mat, q_mat.T @ y_arr)

    def _build_design_matrix(self, y_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Builds design matrix [1, RV_d, RV_w, RV_m] and valid target indices."""
        arr = np.asarray(y_series, dtype=np.float64)
        if len(arr) <= self.lag_monthly:
            raise ValueError(
                f"Series length ({len(arr)}) must be greater than lag_monthly ({self.lag_monthly})."
            )
        s = pd.Series(arr ** 2)
        rv_d = s.shift(1)
        rv_w = s.shift(1).rolling(window=self.lag_weekly).mean()
        rv_m = s.shift(1).rolling(window=self.lag_monthly).mean()

        df = pd.DataFrame({"rv_d": rv_d, "rv_w": rv_w, "rv_m": rv_m}).iloc[self.lag_monthly :]
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

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")

        y_arr = np.asarray(y_train, dtype=np.float64)
        rs_arr = np.asarray(rs_train, dtype=np.float64)
        self.y_train_history = y_arr.copy()
        self.rs_train_history = rs_arr.copy()

        try:
            x_mat, valid_idx = self._build_design_matrix(y_arr)
            y_target = rs_arr[valid_idx]
            beta_qr = self._solve_qr(x_mat, y_target)
            self.model_fit = HARQRFitResult(
                params=beta_qr,
                x_mat=x_mat,
                y_vec=y_target,
                cov_type=self.cov_type,
                hac_lags=self.hac_lags,
                param_names=list(_PARAM_NAMES),
            )
            self.is_valid = True
            self.is_fitted = True
        except Exception as e:
            self.is_fitted = False
            self.is_valid = False
            raise RuntimeError(f"HAR QR fitting failed: {e}") from e

    def forecast(
        self,
        y_seq: np.ndarray,
        rs_seq: np.ndarray | None = None,
        save_dir: str | None = None,
        extra_metadata: dict | None = None,
        debug: bool = False,
    ) -> np.ndarray:
        if not self.is_fitted or not self.is_valid or self.model_fit is None:
            raise RuntimeError("Model must be fitted before forecasting.")
        if rs_seq is None:
            raise ValueError("rs_seq must be provided.")

        y_seq_arr = np.asarray(y_seq, dtype=np.float64)
        rs_seq_arr = np.asarray(rs_seq, dtype=np.float64)

        predictions: list[float] = []
        history_y = list(self.y_train_history)
        history_rs = list(self.rs_train_history)

        for i in range(len(y_seq_arr)):
            try:
                hist_y_arr = np.asarray(history_y, dtype=np.float64)
                hist_rs_arr = np.asarray(history_rs, dtype=np.float64)

                y_with_curr = np.append(hist_y_arr, y_seq_arr[i])
                x_all, _ = self._build_design_matrix(y_with_curr)
                x_pred = x_all[-1]

                x_hist, valid_hist = self._build_design_matrix(hist_y_arr)
                y_hist = hist_rs_arr[valid_hist]

                beta_local = self._solve_qr(x_hist, y_hist)
                pred = float(x_pred @ beta_local)
                predictions.append(max(pred, 1e-8))
            except Exception as e:  # noqa: BLE001
                if len(predictions) > 0:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(float(np.var(history_y)) if len(history_y) > 0 else 1e-8)
                if debug:
                    print(f"[DEBUG] QR HAR failed at step {i}: {e}")

            history_y.append(float(y_seq_arr[i]))
            history_rs.append(float(rs_seq_arr[i]))

        preds_arr = np.asarray(predictions, dtype=np.float64)
        if save_dir is not None:
            self._save_artifact(save_dir, y_seq_arr, rs_seq_arr, preds_arr, extra_metadata)
        return preds_arr

    def _save_artifact(
        self,
        save_dir: str | Path,
        y_seq: np.ndarray,
        rs_seq: np.ndarray,
        predictions: np.ndarray,
        extra_metadata: dict | None,
    ) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        metadata: dict[str, Any] = {
            "class": self.__class__.__name__,
            "params": {
                "lag_weekly": self.lag_weekly,
                "lag_monthly": self.lag_monthly,
                "cov_type": self.cov_type,
                "hac_lags": self.hac_lags,
            },
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "fit_diagnostics": self.get_diagnostics(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pd.DataFrame(
            {
                "step": range(len(predictions)),
                "y_true": y_seq[-len(predictions) :] if len(y_seq) >= len(predictions) else y_seq,
                "y_pred": predictions,
                "rs_true": (
                    rs_seq[-len(predictions) :]
                    if rs_seq is not None and len(rs_seq) >= len(predictions)
                    else rs_seq
                ),
            }
        ).to_csv(save_path / "walk_forward_predictions.csv", index=False)

    def get_diagnostics(self) -> dict:
        if not self.is_fitted or self.model_fit is None:
            return {}
        return {
            "r_squared": float(self.model_fit.rsquared),
            "converged": self.converged,
            "epochs_run": 1,
        }

    def get_history(self) -> dict:
        return {"train_loss": [], "val_loss": []}

    def get_parameter_table(
        self,
        B: int = 0,
        decimals: int = 4,
        block_size: int = 22,
        verbose: bool = False,
    ) -> pd.DataFrame:
        if not self.is_fitted or self.model_fit is None:
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
            lower_vals = conf_int.iloc[:, 0].to_numpy()
            upper_vals = conf_int.iloc[:, 1].to_numpy()

        df = pd.DataFrame(
            {
                "Parameter": params.index,
                "Coef": params.to_numpy(),
                "Std. Error": std_err.to_numpy(),
                "P-value": pvalues.to_numpy(),
                "CI Lower": lower_vals,
                "CI Upper": upper_vals,
            }
        )
        return df.round(decimals)

    def _block_bootstrap_ci(
        self,
        B: int,
        block_size: int,
        verbose: bool,
    ) -> tuple[dict[str, float], dict[str, float]]:
        if self.model_fit is None:
            raise RuntimeError("Model must be fitted before computing bootstrap intervals.")

        y_hat = np.asarray(self.model_fit.fittedvalues, dtype=np.float64)
        resid = np.asarray(self.model_fit.resid, dtype=np.float64)
        x_mat = np.asarray(self.model_fit.x_mat, dtype=np.float64)
        param_names = list(self.model_fit.params.index)

        n_obs = len(resid)
        eff_block = max(1, min(int(block_size), n_obs))
        n_blocks = int(np.ceil(n_obs / eff_block))
        rng = np.random.default_rng(self.seed)

        boot_params: dict[str, list[float]] = {p: [] for p in param_names}

        for _ in tqdm(range(B), desc="Block Bootstrap (QR)", leave=False, disable=not verbose):
            starts = rng.integers(0, n_obs - eff_block + 1, size=n_blocks)
            boot_resid = np.concatenate([resid[s : s + eff_block] for s in starts])[:n_obs]
            y_boot = y_hat + boot_resid

            try:
                beta_boot = self._solve_qr(x_mat, y_boot)
                for idx, name in enumerate(param_names):
                    boot_params[name].append(float(beta_boot[idx]))
            except Exception:  # noqa: BLE001, S112
                continue

        ci_lower: dict[str, float] = {}
        ci_upper: dict[str, float] = {}
        for name in param_names:
            vals = np.asarray(boot_params[name], dtype=np.float64)
            if len(vals) > 0:
                ci_lower[name] = float(np.percentile(vals, 2.5))
                ci_upper[name] = float(np.percentile(vals, 97.5))
            else:
                ci_lower[name] = float("nan")
                ci_upper[name] = float("nan")

        return ci_lower, ci_upper