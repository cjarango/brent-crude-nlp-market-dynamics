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
import pywt
from scipy import stats
from tqdm.auto import tqdm

from src.models.base import BaseVolatilityModel

__all__ = [
    "HARQRFitResult",
    "HARWTDWrapper",
    "compute_causal_wtd_series",
    "default_wtd_window",
    "min_wtd_window_length",
]

_DWT_MODE = "symmetric"
_PARAM_NAMES = ["const", "RV_d", "RV_w", "RV_m"]


def min_wtd_window_length(wavelet: str, level: int) -> int:
    filter_len = pywt.Wavelet(wavelet).dec_len
    return max(1, (filter_len - 1) * (2 ** level))


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
    arr = np.asarray(prefix, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0

    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level, mode=_DWT_MODE)
    finest_detail = coeffs[-1]
    sigma = float(np.median(np.abs(finest_detail))) / 0.6745 if len(finest_detail) > 0 else 0.0

    if sigma > 0.0 and n > 1:
        thresh = sigma * np.sqrt(2.0 * np.log(n))
        coeffs_denoised = [coeffs[0]] + [
            pywt.threshold(c, value=thresh, mode=threshold_mode) for c in coeffs[1:]
        ]
    else:
        coeffs_denoised = coeffs

    rec = pywt.waverec(coeffs_denoised, wavelet=wavelet, mode=_DWT_MODE)
    return float(rec[n - 1])


def compute_causal_wtd_series(
    series: np.ndarray,
    wavelet: str = "sym4",
    level: int = 3,
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


class HARQRFitResult:
    """Stores QR regression estimates and HAC/Newey-West inference for HARWTDWrapper."""

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
        self.dof = max(1, n_obs - n_features)

        rss = float(np.sum(self.resid ** 2))
        ss_tot = float(np.sum((self.y_vec - np.mean(self.y_vec)) ** 2))
        self.rsquared = float(1.0 - rss / ss_tot) if ss_tot > 0.0 else 0.0

        _, r_mat = np.linalg.qr(self.x_mat, mode="reduced")

        if cov_type == "hac":
            cov_beta = self._newey_west_cov(self.x_mat, self.resid, r_mat, max_lags=hac_lags)
        elif cov_type == "qr":
            s2 = rss / self.dof
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
        p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=self.dof)
        self.pvalues = pd.Series(np.clip(p_vals, 0.0, 1.0), index=self.param_names)

    @staticmethod
    def _newey_west_cov(
        x_mat: np.ndarray,
        resid: np.ndarray,
        r_mat: np.ndarray,
        max_lags: int | None = None,
        small_sample_adj: bool = True,
    ) -> np.ndarray:
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

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        q_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=self.dof))
        coef = self.params.to_numpy(dtype=np.float64)
        se = self.bse.to_numpy(dtype=np.float64)
        return pd.DataFrame(
            {"lower": coef - q_crit * se, "upper": coef + q_crit * se},
            index=self.param_names,
        )


class HARWTDWrapper(BaseVolatilityModel):
    """Causal WTD-HAR model estimated via QR decomposition with HAC inference."""

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> dict:
        return {
            "lag_weekly": trial.suggest_categorical("lag_weekly", [5, 7]),
            "lag_monthly": trial.suggest_categorical("lag_monthly", [22, 30]),
            "wtd_level": trial.suggest_categorical("wtd_level", [1, 2, 3]),
            "seed": 42,
        }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lag_weekly: int = int(kwargs.get("lag_weekly", 5))
        self.lag_monthly: int = int(kwargs.get("lag_monthly", 22))
        self.seed: int = int(kwargs.get("seed", 42))
        self.wavelet: str = str(kwargs.get("wavelet", "sym4"))
        self.wtd_level: int = int(kwargs.get("wtd_level", kwargs.get("level", 3)))
        self.level: int = self.wtd_level
        raw_decomp = kwargs.get("decomp_window", None)
        self.decomp_window: int = int(
            raw_decomp if raw_decomp is not None else default_wtd_window(self.wavelet, self.wtd_level)
        )
        self.threshold_mode: str = str(kwargs.get("threshold_mode", "soft"))
        self.cov_type: str = str(kwargs.get("cov_type", "hac"))
        self.hac_lags: int | None = kwargs.get("hac_lags", None)

        min_len = min_wtd_window_length(self.wavelet, self.wtd_level)
        if self.decomp_window < min_len:
            raise ValueError(
                f"decomp_window={self.decomp_window} < {min_len} "
                f"for wavelet='{self.wavelet}' and wtd_level={self.wtd_level}."
            )

        self.model_fit: HARQRFitResult | None = None
        self.y_train_history: np.ndarray = np.array([], dtype=np.float64)
        self.rs_train_history: np.ndarray = np.array([], dtype=np.float64)
        self._wtd_cache: dict[int, float] = {}

        self.converged: bool = True
        self.is_fitted: bool = False
        self.is_valid: bool = False

    @property
    def context_length(self) -> int:
        return self.lag_monthly

    @staticmethod
    def _solve_qr(x_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_mat, dtype=np.float64)
        y_arr = np.asarray(y_vec, dtype=np.float64).reshape(-1)
        if len(x_arr) == 0:
            return np.zeros(len(_PARAM_NAMES), dtype=np.float64)
        q_mat, r_mat = np.linalg.qr(x_arr, mode="reduced")
        diag_r = np.abs(np.diag(r_mat))
        if len(diag_r) == 0 or diag_r.min() < 1e-10:
            beta, _, _, _ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
            return np.asarray(beta, dtype=np.float64)
        return np.linalg.solve(r_mat, q_mat.T @ y_arr)

    def _cached_wtd_for_prefix(self, arr: np.ndarray, offset: int = 0, use_cache: bool = True) -> np.ndarray:
        n_samples = len(arr)
        min_len = min_wtd_window_length(self.wavelet, self.wtd_level)
        if n_samples < min_len:
            raise ValueError(
                f"Series length ({n_samples}) < min_wtd_window_length({self.wavelet}, {self.wtd_level})={min_len}."
            )

        out = np.zeros(n_samples, dtype=np.float64)
        for t in range(n_samples):
            global_t = offset + t
            if use_cache and global_t in self._wtd_cache:
                out[t] = self._wtd_cache[global_t]
            else:
                start = max(0, t + 1 - self.decomp_window)
                window = arr[start : t + 1]
                if len(window) < min_len:
                    window = np.pad(window, (min_len - len(window), 0), mode="edge")
                val = _step_causal_wtd(
                    window,
                    wavelet=self.wavelet,
                    level=self.wtd_level,
                    threshold_mode=self.threshold_mode,
                )
                if use_cache:
                    self._wtd_cache[global_t] = val
                out[t] = val
        return out

    def _build_har_design(self, wtd_rv_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Builds causal HAR design matrix [1, RV_d, RV_w, RV_m] aligned from `lag_monthly`."""
        s = pd.Series(np.asarray(wtd_rv_series, dtype=np.float64))
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

    def _build_design_matrix(
        self,
        y_series: np.ndarray,
        use_cache: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(y_series, dtype=np.float64).flatten()
        if len(arr) <= self.lag_monthly:
            raise ValueError(
                f"Series length ({len(arr)}) must be greater than lag_monthly ({self.lag_monthly})."
            )

        # Matches hybrid WTD-HAR feature construction: causal WTD applied to squared returns
        rv_raw = arr ** 2
        wtd_rv = self._cached_wtd_for_prefix(rv_raw, offset=0, use_cache=use_cache)
        return self._build_har_design(wtd_rv)

    def fit(self, y_train: np.ndarray, rs_train: np.ndarray | None = None) -> None:
        if rs_train is None:
            raise ValueError("rs_train must be provided.")

        y_arr = np.asarray(y_train, dtype=np.float64)
        rs_arr = np.asarray(rs_train, dtype=np.float64)
        self.y_train_history = y_arr.copy()
        self.rs_train_history = rs_arr.copy()
        self._wtd_cache.clear()

        try:
            x_mat, valid_idx = self._build_design_matrix(y_arr, use_cache=True)
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
            raise RuntimeError(f"WTD-HAR QR fitting failed: {e}") from e

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
                x_all, _ = self._build_design_matrix(y_with_curr, use_cache=True)
                x_pred = x_all[-1]

                x_hist, valid_hist = self._build_design_matrix(hist_y_arr, use_cache=True)
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
                    print(f"[DEBUG] WTD-QR failed at step {i}: {e}")

            history_y.append(float(y_seq_arr[i]))
            history_rs.append(float(rs_seq_arr[i]))

            if debug:
                print(f"[DEBUG] pred i={i:03d} | y={y_seq[i]:+.6f} | pred={predictions[-1]:.6e}")

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
                "wavelet": self.wavelet,
                "wtd_level": self.wtd_level,
                "level": self.level,
                "decomp_window": self.decomp_window,
                "threshold_mode": self.threshold_mode,
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
            "epochs_run": 0,
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

        for _ in tqdm(range(B), desc="Block Bootstrap (WTD-QR)", leave=False, disable=not verbose):
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