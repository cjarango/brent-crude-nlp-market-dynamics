"""
egarch_wrapper
==============

Implementation of the EGARCH model wrapper.

This module provides a standardized wrapper for the Exponential GARCH 
(EGARCH) model using the `arch` library. It includes parameter estimation, 
forecasting, diagnostic extraction, and non-parametric block bootstrap.

Dependencies
------------
numpy, pandas, arch, scipy, tqdm
"""

__version__ = "1.0.0"

from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.special import gamma
from src.models.base import BaseVolatilityModel
from tqdm.auto import tqdm

__all__ = ["EGARCHWrapper"]


class EGARCHWrapper(BaseVolatilityModel):
    """Wrapper for the Exponential GARCH (EGARCH) model."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Initializes the EGARCHWrapper with a dictionary of hyperparameters.

        Args:
            params (Dict[str, Any]): Expected keys: 'p' (int), 'q' (int), 'dist' (str).
            
        Raises:
            ValueError: If 'p' or 'q' are not integers >= 1, or 'dist' is invalid.
        """
        super().__init__(**params)
        
        if 'p' not in params:
            params['p'] = 1
        if 'q' not in params:
            params['q'] = 1
        if 'dist' not in params:
            params['dist'] = 'normal'
            
        self.p = params['p']
        self.q = params['q']
        self.dist = params['dist']

        if not isinstance(self.p, int) or self.p < 1:
            raise ValueError(f"Parameter 'p' must be an integer >= 1. Got {self.p}.")
        if not isinstance(self.q, int) or self.q < 1:
            raise ValueError(f"Parameter 'q' must be an integer >= 1. Got {self.q}.")
            
        valid_dists = ['normal', 't', 'ged']
        if self.dist not in valid_dists:
            raise ValueError(f"Parameter 'dist' must be one of {valid_dists}. Got '{self.dist}'.")

        self.omega: float = 0.0
        self.alphas: np.ndarray = np.array([])
        self.gammas: np.ndarray = np.array([])
        self.betas: np.ndarray = np.array([])
        self.E_abs_z: float = 0.0
        self.last_log_variances: list[float] = []
        self.last_variances: list[float] = []
        self.model_fit = None
        self.y_train: np.ndarray = np.array([])
        
        self.converged: bool = False
        self.is_stationary: bool = False
        self.is_valid: bool = False

    @property
    def context_length(self) -> int:
        """Returns the number of past observations needed as context."""
        return max(self.p, self.q)

    def _compute_expected_abs_z(self) -> float:
        """Computes the expected value of the absolute standardized residuals."""
        if self.dist == 'normal':
            return float(np.sqrt(2 / np.pi))
        elif self.dist == 't':
            nu = float(self.model_fit.params['nu'])
            return float(2 * np.sqrt(nu - 2) * gamma((nu + 1) / 2) / ((nu - 1) * gamma(nu / 2) * np.sqrt(np.pi)))
        elif self.dist == 'ged':
            nu = float(self.model_fit.params['nu'])
            lambda_ged = np.sqrt(2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu))
            return float(2 ** (1 / nu) * gamma(2 / nu) / gamma(1 / nu) * lambda_ged)
        return 0.0

    def fit(self, y_train: np.ndarray) -> None:
        """Fits the EGARCH model using Maximum Likelihood Estimation (MLE).

        Args:
            y_train (np.ndarray): Array of training log returns.
        """
        self.y_train = y_train
        
        am = arch_model(
            y_train, 
            mean='Zero', 
            vol='EGARCH', 
            p=self.p, 
            o=self.q, 
            q=self.q, 
            dist=self.dist
        )
        
        self.model_fit = am.fit(disp='off')
        self.converged = bool(self.model_fit.convergence_flag == 0)
        
        if self.converged:
            self.omega = float(self.model_fit.params['omega'])
            self.alphas = np.array([
                float(self.model_fit.params[f'alpha[{i}]']) 
                for i in range(1, self.q + 1)
            ])
            self.gammas = np.array([
                float(self.model_fit.params[f'gamma[{i}]']) 
                for i in range(1, self.q + 1)
            ])
            self.betas = np.array([
                float(self.model_fit.params[f'beta[{j}]']) 
                for j in range(1, self.p + 1)
            ])
            
            self.is_stationary = bool(np.sum(self.betas) < 1.0)
            self.E_abs_z = self._compute_expected_abs_z()
            
            max_lag = self.context_length
            last_vols = self.model_fit.conditional_volatility[-max_lag:]
            self.last_variances = list(last_vols ** 2)
            self.last_log_variances = list(2 * np.log(last_vols))
        else:
            self.is_stationary = False

        self.is_valid = self.converged and self.is_stationary
        self.is_fitted = True

    def forecast(self, y_target: np.ndarray, initial_context: np.ndarray = None) -> np.ndarray:
        """Generates one-step-ahead variance forecasts.
        
        Args:
            y_target (np.ndarray): The new data to forecast.
            initial_context (np.ndarray, optional): The last context_length returns from 
                the training set to initialize the recursion. Defaults to None.
                
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
            full_history = y_target

        sigma2_history = list(self.last_variances)
        log_sigma2_history = list(self.last_log_variances)

        for t in range(self.context_length, len(full_history)):
            past_returns = full_history[t - self.q : t]
            past_sigmas = np.sqrt(sigma2_history[-self.q:])
            
            z = past_returns / past_sigmas
            z_rev = z[::-1]
            
            mag_term = np.sum(self.alphas * (np.abs(z_rev) - self.E_abs_z))
            asym_term = np.sum(self.gammas * z_rev)
            
            past_log_sigmas_rev = log_sigma2_history[-self.p:][::-1]
            persist_term = np.sum(self.betas * past_log_sigmas_rev)
            
            log_sigma2_t = self.omega + mag_term + asym_term + persist_term
            sigma2_t = np.exp(log_sigma2_t)
            
            log_sigma2_history.append(log_sigma2_t)
            sigma2_history.append(sigma2_t)

        return np.array(sigma2_history[len(self.last_variances):])

    def get_diagnostics(self) -> dict:
        """Returns diagnostic data from the fitted model."""
        if not self.is_fitted:
            return {}
            
        return {
            "params": self.model_fit.params.to_dict() if self.converged else {},
            "residuals": self.model_fit.resid if self.converged else np.array([]),
            "conditional_volatility": self.model_fit.conditional_volatility if self.converged else np.array([]),
            "loglikelihood": self.model_fit.loglikelihood if self.converged else np.nan,
            "converged": self.converged,
            "is_stationary": self.is_stationary
        }

    def _block_bootstrap_ci(self, B: int, block_size: int, verbose: bool) -> tuple[dict, dict]:
        """Calculates confidence intervals using non-parametric block bootstrap."""
        n = len(self.y_train)
        n_blocks = int(np.ceil(n / block_size))
        
        std_resid = self.model_fit.std_resid
        cond_vol = self.model_fit.conditional_volatility
        
        boot_params = {p: [] for p in self.model_fit.params.index}
        
        for _ in tqdm(range(B), desc="Block Bootstrap", leave=False, disable=not verbose):
            block_starts = np.random.randint(0, n - block_size, n_blocks)
            boot_resid = np.concatenate([std_resid[s:s+block_size] for s in block_starts])[:n]
            
            y_boot = boot_resid * cond_vol
            
            try:
                am_boot = arch_model(y_boot, mean='Zero', vol='EGARCH', p=self.p, o=self.q, q=self.q, dist=self.dist)
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

    def get_parameter_table(
        self, 
        B: int = 0, 
        decimals: int = 4, 
        block_size: int = 22,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Returns a DataFrame with coefficients, standard errors, p-values, and CIs.

        Args:
            B (int): Number of block bootstrap iterations. If B=0, uses theoretical CIs. Defaults to 0.
            decimals (int): Number of decimals for rounding. Defaults to 4.
            block_size (int): Block size for the bootstrap. Defaults to 22.
            verbose (bool): If True, displays the bootstrap progress bar. Defaults to False.

        Returns:
            pd.DataFrame: Table with parameters, coefficients, SE, p-values, and CIs.
        """
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