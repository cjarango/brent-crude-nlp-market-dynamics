"""
scorer
======

Calculates error and probabilistic metrics for volatility forecasts.
"""

import numpy as np
from scipy import stats

__all__ = ["VolatilityScorer"]


class VolatilityScorer:
    """Calculates error and probabilistic metrics for volatility forecasts."""

    def __init__(self) -> None:
        """Initializes the VolatilityScorer."""
        pass

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Validates that inputs are positive and finite to prevent silent NaNs.

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.
        """
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Inputs contain NaN or Inf values.")
        if np.any(y_true <= 0) or np.any(y_pred <= 0):
            raise ValueError("Volatility metrics require strictly positive variances.")

    def qlike(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the QLIKE loss.

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.

        Returns:
            float: Mean QLIKE loss.
        """
        self._validate_inputs(y_true, y_pred)
        eps = 1e-8
        ratio = (y_true + eps) / (y_pred + eps)
        return float(np.mean(ratio - np.log(ratio) - 1))

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Squared Error (MSE).

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.

        Returns:
            float: MSE value.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Root Mean Squared Error (RMSE).

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.

        Returns:
            float: RMSE value.
        """
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Absolute Error (MAE).

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.

        Returns:
            float: MAE value.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    def hmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Heteroskedasticity-Adjusted MSE (HMSE).

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.

        Returns:
            float: HMSE value.
        """
        self._validate_inputs(y_true, y_pred)
        eps = 1e-8
        ratio = (y_true + eps) / (y_pred + eps)
        return float(np.mean((ratio - 1) ** 2))

    def hmae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Heteroskedasticity-Adjusted MAE (HMAE).

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.

        Returns:
            float: HMAE value.
        """
        self._validate_inputs(y_true, y_pred)
        eps = 1e-8
        ratio = (y_true + eps) / (y_pred + eps)
        return float(np.mean(np.abs(ratio - 1)))

    def _get_quantile(
        self, tau: float, sigma_pred: np.ndarray, dist: str, nu: float | None
    ) -> np.ndarray:
        """Calculates the theoretical quantile on the volatility scale.

        Args:
            tau (float): Quantile level (between 0 and 1).
            sigma_pred (np.ndarray): Predicted volatility (standard deviation).
            dist (str): Distribution family ('normal', 't', 'ged').
            nu (float | None): Shape parameter for t or ged.

        Returns:
            np.ndarray: Quantiles scaled by predicted volatility.
        """
        if dist == "normal":
            return stats.norm.ppf(tau, loc=0, scale=sigma_pred)
        elif dist == "t":
            if nu is None or nu <= 2:
                raise ValueError("Student-t requires nu > 2 for finite variance.")
            std_t = np.sqrt(nu / (nu - 2))
            return (stats.t.ppf(tau, df=nu) / std_t) * sigma_pred
        elif dist == "ged":
            if nu is None or nu <= 0:
                raise ValueError("GED requires nu > 0.")
            # Correct standardization: sqrt( Gamma(3/nu) / Gamma(1/nu) )
            std_ged = np.sqrt(stats.gamma(3 / nu) / stats.gamma(1 / nu))
            return (stats.gennorm.ppf(tau, beta=nu) / std_ged) * sigma_pred
        else:
            raise ValueError(f"Distribution not supported: {dist}")

    def pinball_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dist: str = "normal",
        nu: float | None = None,
        tau: float = 0.05,
    ) -> float:
        """Calculates the Pinball Loss for a specific quantile.

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.
            dist (str): Distribution family ('normal', 't', 'ged').
            nu (float | None): Shape parameter for t or ged.
            tau (float): Quantile level (default 0.05).

        Returns:
            float: Mean Pinball loss.
        """
        self._validate_inputs(y_true, y_pred)
        
        sigma_true = np.sqrt(y_true)
        sigma_pred = np.sqrt(y_pred)
        
        q_tau = self._get_quantile(tau, sigma_pred, dist, nu)
        
        diff = sigma_true - q_tau
        loss = np.maximum(tau * diff, (tau - 1) * diff)
        return float(np.mean(loss))

    def crps(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dist: str = "normal",
        nu: float | None = None,
        n_nodes: int = 20,
    ) -> float:
        """Calculates the CRPS using Gauss-Legendre quadrature.

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.
            dist (str): Distribution family ('normal', 't', 'ged').
            nu (float | None): Shape parameter for t or ged.
            n_nodes (int): Number of quadrature nodes.

        Returns:
            float: Mean CRPS value.
        """
        self._validate_inputs(y_true, y_pred)
        
        sigma_true = np.sqrt(y_true)
        sigma_pred = np.sqrt(y_pred)

        x, w = np.polynomial.legendre.leggauss(n_nodes)
        taus = (x + 1) / 2
        crps_values = np.zeros(len(sigma_true))

        for tau, weight in zip(taus, w):
            q_tau = self._get_quantile(tau, sigma_pred, dist, nu)
            diff = sigma_true - q_tau
            loss = np.maximum(tau * diff, (tau - 1) * diff)
            crps_values += weight * loss * 0.5

        return float(np.mean(crps_values * 2))

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dist: str = "normal",
        nu: float | None = None,
    ) -> dict:
        """Calculates and returns all implemented metrics in a dictionary.

        Args:
            y_true (np.ndarray): Realized variance targets.
            y_pred (np.ndarray): Predicted variances.
            dist (str): Distribution family ('normal', 't', 'ged').
            nu (float | None): Shape parameter for t or ged.

        Returns:
            dict: Dictionary with all metric values.
        """
        return {
            "qlike": self.qlike(y_true, y_pred),
            "crps": self.crps(y_true, y_pred, dist, nu),
            "pinball_05": self.pinball_loss(y_true, y_pred, dist, nu, tau=0.05),
            "pinball_95": self.pinball_loss(y_true, y_pred, dist, nu, tau=0.95),
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "hmse": self.hmse(y_true, y_pred),
            "hmae": self.hmae(y_true, y_pred),
        }