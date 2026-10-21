# src/eval/scorer.py
import numpy as np
from scipy import stats


class VolatilityScorer:
    """Calculates error and probabilistic metrics for volatility forecasts."""

    def __init__(self) -> None:
        """Initializes the VolatilityScorer."""

    def qlike(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the QLIKE (Quasi-Likelihood) loss.

        The formula is: (y_true / y_pred) - ln(y_true / y_pred) - 1.

        Note:
            An epsilon is added to both numerator and denominator to prevent 
            division by zero and logarithm of zero.

        Args:
            y_true (np.ndarray): Array of true variance values (Rogers-Satchell).
            y_pred (np.ndarray): Array of predicted variance values.

        Returns:
            float: The mean QLIKE loss.
        """
        eps = 1e-8
        ratio = (y_true + eps) / (y_pred + eps)
        return float(np.mean(ratio - np.log(ratio) - 1))

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Squared Error (MSE).

        Args:
            y_true (np.ndarray): Array of true variance values.
            y_pred (np.ndarray): Array of predicted variance values.

        Returns:
            float: The MSE value.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Root Mean Squared Error (RMSE).

        Args:
            y_true (np.ndarray): Array of true variance values.
            y_pred (np.ndarray): Array of predicted variance values.

        Returns:
            float: The RMSE value.
        """
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Absolute Error (MAE).

        Args:
            y_true (np.ndarray): Array of true variance values.
            y_pred (np.ndarray): Array of predicted variance values.

        Returns:
            float: The MAE value.
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    def _get_quantile(
        self, tau: float, sigma_pred: np.ndarray, dist: str, nu: float
    ) -> np.ndarray:
        """Calculates the theoretical quantile for a given probability `tau`.

        Args:
            tau (float): The probability (quantile level) between 0 and 1.
            sigma_pred (np.ndarray): Array of predicted standard deviations.
            dist (str): The distribution name ('normal', 't', 'ged').
            nu (float): Degrees of freedom (for 't') or shape parameter (for 'ged').

        Returns:
            np.ndarray: Array of calculated quantiles.

        Raises:
            ValueError: If `dist` is not one of 'normal', 't', or 'ged'.
        """
        if dist == "normal":
            return stats.norm.ppf(tau, loc=0, scale=sigma_pred)
        elif dist == "t":
            std_t = np.sqrt(nu / (nu - 2))
            return (stats.t.ppf(tau, df=nu) / std_t) * sigma_pred
        elif dist == "ged":
            std_ged = np.sqrt(stats.gamma(1 / nu) / stats.gamma(3 / nu))
            return (stats.gennorm.ppf(tau, beta=nu) / std_ged) * sigma_pred
        else:
            raise ValueError(f"Distribution not supported: {dist}")

    def pinball_loss(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        dist: str = "normal", 
        nu: float | None = None, 
        tau: float = 0.05
    ) -> float:
        """Calculates the Pinball Loss (Quantile Loss) for a specific quantile.

        Args:
            y_true (np.ndarray): Array of true variance values.
            y_pred (np.ndarray): Array of predicted variance values.
            dist (str, optional): Distribution name. Defaults to "normal".
            nu (float, optional): Degrees of freedom or shape parameter. Defaults to None.
            tau (float, optional): Quantile level. Defaults to 0.05.

        Returns:
            float: The mean Pinball loss.
        """
        sigma_pred = np.sqrt(y_pred)
        q_tau = self._get_quantile(tau, sigma_pred, dist, nu)

        diff = y_true - q_tau
        loss = np.maximum(tau * diff, (tau - 1) * diff)
        return float(np.mean(loss))

    def crps(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        dist: str = "normal", 
        nu: float | None = None, 
        n_nodes: int = 20
    ) -> float:
        """Calculates the Continuous Ranked Probability Score (CRPS).

        This implementation uses Gauss-Legendre quadrature to numerically 
        integrate the Pinball loss across all quantiles.

        Args:
            y_true (np.ndarray): Array of true variance values.
            y_pred (np.ndarray): Array of predicted variance values.
            dist (str, optional): Distribution name. Defaults to "normal".
            nu (float, optional): Degrees of freedom or shape parameter. Defaults to None.
            n_nodes (int, optional): Number of nodes for Gauss-Legendre quadrature. 
                Defaults to 20.

        Returns:
            float: The mean CRPS value.
        """
        sigma_pred = np.sqrt(y_pred)

        x, w = np.polynomial.legendre.leggauss(n_nodes)
        taus = (x + 1) / 2
        crps_values = np.zeros(len(y_true))

        for tau, weight in zip(taus, w):
            q_tau = self._get_quantile(tau, sigma_pred, dist, nu)
            diff = y_true - q_tau
            loss = np.maximum(tau * diff, (tau - 1) * diff)
            crps_values += weight * loss * 0.5

        return float(np.mean(crps_values * 2))

    def calculate_all(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        dist: str = "normal", 
        nu: float | None = None
    ) -> dict:
        """Calculates and returns all implemented metrics in a dictionary.

        Args:
            y_true (np.ndarray): Array of true variance values.
            y_pred (np.ndarray): Array of predicted variance values.
            dist (str, optional): Distribution name. Defaults to "normal".
            nu (float, optional): Degrees of freedom or shape parameter. Defaults to None.

        Returns:
            dict: A dictionary containing all calculated metrics.
        """
        return {
            "qlike": self.qlike(y_true, y_pred),
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "crps": self.crps(y_true, y_pred, dist, nu),
            "pinball_05": self.pinball_loss(y_true, y_pred, dist, nu, tau=0.05),
            "pinball_95": self.pinball_loss(y_true, y_pred, dist, nu, tau=0.95),
        }