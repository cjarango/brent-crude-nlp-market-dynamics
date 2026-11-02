from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna


class BaseVolatilityModel(ABC):
    """Universal abstract base class for all volatility models in the project."""

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs
        self.is_fitted = False
        self.is_valid = False
        
    @staticmethod
    @abstractmethod
    def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
        """Defines the hyperparameter search space for Optuna.
        
        Args:
            trial: The Optuna trial object.
            
        Returns:
            dict: A dictionary with the suggested hyperparameters.
        """

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Returns the number of past observations needed as context to forecast."""

    @abstractmethod
    def fit(self, y_train: np.ndarray, rs_train: np.ndarray = None) -> None:
        """Fits the model using the training log returns and RS targets.
        
        Args:
            y_train (np.ndarray): Log returns for training.
            rs_train (np.ndarray, optional): Rogers-Satchell targets for training. 
                Required for DL models (for internal validation), optional for others.

        Must set `self.is_fitted = True` and `self.is_valid = True` on success.
        """

    @abstractmethod
    def forecast(self, y_seq: np.ndarray, rs_seq: np.ndarray = None) -> np.ndarray:
        """Generates variance predictions for the validation block.
        
        Args:
            y_seq (np.ndarray): Returns including the context window.
            rs_seq (np.ndarray, optional): Actual RS values for the validation block. 
                Required for DL models (for continuous fine-tuning).
        """
        
    @abstractmethod
    def get_diagnostics(self) -> dict[str, Any]:
        """Retrieves the data necessary to run diagnostic tests.

        Returns:
            Dict[str, Any]: Must include at least:
                - 'converged' (bool): Whether the optimizer converged.
                - 'epochs_run' (int): Number of epochs/iterations run (0 for MLE).
                - 'residuals' (np.ndarray, optional): Standardized residuals.
        """