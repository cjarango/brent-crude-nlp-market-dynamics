from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseVolatilityModel(ABC):
    """Universal abstract base class for all volatility models in the project.

    This interface ensures that the Trainer can interact identically with
    econometric models, Machine Learning models, and Deep Learning models.

    Attributes:
        params (dict): Hyperparameters specific to the model.
        is_fitted (bool): Flag indicating whether the model has been trained.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the model with its specific hyperparameters.

        Args:
            **kwargs: Arbitrary keyword arguments containing the model's 
                hyperparameters (e.g., lags, distribution type, learning rate).
        """
        self.params = kwargs
        self.is_fitted = False

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Returns the number of past observations needed as context to forecast.

        The Trainer uses this property to extract the necessary historical data 
        before the validation block begins.

        Returns:
            int: The number of required past observations (e.g., max(p, q) for GARCH).
        """

    @abstractmethod
    def fit(self, y_train: np.ndarray) -> None:
        """Fits the model using the training log returns.

        Implementation specifics depend on the model family:
            - Econometric: Uses Maximum Likelihood Estimation (MLE) to estimate 
              parameters (e.g., omega, alpha, beta).
            - ML/DL: Uses optimization algorithms to adjust weights.

        Args:
            y_train: Array containing the log returns for the training set.

        Returns:
            None
        """

    @abstractmethod
    def forecast(self, y_seq: np.ndarray) -> np.ndarray:
        """Generates variance predictions for the validation block.

        It is the responsibility of each specific Wrapper to handle its 
        own prediction logic using the provided sequence:
            - ARCH/GARCH: Uses the first `context_length` elements to initialize 
              the recursion, then predicts step-by-step.
            - SVM/MLP: Constructs the lag matrix from `y_seq` and predicts 
              the entire block in a single batch.

        Args:
            y_seq: Array containing the returns to forecast. It must include 
                at least `context_length` past returns as context. Predictions 
                are generated for all elements after the initial context.

        Returns:
            np.ndarray: Array of variance predictions (sigma^2) for the 
                validation block.
        """
        
    @abstractmethod
    def get_diagnostics(self) -> dict[str, Any]:
        """Retrieves the data necessary to run diagnostic tests.

        Returns:
            Dict[str, Any]: A dictionary containing diagnostic data, such as 
                standardized residuals, log-likelihood, and fitted values.
        """