"""
logger
======

Logging and warning management utilities.

This module configures file-based logging to keep the console output clean 
during long-running experiments (e.g., Optuna trials and Grid Search). 
It also suppresses specific convergence warnings from third-party 
econometric libraries that are expected during hyperparameter search.
"""

__version__ = "1.0.0"

import logging
import warnings
from pathlib import Path

from statsmodels.tools.sm_exceptions import ConvergenceWarning

__all__ = ["setup_logger", "silence_statsmodels_warnings"]


def setup_logger(model_name: str, log_dir: str = "logs") -> logging.Logger:
    """Configures a logger that writes messages to a specific file.

    This function prevents logs from cluttering the standard output 
    (console) by redirecting them to a text file inside the specified 
    directory.

    Args:
        model_name (str): The name of the model or component (used for the filename).
        log_dir (str, optional): The directory where the log file will be saved. 
            Defaults to "logs".

    Returns:
        logging.Logger: A configured logger instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.WARNING)
    
    # Prevent duplicate handlers if the function is called multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path / f"{model_name}.log", mode='a')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger


def silence_statsmodels_warnings() -> None:
    """Suppresses expected warnings from `statsmodels` and `arch` libraries.

    During grid search or Bayesian optimization, many hyperparameter 
    combinations may fail to converge (e.g., due to incompatible inequality 
    constraints). These warnings are captured and logged by the wrapper, 
    making it unnecessary to print them to the console.
    """
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="arch")