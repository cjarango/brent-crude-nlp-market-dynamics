from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from services.model_loader import load_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"


@lru_cache(maxsize=1)
def load_fhs_bands():

    df_test = pd.read_parquet(
        DATA_PATH / "brent_test.parquet"
    )

    df_test.index = pd.to_datetime(df_test.index)

    close_col = next(
        (
            c for c in df_test.columns
            if c.lower() in {
                "close", "adj_close", "adj close", "price"
            }
        ),
        None
    )

    if close_col is None:
        raise KeyError("Close column not found")

    df = load_predictions("tcn", "test")

    dates = pd.DatetimeIndex(df["date"])

    p_true = (
        df_test.reindex(dates)[close_col]
        .to_numpy(dtype=float)
    )

    variance = df["y_pred"].to_numpy(dtype=float)

    if not np.all(np.isfinite(p_true)):
        raise ValueError("Invalid prices")

    if not np.all(np.isfinite(variance)):
        raise ValueError("Invalid variance")

    if np.any(p_true <= 0):
        raise ValueError("Non-positive prices")

    sigma = np.sqrt(np.maximum(variance, 1e-12))

    r_all = np.empty_like(p_true)
    r_all[0] = 0.0

    r_all[1:] = np.log(
        np.maximum(p_true[1:], 1e-12)
        / np.maximum(p_true[:-1], 1e-12)
    )

    z_cal = r_all / sigma
    z_cal = z_cal[np.isfinite(z_cal)]

    q_low = float(np.quantile(z_cal, 0.025))
    q_high = float(np.quantile(z_cal, 0.975))

    p_prev = np.empty_like(p_true)
    p_prev[0] = p_true[0]
    p_prev[1:] = p_true[:-1]

    r_low = q_low * sigma
    r_high = q_high * sigma

    p_low = p_prev * np.exp(r_low)
    p_high = p_prev * np.exp(r_high)

    covered = (
        (r_all >= r_low)
        & (r_all <= r_high)
    )

    return pd.DataFrame({
        "date": dates,
        "p_true": p_true,
        "p_low": p_low,
        "p_high": p_high,
        "r_true": r_all,
        "r_low": r_low,
        "r_high": r_high,
        "covered": covered
    })
