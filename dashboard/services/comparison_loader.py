from functools import lru_cache

import numpy as np
import pandas as pd

from services.model_loader import load_predictions, calculate_metrics


MODEL_ORDER = [
    ("garch", "GARCH"),
    ("egarch", "EGARCH"),
    ("har", "HAR"),
    ("wtd_har", "WTD-HAR"),
    ("gru", "GRU"),
    ("tcn", "TCN"),
    ("markov_tcn", "Markov TCN"),
    ("bigru_tcn", "BiGRU-TCN"),
    ("causal_transformer", "Cau. Trans"),
    ("conv_transformer", "Conv. Trans"),
    ("band_causal_tcn", "Band TCN")
]


METRICS = [
    "QLIKE",
    "CRPS",
    "Pinball (\u03c4 = 0.05)",
    "Pinball (\u03c4 = 0.95)",
    "RMSE",
    "HMSE",
    "HMAE"
]


@lru_cache(maxsize=1)
def load_benchmark():

    rows = []
    reference = None

    for key, name in MODEL_ORDER:

        df = load_predictions(key, "test")

        current = df[["date", "rs_true"]].reset_index(drop=True)

        if reference is None:
            reference = current

        else:
            if not current["date"].equals(reference["date"]):
                raise ValueError(f"Date mismatch: {name}")

            if not np.allclose(
                current["rs_true"],
                reference["rs_true"],
                rtol=1e-6,
                atol=1e-12
            ):
                raise ValueError(f"Target mismatch: {name}")

        metrics = calculate_metrics(df)

        rows.append({
            "Modelo": name,
            **{metric: metrics[metric] for metric in METRICS}
        })

    return (
        pd.DataFrame(rows)
        .sort_values("QLIKE")
        .reset_index(drop=True)
    )
