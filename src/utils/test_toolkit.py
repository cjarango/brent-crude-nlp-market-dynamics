"""
test_toolkit
============

Evaluation and comparison toolkit for final testing.

This module provides functions to visualize the model's fit and compare
in-sample (Train) versus out-of-sample (Test) performance. It generates
publication-ready figures and tables using the project's standard style.

The toolkit is strictly model-agnostic: it only requires the true realized
variance and the predicted variance series.
"""

__version__ = "1.1.0"

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from src.eval.scorer import VolatilityScorer

__all__ = [
    "display_train_test_comparison",
    "plot_volatility_fit",
]


# ===========================================================================
# PUBLIC API
# ===========================================================================

def plot_volatility_fit(
    dates: pd.DatetimeIndex,
    rs_variance: np.ndarray,
    pred_variance: np.ndarray,
    title: str = "Volatilidad Realizada vs Condicional",
    xlab: str = "Fecha",
    ylab: str = "Volatilidad Diaria",
    label_realized: str = "Realizada (RS)",
    label_pred: str = "Condicional del Modelo",
    legend_loc: str = "upper left",
    legend_size: float = 12,
    pads: tuple = (9, 9, 12),
    label_size: float = 12,
    title_size: float = 12,
    tick_size: float = 11,
    line_width: float = 0.8,
    figsize: tuple = (12, 5),
    fmt: str = "svg",
    dpi: int = 100
) -> None:
    """Plots the realized variance against the predicted variance.
    
    Args:
        dates (pd.DatetimeIndex): Dates for the x-axis.
        rs_variance (np.ndarray): True realized variance values.
        pred_variance (np.ndarray): Predicted variance values.
        ... (other styling args)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Se aplica la raíz cuadrada internamente para graficar volatilidad (std)
    ax.plot(dates, np.sqrt(rs_variance), color="gray", alpha=0.6, linewidth=line_width, label=label_realized)
    ax.plot(dates, np.sqrt(pred_variance), color="#0072B2", linewidth=1.2, label=label_pred)

    pad_x, pad_y, pad_title = pads

    ax.set_title(title, fontsize=title_size, pad=pad_title, loc="center")
    ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
    ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=tick_size, direction="out", length=4, width=0.8, colors="black")
    ax.tick_params(axis="y", which="both", left=True, right=False, labelsize=tick_size, direction="out", length=4, width=0.8, colors="black")

    ax.margins(x=0.02, y=0.10)
    ax.legend(frameon=False, fontsize=legend_size, loc=legend_loc)
    ax.grid(False)
    plt.tight_layout()

    if fmt == "svg":
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        display(HTML(f'<div style="display: flex; justify-content: center; width: 100%;">{buf.getvalue().decode("utf-8")}</div>'))
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        display(HTML(f'<div style="text-align: center; width: 100%;"><img src="data:image/png;base64,{base64.b64encode(buf.getbuffer()).decode("ascii")}"></div>'))


def display_train_test_comparison(
    rs_train: np.ndarray,
    preds_train: np.ndarray,
    rs_test: np.ndarray,
    preds_test: np.ndarray,
    dist: str = "normal",
    nu: float | None = None,
    decimals: int = 4,
    column_titles: tuple = ("Métrica", "Entrenamiento (In-Sample)", "Prueba (Out-of-Sample)", "Diferencia Absoluta"),
    metrics: list | None = None,
    show_diff: bool = True
) -> None:
    """Generates a comparison table of Train (In-Sample) vs Test (Out-of-Sample) metrics.

    Args:
        rs_train (np.ndarray): True realized variance for training.
        preds_train (np.ndarray): Predicted variance for training.
        rs_test (np.ndarray): True realized variance for testing.
        preds_test (np.ndarray): Predicted variance for testing.
        dist (str): Distribution used (for CRPS/Pinball). Defaults to "normal".
        nu (float, optional): Shape parameter. Defaults to None.
        ... (other formatting args)
    """
    expected_cols = 4 if show_diff else 3
    if len(column_titles) != expected_cols:
        raise ValueError(f"column_titles must be a tuple of exactly {expected_cols} strings.")

    scorer = VolatilityScorer()
    
    train_metrics = scorer.calculate_all(y_true=rs_train, y_pred=preds_train, dist=dist, nu=nu)
    test_metrics = scorer.calculate_all(y_true=rs_test, y_pred=preds_test, dist=dist, nu=nu)

    metric_labels = {
        "qlike": "QLIKE",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "hmse": "HMSE",
        "hmae": "HMAE",
        "crps": "CRPS",
        "pinball_05": "Pinball (τ=0.05)",
        "pinball_95": "Pinball (τ=0.95)"
    }

    if metrics is None:
        metrics = list(train_metrics.keys())

    labels = [metric_labels.get(m, m) for m in metrics]
    train_vals = [train_metrics[m] for m in metrics]
    test_vals = [test_metrics[m] for m in metrics]

    data = {
        column_titles[0]: labels,
        column_titles[1]: [f"{v:.{decimals}f}" for v in train_vals],
        column_titles[2]: [f"{v:.{decimals}f}" for v in test_vals]
    }

    if show_diff:
        diff_vals = [t - tr for t, tr in zip(test_vals, train_vals)]
        data[column_titles[3]] = [f"{v:+.{decimals}f}" for v in diff_vals]

    df = pd.DataFrame(data)

    styled_table = (
        df.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules())
    )

    _render_styled(styled_table)


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _table_style_rules() -> list:
    return [
        {'selector': '', 'props': [('margin-left', 'auto'), ('margin-right', 'auto'), ('width', 'auto'), ('border-collapse', 'collapse')]},
        {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', '#f2f2f2'), ('color', 'black'), ('font-weight', 'bold'), ('border', '1px solid black'), ('padding', '10px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid black'), ('padding', '10px')]},
        {'selector': 'th:first-child', 'props': [('text-align', 'left')]},
        {'selector': 'td:first-child', 'props': [('text-align', 'left')]}
    ]


def _render_styled(styler) -> None:
    display(
        HTML(
            "<div style='text-align: center; width: 100%;'>"
            + styler.to_html()
            + "</div>"
        )
    )