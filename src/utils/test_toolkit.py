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

__version__ = "1.3.2"

import base64
import io

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.patches import ConnectionPatch, Rectangle
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
    title: str = "Volatilidad Realizada vs Predicha",
    xlab: str = "Fecha",
    ylab: str = "Varianza",
    label_realized: str = r"Volatilidad Realizada $\left(\sigma^2_R\right)$",
    label_pred: str = r"Volatilidad $\left(\hat{\sigma}^2_t\right)$",
    legend_loc: str = "upper left",
    legend_bbox: tuple = (0.08, 1.0),
    legend_size: float = 13,
    ylim: tuple | None = None,
    y_step: float | None = None,
    zoom_slice: np.ndarray | None = None,
    zoom_title: str = "Zoom: Crisis",
    zoom_bbox: list | tuple = (0.15, 0.30, 0.35, 0.35),
    zoom_pad_frac: float = 0.15,
    pads: tuple = (12, 12, 15),
    label_size: float = 12,
    title_size: float = 12,
    tick_size: float = 11,
    line_width: float = 1.2,
    figsize: tuple = (12, 6),
    fmt: str = "svg",
    dpi: int = 100
) -> None:
    """Plots realized vs predicted variance, with optional zoom; ``zoom_pad_frac`` adds headroom around the zoomed region (rect + inset)."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, rs_variance, color="gray", linewidth=line_width, linestyle="-", label=label_realized)
    ax.plot(dates, pred_variance, color="#0072B2", linewidth=line_width, linestyle="-", label=label_pred)

    pad_x, pad_y, pad_title = pads

    ax.set_title(title, fontsize=title_size, pad=pad_title, loc="center")
    ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
    ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)

    _style_axes(ax, tick_size)
    ax.grid(False)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    if y_step is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))

    ax.legend(frameon=False, fontsize=legend_size, loc=legend_loc, bbox_to_anchor=legend_bbox)

    if zoom_slice is not None:
        ax_ins = ax.inset_axes(zoom_bbox)
        
        ax_ins.plot(dates[zoom_slice], rs_variance[zoom_slice], color="gray", linewidth=line_width, linestyle="-")
        ax_ins.plot(dates[zoom_slice], pred_variance[zoom_slice], color="#0072B2", linewidth=line_width, linestyle="-")
        
        _style_axes(ax_ins, tick_size - 2)
        ax_ins.grid(False)
        
        ax_ins.set_title(zoom_title, fontsize=tick_size, pad=12)
        
        xmin_data = dates[zoom_slice].min()
        xmax_data = dates[zoom_slice].max()
        ymin = min(rs_variance[zoom_slice].min(), pred_variance[zoom_slice].min())
        ymax = max(rs_variance[zoom_slice].max(), pred_variance[zoom_slice].max())

        zoom_pad = (ymax - ymin) * zoom_pad_frac
        rect_ymin = ymin - zoom_pad
        rect_ymax = ymax + zoom_pad
        ax_ins.set_ylim(rect_ymin, rect_ymax)

        xmin_num = mdates.date2num(xmin_data)
        xmax_num = mdates.date2num(xmax_data)
        
        rect = Rectangle(
            (xmin_num, rect_ymin), 
            xmax_num - xmin_num, 
            rect_ymax - rect_ymin, 
            fill=False, 
            edgecolor="black", 
            linewidth=1.0, 
            alpha=1.0 
        )
        ax.add_patch(rect)

        con_top = ConnectionPatch(
            xyA=(1.0, 1.0), 
            coordsA=ax_ins.transAxes,
            xyB=(xmin_num, rect_ymax), 
            coordsB=ax.transData,
            color="black", 
            linewidth=1.0, 
            alpha=1.0, 
            zorder=10
        )
        
        con_bot = ConnectionPatch(
            xyA=(1.0, 0.0), 
            coordsA=ax_ins.transAxes,
            xyB=(xmin_num, rect_ymin), 
            coordsB=ax.transData,
            color="black", 
            linewidth=1.0, 
            alpha=1.0, 
            zorder=10
        )
        
        fig.add_artist(con_top)
        fig.add_artist(con_bot)

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
    """Generates a comparison table of Train (In-Sample) vs Test (Out-of-Sample) metrics."""
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

def _style_axes(ax, tick_size):
    """Apply the toolkit's standard axis style (spines + ticks)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.tick_params(
        axis="x", which="both", bottom=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )
    ax.tick_params(
        axis="y", which="both", left=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )

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