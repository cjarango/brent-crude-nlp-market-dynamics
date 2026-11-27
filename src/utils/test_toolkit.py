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

__version__ = "1.9.0"

import base64
import io
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy import stats
from src.eval.scorer import VolatilityScorer

__all__ = [
    "cliffs_delta",
    "diebold_mariano_test",
    "display_models_comparison",
    "display_train_test_comparison",
    "load_test_predictions_matrix",
    "model_confidence_set",
    "plot_cliffs_dm_heatmap",
    "plot_mcs_bars",
    "plot_price_prediction_bands",
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
    dpi: int = 100,
) -> None:
    """Plots realized vs predicted variance, with optional zoom."""
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
            alpha=1.0,
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
            zorder=10,
        )

        con_bot = ConnectionPatch(
            xyA=(1.0, 0.0),
            coordsA=ax_ins.transAxes,
            xyB=(xmin_num, rect_ymin),
            coordsB=ax.transData,
            color="black",
            linewidth=1.0,
            alpha=1.0,
            zorder=10,
        )

        fig.add_artist(con_top)
        fig.add_artist(con_bot)

    plt.tight_layout()
    _display_figure(fig, fmt=fmt, dpi=dpi)


def plot_price_prediction_bands(
    dates: pd.DatetimeIndex,
    prices_true: np.ndarray,
    pred_volatility: np.ndarray,
    *,
    is_variance: bool = False,
    returns_true: np.ndarray | None = None,
    z_train: np.ndarray | None = None,
    r_train: np.ndarray | None = None,
    var_pred_train: np.ndarray | None = None,
    alpha: float = 0.05,
    method: str = "fhs",
    mu: float = 0.0,
    show_train_band: bool = False,
    train_mask: np.ndarray | None = None,
    title: str = "Trayectoria de Precio y Bandas de Predicción",
    xlab: str = "Fecha",
    ylab: str = "Precio (USD)",
    label_real: str = "Precio Real",
    label_band: str | None = None,
    legend_loc: str = "upper left",
    legend_bbox: tuple = (0.01, 0.99),
    legend_size: float = 12,
    ylim: tuple | None = None,
    y_step: float | None = None,
    pads: tuple = (12, 12, 15),
    label_size: float = 12,
    title_size: float = 12,
    tick_size: float = 11,
    line_width: float = 1.2,
    figsize: tuple = (12, 6),
    fmt: str = "svg",
    dpi: int = 100,
) -> pd.DataFrame:
    """Plots price trajectory with FHS or Student-t prediction bands derived from volatility."""
    dates_idx = pd.DatetimeIndex(dates)
    p_true = np.asarray(prices_true, dtype=float)
    sigma_all = np.asarray(pred_volatility, dtype=float)
    if is_variance:
        sigma_all = np.sqrt(np.maximum(sigma_all, 1e-12))
    else:
        sigma_all = np.maximum(sigma_all, 1e-12)

    if returns_true is not None:
        r_all = np.asarray(returns_true, dtype=float)
    else:
        r_all = np.empty_like(p_true)
        r_all[0] = 0.0
        r_all[1:] = np.log(np.maximum(p_true[1:], 1e-12) / np.maximum(p_true[:-1], 1e-12))

    if z_train is not None:
        z_cal = np.asarray(z_train, dtype=float)
    elif r_train is not None and var_pred_train is not None:
        sig_tr = np.sqrt(np.maximum(np.asarray(var_pred_train, dtype=float), 1e-12))
        r_tr = np.asarray(r_train, dtype=float)
        z_cal = r_tr / sig_tr
    elif train_mask is not None:
        z_cal = r_all[train_mask] / sigma_all[train_mask]
    else:
        z_cal = r_all / sigma_all

    z_cal = z_cal[np.isfinite(z_cal)]

    if method == "fhs":
        q_low = float(np.quantile(z_cal, alpha / 2.0))
        q_high = float(np.quantile(z_cal, 1.0 - alpha / 2.0))
    elif method == "student_t":
        nu, _, _ = stats.t.fit(z_cal, floc=0.0, fscale=1.0)
        nu = max(float(nu), 2.05)
        scale_unit = np.sqrt((nu - 2.0) / nu)
        q_low = float(stats.t.ppf(alpha / 2.0, df=nu) * scale_unit)
        q_high = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu) * scale_unit)
    else:
        raise ValueError("method must be 'fhs' or 'student_t'.")

    p_prev = np.empty_like(p_true)
    p_prev[0] = p_true[0]
    p_prev[1:] = p_true[:-1]

    r_low = mu + q_low * sigma_all
    r_high = mu + q_high * sigma_all
    p_low = p_prev * np.exp(r_low)
    p_high = p_prev * np.exp(r_high)
    covered = (r_all >= r_low) & (r_all <= r_high)

    fig, ax = plt.subplots(figsize=figsize)

    band_label = label_band or f"Banda {method.upper()} ({int((1.0 - alpha) * 100)}%)"

    if train_mask is not None and show_train_band:
        ax.fill_between(
            dates_idx[train_mask],
            p_low[train_mask],
            p_high[train_mask],
            color="#0072B2",
            alpha=0.14,
            label=band_label,
        )
        test_mask = ~train_mask
        ax.fill_between(
            dates_idx[test_mask],
            p_low[test_mask],
            p_high[test_mask],
            color="#0072B2",
            alpha=0.28,
        )
        if np.any(test_mask):
            ax.axvline(dates_idx[test_mask][0], color="black", linestyle="--", linewidth=1.0)
    elif train_mask is not None:
        test_mask = ~train_mask
        ax.fill_between(
            dates_idx[test_mask],
            p_low[test_mask],
            p_high[test_mask],
            color="#0072B2",
            alpha=0.25,
            label=band_label,
        )
    else:
        ax.fill_between(
            dates_idx,
            p_low,
            p_high,
            color="#0072B2",
            alpha=0.25,
            label=band_label,
        )

    ax.plot(
        dates_idx,
        p_true,
        color="gray",
        linewidth=line_width,
        linestyle="-",
        label=label_real,
    )

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

    plt.tight_layout()
    _display_figure(fig, fmt=fmt, dpi=dpi)

    return pd.DataFrame({
        "date": dates_idx,
        "p_true": p_true,
        "p_low": p_low,
        "p_high": p_high,
        "r_true": r_all,
        "r_low": r_low,
        "r_high": r_high,
        "covered": covered,
    })


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
    show_diff: bool = True,
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
        "pinball_95": "Pinball (τ=0.95)",
    }

    if metrics is None:
        metrics = list(train_metrics.keys())

    labels = [metric_labels.get(m, m) for m in metrics]
    train_vals = [train_metrics[m] for m in metrics]
    test_vals = [test_metrics[m] for m in metrics]

    data = {
        column_titles[0]: labels,
        column_titles[1]: [f"{v:.{decimals}f}" for v in train_vals],
        column_titles[2]: [f"{v:.{decimals}f}" for v in test_vals],
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


def display_models_comparison(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    model_names: list[str] | None = None,
    metrics: tuple[str, ...] = ("qlike", "crps", "rmse", "hmse", "hmae"),
    sort_by: str = "qlike",
    dist: str = "normal",
    nu: float | None = None,
    decimals: int = 4,
    first_col_title: str = "Modelo",
) -> None:
    """Generates a styled comparison table across models sorted by QLIKE (or selected metric)."""
    y_arr = np.asarray(y_true, dtype=float).ravel()
    preds_arr = np.asarray(preds_matrix, dtype=float)
    if preds_arr.ndim != 2:
        raise ValueError("preds_matrix must be a 2D array of shape (n_obs, n_models).")

    n_obs, n_models = preds_arr.shape
    if len(y_arr) != n_obs or n_obs == 0:
        raise ValueError("y_true must be non-empty and match preds_matrix rows.")

    if model_names is None:
        names = [f"model_{idx}" for idx in range(n_models)]
    else:
        if len(model_names) != n_models:
            raise ValueError("model_names length must match preds_matrix columns.")
        names = list(model_names)

    metric_labels = {
        "qlike": "QLIKE",
        "crps": "CRPS",
        "pinball_05": "Pinball (τ=0.05)",
        "pinball_95": "Pinball (τ=0.95)",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "hmse": "HMSE",
        "hmae": "HMAE",
    }

    scorer = VolatilityScorer()
    raw_rows: list[dict] = []

    for col_idx, name in enumerate(names):
        calc = scorer.calculate_all(y_true=y_arr, y_pred=preds_arr[:, col_idx], dist=dist, nu=nu)
        for m in metrics:
            if m not in calc:
                raise ValueError(f"Metric '{m}' is not available in VolatilityScorer.")
        if sort_by not in calc:
            raise ValueError(f"sort_by metric '{sort_by}' is not available in VolatilityScorer.")
        raw_rows.append({"name": name, "calc": calc})

    raw_rows.sort(key=lambda item: float(item["calc"][sort_by]))

    formatted_rows: list[dict] = []
    for item in raw_rows:
        row = {first_col_title: item["name"]}
        for m in metrics:
            col_label = metric_labels.get(m, m)
            row[col_label] = f"{item['calc'][m]:.{decimals}f}"
        formatted_rows.append(row)

    df = pd.DataFrame(formatted_rows)
    styled_table = (
        df.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules())
    )
    _render_styled(styled_table)


def load_test_predictions_matrix(
    csv_paths: list[str | Path],
    model_names: list[str] | None = None,
) -> dict:
    """Loads test prediction CSVs (date, rs_true, y_pred), aligns by date, and builds preds_matrix."""
    if not csv_paths or len(csv_paths) < 2:
        raise ValueError("csv_paths must contain at least two CSV file paths.")

    if model_names is None:
        names = [f"model_{idx}" for idx in range(len(csv_paths))]
    else:
        if len(model_names) != len(csv_paths):
            raise ValueError("model_names must have the same length as csv_paths.")
        names = list(model_names)

    merged_df: pd.DataFrame | None = None

    for idx, (path, name) in enumerate(zip(csv_paths, names)):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Prediction CSV not found: {file_path}")

        df = pd.read_csv(file_path)
        required_cols = {"date", "rs_true", "y_pred"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns in {file_path}: {required_cols - set(df.columns)}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"])

        if idx == 0:
            merged_df = df[["date", "rs_true", "y_pred"]].rename(columns={"y_pred": name})
        else:
            right_df = df[["date", "rs_true", "y_pred"]].rename(
                columns={"rs_true": f"_rs_{idx}", "y_pred": name}
            )
            merged_df = pd.merge(merged_df, right_df, on="date", how="inner")
            if not np.allclose(
                merged_df["rs_true"].to_numpy(dtype=float),
                merged_df[f"_rs_{idx}"].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-12,
                equal_nan=False,
            ):
                raise ValueError(f"rs_true mismatch between base series and {file_path}.")
            merged_df = merged_df.drop(columns=[f"_rs_{idx}"])

    if merged_df is None or merged_df.empty:
        raise ValueError("No overlapping dates found across prediction CSVs.")

    dates = pd.DatetimeIndex(merged_df["date"])
    y_true = merged_df["rs_true"].to_numpy(dtype=float)
    preds_matrix = merged_df[names].to_numpy(dtype=float)

    return {
        "dates": dates,
        "y_true": y_true,
        "preds_matrix": preds_matrix,
        "model_names": names,
    }


def diebold_mariano_test(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    model_names: list[str] | None = None,
    metric: str = "qlike",
    h: int = 1,
    lag: int | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    block_len: int | None = None,
    bootstrap_type: str = "stationary",
    seed: int = 42,
) -> dict:
    """Pairwise two-sided Diebold-Mariano test across models from y_true and preds_matrix (T, M)."""
    losses_by_model = _compute_model_losses(
        y_true=y_true,
        preds_matrix=preds_matrix,
        model_names=model_names,
        metric=metric,
    )
    names = list(losses_by_model.keys())
    n = len(losses_by_model[names[0]])
    eff_block_len = _resolve_block_len(n=n, block_len=block_len)
    hac_lag = lag if lag is not None else max(1, h + int(np.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))

    pairwise_results: dict[str, dict] = {}
    pair_idx = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m_i, m_j = names[i], names[j]
            d = losses_by_model[m_i] - losses_by_model[m_j]
            mean_diff = float(np.mean(d))
            lrv = _newey_west_long_run_var(d, lag=hac_lag)
            se_dm = np.sqrt(lrv / n)
            dm_stat = mean_diff / se_dm if se_dm > 0.0 else 0.0

            rng = np.random.default_rng(seed + pair_idx)
            boot_indices = _generate_block_bootstrap_batch(
                n=n,
                n_bootstrap=n_bootstrap,
                block_len=eff_block_len,
                bootstrap_type=bootstrap_type,
                rng=rng,
            )

            d_boot_mat = d[boot_indices]
            boot_means = np.mean(d_boot_mat, axis=1)
            boot_dm = np.empty(n_bootstrap, dtype=float)

            for b in range(n_bootstrap):
                lrv_b = _newey_west_long_run_var(d_boot_mat[b], lag=hac_lag)
                se_b = np.sqrt(lrv_b / n)
                boot_dm[b] = ((boot_means[b] - mean_diff) / se_b) if se_b > 0.0 else 0.0

            p_value = float(np.mean(np.abs(boot_dm) >= abs(dm_stat)))
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            excludes_zero = bool((float(ci_low) > 0.0) or (float(ci_high) < 0.0))

            pairwise_results[f"{m_i}_vs_{m_j}"] = {
                "model_a": m_i,
                "model_b": m_j,
                "n": int(n),
                "mean_loss_diff": mean_diff,
                "dm_stat": float(dm_stat),
                "p_value": p_value,
                "significant": bool(p_value < alpha),
                "excludes_zero": excludes_zero,
                "alpha": float(alpha),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "n_bootstrap": int(n_bootstrap),
                "block_len": int(eff_block_len),
                "bootstrap_type": bootstrap_type,
                "hac_lag": int(hac_lag),
            }
            pair_idx += 1

    return {
        "metric": metric,
        "models": names,
        "pairwise": pairwise_results,
    }


def cliffs_delta(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    model_names: list[str] | None = None,
    metric: str = "qlike",
    threshold: float = 0.33,
) -> dict:
    """Computes Cliff's delta effect size across model pairs without bootstrap."""
    losses_by_model = _compute_model_losses(
        y_true=y_true,
        preds_matrix=preds_matrix,
        model_names=model_names,
        metric=metric,
    )
    names = list(losses_by_model.keys())

    def _delta(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
        diff_matrix = a_vec[:, None] - b_vec[None, :]
        gt = float(np.sum(diff_matrix > 0.0))
        lt = float(np.sum(diff_matrix < 0.0))
        return (gt - lt) / float(len(a_vec) * len(b_vec))

    pairwise_results: dict[str, dict] = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m_i, m_j = names[i], names[j]
            x_arr = losses_by_model[m_i]
            y_arr = losses_by_model[m_j]
            n = len(x_arr)
            delta_val = _delta(x_arr, y_arr)
            abs_d = abs(delta_val)

            if abs_d < 0.147:
                magnitude = "negligible"
            elif abs_d < 0.33:
                magnitude = "small"
            elif abs_d < 0.474:
                magnitude = "medium"
            else:
                magnitude = "large"

            pairwise_results[f"{m_i}_vs_{m_j}"] = {
                "model_a": m_i,
                "model_b": m_j,
                "n": int(n),
                "cliffs_delta": float(delta_val),
                "abs_delta": float(abs_d),
                "magnitude": magnitude,
                "practically_relevant": bool(abs_d >= threshold),
                "threshold": float(threshold),
            }

    return {
        "metric": metric,
        "models": names,
        "pairwise": pairwise_results,
    }


def plot_cliffs_dm_heatmap(
    cliff_results: dict,
    dm_results: dict | None = None,
    model_order: list[str] | None = None,
    title: str = "Delta de Cliff por Pares (Referencia DM)",
    xlab: str = "Modelo B",
    ylab: str = "Modelo A",
    annot_decimals: int = 2,
    max_marker_size: float = 1450.0,
    min_marker_size: float = 750.0,
    circle_alpha: float = 0.45,
    legend_size: float = 11.5,
    figsize: tuple = (10, 8),
    fmt: str = "svg",
    dpi: int = 100,
) -> None:
    """Plots lower-triangle translucent pastel Cliff's delta circles with legible centered labels."""
    cliff_pair = cliff_results.get("pairwise", {})
    dm_pair = dm_results.get("pairwise", {}) if dm_results is not None else {}

    base_models = list(cliff_results.get("models", []))
    if model_order is not None:
        models = [m for m in model_order if m in base_models]
    else:
        models = base_models

    m_len = len(models)
    if m_len < 2:
        raise ValueError("At least two models are required to plot the pairwise matrix.")

    delta_mat = np.zeros((m_len, m_len), dtype=float)
    active_mat = np.ones((m_len, m_len), dtype=bool)
    annot_mat = np.full((m_len, m_len), "", dtype=object)

    for i, m_i in enumerate(models):
        for j in range(i + 1):
            m_j = models[j]
            if i == j:
                delta_mat[i, j] = 0.0
                active_mat[i, j] = False
                annot_mat[i, j] = "—"
                continue

            key_ij = f"{m_i}_vs_{m_j}"
            key_ji = f"{m_j}_vs_{m_i}"

            if key_ij in cliff_pair:
                d_val = float(cliff_pair[key_ij]["cliffs_delta"])
                dm_entry = dm_pair.get(key_ij)
            elif key_ji in cliff_pair:
                d_val = -float(cliff_pair[key_ji]["cliffs_delta"])
                dm_entry = dm_pair.get(key_ji)
                if dm_entry is not None:
                    dm_entry = {
                        "ci_95_low": -float(dm_entry["ci_95_high"]),
                        "ci_95_high": -float(dm_entry["ci_95_low"]),
                    }
            else:
                d_val = 0.0
                dm_entry = None

            delta_mat[i, j] = d_val
            annot_mat[i, j] = f"{d_val:.{annot_decimals}f}"

            if dm_entry is not None and "ci_95_low" in dm_entry and "ci_95_high" in dm_entry:
                ci_l = float(dm_entry["ci_95_low"])
                ci_h = float(dm_entry["ci_95_high"])
                active_mat[i, j] = bool((ci_l > 0.0) or (ci_h < 0.0))
            else:
                active_mat[i, j] = True

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    for i in range(m_len):
        for j in range(i + 1):
            if i == j:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor="#F5F5F5",
                    edgecolor="#CCCCCC",
                    linewidth=0.8,
                )
                ax.add_patch(rect)
                ax.text(j, i, "—", ha="center", va="center", color="#555555", fontsize=10.5)
                continue

            rect = Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor="white",
                edgecolor="#D8D8D8",
                linewidth=0.8,
            )
            ax.add_patch(rect)

            d_val = delta_mat[i, j]
            abs_d = min(abs(d_val), 1.0)

            if not active_mat[i, j]:
                ax.scatter(
                    [j],
                    [i],
                    s=55.0,
                    color="#222222",
                    marker="o",
                    zorder=4,
                )
            else:
                marker_size = min_marker_size + abs_d * (max_marker_size - min_marker_size)
                circle_color = "#F4A28C" if d_val > 0.0 else "#89C4E1"
                ax.scatter(
                    [j],
                    [i],
                    s=marker_size,
                    color=circle_color,
                    alpha=circle_alpha,
                    edgecolors="#333333",
                    linewidths=0.9,
                    marker="o",
                    zorder=4,
                )
                ax.text(
                    j,
                    i,
                    annot_mat[i, j],
                    ha="center",
                    va="center",
                    color="#000000",
                    fontweight="bold",
                    fontsize=9.5,
                    zorder=5,
                )

    ax.set_xlim(-0.5, m_len - 0.5)
    ax.set_ylim(m_len - 0.5, -0.5)
    ax.set_xticks(np.arange(m_len))
    ax.set_yticks(np.arange(m_len))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_yticklabels(models)
    ax.set_xlabel(xlab, fontsize=12, labelpad=10)
    ax.set_ylabel(ylab, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=12, pad=14, loc="center")
    _style_axes(ax, tick_size=11)

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w", label="δ > 0 (Modelo A mayor pérdida)",
            markerfacecolor="#F4A28C", markeredgecolor="#333333", markersize=14, alpha=circle_alpha,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", label="δ < 0 (Modelo A menor pérdida)",
            markerfacecolor="#89C4E1", markeredgecolor="#333333", markersize=14, alpha=circle_alpha,
        ),
        Line2D(
            [0], [0],
            marker="o", color="w", label="DM no significativo (IC 95% incluye 0)",
            markerfacecolor="#222222", markersize=8,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper right",
        fontsize=legend_size,
    )

    plt.tight_layout()
    _display_figure(fig, fmt=fmt, dpi=dpi)


def model_confidence_set(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    model_names: list[str] | None = None,
    metric: str = "qlike",
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    block_len: int | None = None,
    bootstrap_type: str = "stationary",
    lag: int | None = None,
    seed: int = 42,
) -> dict:
    """Sequential Model Confidence Set (MCS) with joint block-bootstrap resamples."""
    losses_by_model = _compute_model_losses(
        y_true=y_true,
        preds_matrix=preds_matrix,
        model_names=model_names,
        metric=metric,
    )
    model_names_list = list(losses_by_model.keys())
    loss_mat = np.column_stack([losses_by_model[m] for m in model_names_list])
    n, m_count = loss_mat.shape
    if m_count < 2 or n < 3:
        raise ValueError("preds_matrix must have at least 2 models and at least 3 time steps.")

    eff_block_len = _resolve_block_len(n=n, block_len=block_len)
    hac_lag = lag if lag is not None else max(1, int(np.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))
    rng = np.random.default_rng(seed)

    boot_indices = _generate_block_bootstrap_batch(
        n=n,
        n_bootstrap=n_bootstrap,
        block_len=eff_block_len,
        bootstrap_type=bootstrap_type,
        rng=rng,
    )

    active = list(model_names_list)
    elimination_order: list[dict] = []

    while len(active) > 1:
        k = len(active)
        active_idx = [model_names_list.index(name) for name in active]
        sub_losses = loss_mat[:, active_idx]

        t_mat = np.zeros((k, k), dtype=float)
        mean_diffs = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(i + 1, k):
                d_ij = sub_losses[:, i] - sub_losses[:, j]
                m_ij = float(np.mean(d_ij))
                se_ij = np.sqrt(_newey_west_long_run_var(d_ij, lag=hac_lag) / n)
                t_ij = (m_ij / se_ij) if se_ij > 0.0 else 0.0
                t_mat[i, j] = t_ij
                t_mat[j, i] = -t_ij
                mean_diffs[i, j] = m_ij
                mean_diffs[j, i] = -m_ij

        t_range_stat = float(np.max(np.abs(t_mat)))

        boot_t_range = np.empty(n_bootstrap, dtype=float)
        for b_idx in range(n_bootstrap):
            b_sub = sub_losses[boot_indices[b_idx], :]
            b_max = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    d_boot = b_sub[:, i] - b_sub[:, j]
                    d_centered = d_boot - mean_diffs[i, j]
                    m_b = float(np.mean(d_centered))
                    se_b = np.sqrt(_newey_west_long_run_var(d_centered, lag=hac_lag) / n)
                    val = abs(m_b / se_b) if se_b > 0.0 else 0.0
                    if val > b_max:
                        b_max = val
            boot_t_range[b_idx] = b_max

        p_val_mcs = float(np.mean(boot_t_range >= t_range_stat))

        avg_excess = np.mean(mean_diffs, axis=1)
        worst_pos = int(np.argmax(avg_excess))
        worst_model = active[worst_pos]

        if p_val_mcs < alpha:
            elimination_order.append({
                "eliminated_model": worst_model,
                "p_value": p_val_mcs,
                "t_range_stat": t_range_stat,
                "avg_loss_diff_vs_active": float(avg_excess[worst_pos]),
            })
            active.pop(worst_pos)
        else:
            break

    superior_set = list(active)
    mean_loss_map = {
        name: float(np.mean(loss_mat[:, idx]))
        for idx, name in enumerate(model_names_list)
    }
    superior_set_sorted = sorted(superior_set, key=lambda m: mean_loss_map[m])

    return {
        "metric": metric,
        "alpha": float(alpha),
        "superior_set": superior_set_sorted,
        "eliminated_models": [item["eliminated_model"] for item in elimination_order],
        "elimination_details": elimination_order,
        "mean_losses": mean_loss_map,
        "n_bootstrap": int(n_bootstrap),
        "block_len": int(eff_block_len),
        "bootstrap_type": bootstrap_type,
    }


def plot_mcs_bars(
    mcs_result: dict | pd.DataFrame,
    title: str = "Model Confidence Set (MCS)",
    xlab: str = "Pérdida Media (Fuera de Muestra)",
    ylab: str = "Modelo",
    decimals: int = 4,
    figsize: tuple = (9, 5),
    fmt: str = "svg",
    dpi: int = 100,
) -> None:
    """Plots horizontal bars of mean loss ordered from best to worst with MCS cutoff line."""
    if isinstance(mcs_result, dict):
        mean_losses = mcs_result.get("mean_losses", {})
        superior_set = set(mcs_result.get("superior_set", []))
        if not mean_losses:
            raise ValueError("mcs_result dict must contain a non-empty 'mean_losses' mapping.")
        df = pd.DataFrame({
            "model": list(mean_losses.keys()),
            "mean_loss": [float(v) for v in mean_losses.values()],
            "in_mcs": [m in superior_set for m in mean_losses.keys()],
        })
    elif isinstance(mcs_result, pd.DataFrame):
        required = {"model", "mean_loss", "in_mcs"}
        if not required.issubset(mcs_result.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")
        df = mcs_result[["model", "mean_loss", "in_mcs"]].copy()
        df["mean_loss"] = df["mean_loss"].astype(float)
        df["in_mcs"] = df["in_mcs"].astype(bool)
    else:
        raise TypeError("mcs_result must be a dict from model_confidence_set or a DataFrame.")

    df_sorted = df.sort_values("mean_loss", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    losses = df_sorted["mean_loss"].to_numpy()
    in_mcs = df_sorted["in_mcs"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    for idx in range(len(df_sorted)):
        if in_mcs[idx]:
            ax.barh(
                y_pos[idx],
                losses[idx],
                height=0.55,
                color="#89C4E1",
                edgecolor="#222222",
                linewidth=0.8,
            )
        else:
            ax.barh(
                y_pos[idx],
                losses[idx],
                height=0.55,
                color="white",
                edgecolor="#222222",
                hatch="///",
                linewidth=0.8,
            )

    excluded_idx = np.where(~in_mcs)[0]
    if len(excluded_idx) > 0 and excluded_idx[0] > 0:
        cut_y = excluded_idx[0] - 0.5
        ax.axhline(cut_y, color="#111111", linestyle="--", linewidth=1.2)
        ax.text(
            float(np.max(losses)) * 0.99,
            cut_y - 0.12,
            "Fuera del MCS",
            ha="right",
            va="bottom",
            fontsize=9.5,
            style="italic",
            color="#111111",
        )

    pad_x = max(float(np.max(losses)) * 0.015, 1e-6)
    for idx, val in enumerate(losses):
        ax.text(
            val + pad_x,
            y_pos[idx],
            f"{val:.{decimals}f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#111111",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["model"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, float(np.max(losses)) * 1.15)
    ax.set_xlabel(xlab, fontsize=12, labelpad=10)
    ax.set_ylabel(ylab, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=12, pad=14, loc="center")
    _style_axes(ax, tick_size=11)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="#89C4E1", edgecolor="#222222", label="Incluido en MCS"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#222222", hatch="///", label="Excluido del MCS"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right", fontsize=10.5)

    plt.tight_layout()
    _display_figure(fig, fmt=fmt, dpi=dpi)


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _display_figure(fig: plt.Figure, fmt: str = "svg", dpi: int = 100) -> None:
    """Renders a matplotlib figure in IPython as centered SVG or PNG."""
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


def _compute_model_losses(
    y_true: np.ndarray,
    preds_matrix: np.ndarray,
    model_names: list[str] | None = None,
    metric: str = "qlike",
) -> dict[str, np.ndarray]:
    """Computes step-by-step loss vectors per model aligned with VolatilityScorer."""
    y_arr = np.asarray(y_true, dtype=float).ravel()
    preds_arr = np.asarray(preds_matrix, dtype=float)

    if preds_arr.ndim != 2:
        raise ValueError("preds_matrix must be a 2D array of shape (n_obs, n_models).")

    n_obs, n_models = preds_arr.shape
    if n_models < 2:
        raise ValueError("preds_matrix must include at least 2 models.")
    if len(y_arr) != n_obs or n_obs == 0:
        raise ValueError("y_true must be a non-empty 1D array matching preds_matrix rows.")

    if model_names is None:
        names = [f"model_{idx}" for idx in range(n_models)]
    else:
        if len(model_names) != n_models:
            raise ValueError("model_names length must match preds_matrix columns.")
        names = list(model_names)

    scorer = VolatilityScorer()
    metric_key = metric.lower()
    losses: dict[str, np.ndarray] = {}

    for col_idx, name in enumerate(names):
        y_pred_col = preds_arr[:, col_idx]
        if metric_key == "qlike":
            eps = getattr(scorer, "eps", 1e-8)
            y_safe = np.maximum(y_arr, eps)
            p_safe = np.maximum(y_pred_col, eps)
            ratio = y_safe / p_safe
            losses[name] = ratio - np.log(ratio) - 1.0
        elif metric_key == "mse":
            losses[name] = (y_arr - y_pred_col) ** 2
        elif metric_key == "mae":
            losses[name] = np.abs(y_arr - y_pred_col)
        else:
            raise ValueError(f"Unsupported metric '{metric}' for step-by-step loss.")

    return losses


def _resolve_block_len(n: int, block_len: int | None) -> int:
    """Resolves block length using sample-size heuristic when block_len is None."""
    if block_len is not None:
        return max(1, min(int(block_len), n))
    heuristic = int(np.floor(n ** (1.0 / 3.0)))
    return max(2, min(heuristic, max(2, n // 2)))


def _generate_block_bootstrap_batch(
    n: int,
    n_bootstrap: int,
    block_len: int,
    bootstrap_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generates block-bootstrap index arrays of shape (n_bootstrap, n)."""
    mode = bootstrap_type.lower()
    if mode == "stationary":
        p = min(max(1.0 / max(float(block_len), 1.0), 1e-4), 1.0)
        batch = np.empty((n_bootstrap, n), dtype=int)
        for b in range(n_bootstrap):
            idx = np.empty(n, dtype=int)
            idx[0] = int(rng.integers(0, n))
            for t in range(1, n):
                if rng.random() < p:
                    idx[t] = int(rng.integers(0, n))
                else:
                    idx[t] = (idx[t - 1] + 1) % n
            batch[b] = idx
        return batch

    if mode == "moving_block":
        n_blocks = int(np.ceil(n / float(block_len)))
        max_start = max(1, n - block_len + 1)
        starts = rng.integers(0, max_start, size=(n_bootstrap, n_blocks))
        offsets = np.arange(block_len, dtype=int)[None, None, :]
        blocks = (starts[:, :, None] + offsets) % n
        return blocks.reshape(n_bootstrap, -1)[:, :n]

    raise ValueError("bootstrap_type must be 'stationary' or 'moving_block'.")


def _newey_west_long_run_var(d: np.ndarray, lag: int | None = None) -> float:
    """Computes Newey-West HAC long-run variance for loss differential series d."""
    d_arr = np.asarray(d, dtype=float).ravel()
    n = len(d_arr)
    if n < 2:
        return 0.0

    if lag is None:
        lag = max(1, int(np.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))
    lag = min(max(0, lag), n - 1)

    d_centered = d_arr - np.mean(d_arr)
    gamma_0 = float(np.dot(d_centered, d_centered) / n)
    var_lr = gamma_0

    for k in range(1, lag + 1):
        weight = 1.0 - (k / (lag + 1.0))
        cov_k = float(np.dot(d_centered[k:], d_centered[:-k]) / n)
        var_lr += 2.0 * weight * cov_k

    return max(var_lr, 1e-12)


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
        {"selector": "", "props": [("margin-left", "auto"), ("margin-right", "auto"), ("width", "auto"), ("border-collapse", "collapse")]},
        {"selector": "th", "props": [("text-align", "center"), ("background-color", "#f2f2f2"), ("color", "black"), ("font-weight", "bold"), ("border", "1px solid black"), ("padding", "10px")]},
        {"selector": "td", "props": [("text-align", "center"), ("border", "1px solid black"), ("padding", "10px")]},
        {"selector": "th:first-child", "props": [("text-align", "left")]},
        {"selector": "td:first-child", "props": [("text-align", "left")]},
    ]


def _render_styled(styler) -> None:
    display(
        HTML(
            "<div style='text-align: center; width: 100%;'>"
            + styler.to_html()
            + "</div>"
        )
    )