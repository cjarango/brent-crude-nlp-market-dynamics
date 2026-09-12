"""
ts_toolkit
==========

Time Series Diagnostics & Visualization Toolkit.

A curated collection of visualization and statistical analysis utilities
for structured and reproducible univariate time-series exploration in
Jupyter notebooks.

All figures are rendered inline as vector SVG by default (resolution
independent, ideal for reports and papers), with an optional raster PNG
mode for cases where file size or renderer compatibility matters. All
statistical outputs are rendered as styled HTML tables using a
consistent visual language.

Visual Language
----------------
- Black lines / stems / boxes, white fills
- Thin (0.8 pt) black spines on every axis
- White background, minimal or no gridlines
- Bold axis labels and titles, tight layout
- Default output format: SVG (vector, dpi-independent)

Main Functionalities
---------------------
Rendering
    render_figure

Autocorrelation
    plot_acf_pacf
    plot_ccf_panels

Stationarity / Residual Diagnostics
    univariate_ts_diagnostics

Trend & Seasonality
    plot_series_and_annual_boxplot
    plot_ts_boxplots_by_calendar
    plot_rolling_smoothing_panels
    plot_wavelet_decomposition

Association
    correlation_windows_table

Data Quality / Summary
    dataframe_integrity_table
    descriptive_stats_table

Modeling Prep
    plot_train_test_split

Dependencies
------------
numpy, pandas, matplotlib, scipy, statsmodels, PyWavelets (pywt), IPython
"""

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__version__ = "1.0.0"

__author__ = (
    "Juan Camilo Mendoza Arango <cjarango@uninorte.edu.co>, "
    "Sergio Rafael Rada De La Cruz <srrada@uninorte.edu.co>, "
    "Natalia Paola Alvarado Corrales <npalvarado@uninorte.edu.co>, "
    "Miguel Ángel Pérez Vargas <vargasmiguel@uninorte.edu.co>"
)

_UNINORTE_AFFILIATION = {
    "name": "Universidad del Norte",
    "department": "Department of Mathematics, Physics and Data Science",
    "city": "Barranquilla",
    "country": "Colombia",
}

__authors_detail__ = [
    {
        "name": "Juan Camilo Mendoza Arango",
        "email": "cjarango@uninorte.edu.co",
        "orcid": "0000-0001-6872-8311",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "cjarango",
    },
    {
        "name": "Sergio Rafael Rada De La Cruz",
        "email": "srrada@uninorte.edu.co",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "s3rgiorafael2018-create",
    },
    {
        "name": "Natalia Paola Alvarado Corrales",
        "email": "npalvarado@uninorte.edu.co",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "paolacorr67-ctrl",
    },
    {
        "name": "Miguel Ángel Pérez Vargas",
        "email": "vargasmiguel@uninorte.edu.co",
        "affiliations": [_UNINORTE_AFFILIATION],
        "github": "miguelpvmr",
    },
]

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import io
import base64
import warnings

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
import pywt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from IPython.display import HTML, display

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------
__all__ = [
    "render_figure",
    "plot_acf_pacf",
    "plot_ccf_panels",
    "univariate_ts_diagnostics",
    "plot_series_and_annual_boxplot",
    "plot_ts_boxplots_by_calendar",
    "plot_rolling_smoothing_panels",
    "plot_wavelet_decomposition",
    "correlation_windows_table",
    "dataframe_integrity_table",
    "descriptive_stats_table",
    "plot_train_test_split",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_VALID_P_ADJUST_METHODS = {
    "bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg",
    "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky",
}

_P_ADJUST_ALIASES = {
    "bh": "fdr_bh",
    "by": "fdr_by",
    "tsbh": "fdr_tsbh",
    "tsbky": "fdr_tsbky",
    "bonf": "bonferroni",
}

_METHOD_FUNCS = {
    "pearson": stats.pearsonr,
    "spearman": stats.spearmanr,
    "kendall": stats.kendalltau,
}

_DEFAULT_ROW_LABELS = (
    "Count", "Mean", "Std. Dev.", "Min",
    "Q1", "Median", "Q3", "Max", "IQR",
)


# ===========================================================================
# PUBLIC API
# ===========================================================================

def render_figure(fig, fmt="svg", dpi=100):
    """Render a matplotlib figure inline as centered HTML.

    Args:
        fig (matplotlib.figure.Figure): The figure to render. The figure
            is closed after encoding, so it will not also pop up as a
            separate matplotlib window.
        fmt ({"svg", "png"}): Output format. ``"svg"`` (default) embeds a
            resolution-independent vector image directly in the notebook
            output; ``dpi`` has no effect in this mode. ``"png"``
            rasterizes the figure and embeds it as a Base64-encoded
            image.
        dpi (int): Resolution (dots per inch) used only when
            ``fmt="png"``. Defaults to ``100``, a reasonable balance
            between on-screen clarity and output size for inline
            notebook display.

    Returns:
        None: The figure is displayed as a side effect; nothing is
        returned.

    Raises:
        ValueError: If ``fmt`` is not one of ``{"svg", "png"}``.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> render_figure(fig)
        >>> render_figure(fig, fmt="png", dpi=150)
    """
    if fmt not in ("svg", "png"):
        raise ValueError(f"'fmt' must be 'svg' or 'png', got '{fmt}'.")

    buf = io.BytesIO()

    if fmt == "svg":
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg_data = buf.getvalue().decode("utf-8")
        display(
            HTML(
                f'<div style="display: flex; justify-content: center; '
                f'width: 100%;">{svg_data}</div>'
            )
        )
    else:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        encoded = base64.b64encode(buf.getbuffer()).decode("ascii")
        display(
            HTML(
                f'<div style="text-align: center; width: 100%;">'
                f'<img src="data:image/png;base64,{encoded}"></div>'
            )
        )


def plot_acf_pacf(
    series,
    titles=("Autocorrelation Function", "Partial Autocorrelation Function"),
    xlab="Lag",
    ylab="Autocorrelation",
    pads=(9, 9, 12),
    label_size=12,
    title_size=12,
    tick_size=11,
    xlim=None,
    ylim=(-1.1, 1.1),
    x_ticks=None,
    y_ticks=None,
    lags=50,
    figsize=(8, 4),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot the ACF and PACF of a univariate series side by side.

    Draws the autocorrelation and partial autocorrelation functions of
    the given series using ``statsmodels`` and applies a consistent,
    publication-ready style (black stems, light-gray confidence bands,
    thin black spines). The figure is optionally passed to
    ``render_figure`` for inline rendering.

    Args:
        series (array-like or pandas.Series): Input time series. NaNs are
            dropped before plotting.
        titles (tuple or list of str): Titles for the ACF and PACF
            subplots respectively. Must contain exactly two elements.
            Defaults to ``("Autocorrelation Function",
            "Partial Autocorrelation Function")``.
        xlab (str): Label for the x-axis of both subplots. Defaults to
            ``"Lag"``.
        ylab (str): Label for the y-axis of the left subplot. Defaults to
            ``"Autocorrelation"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis label, the y-axis label and the titles,
            respectively. Defaults to ``(9, 9, 12)``.
        label_size (float): Font size for the axis labels. Defaults to 12.
        title_size (float): Font size for the subplot titles. Defaults to
            12.
        tick_size (float): Font size for the tick labels. Defaults to 11.
        xlim (tuple of float, optional): ``(xmin, xmax)`` limits for the
            x-axis of both subplots. If ``None``, matplotlib defaults are
            used.
        ylim (tuple of float, optional): ``(ymin, ymax)`` limits for the
            y-axis of both subplots. Defaults to ``(-1.1, 1.1)``.
        x_ticks (array-like, optional): Explicit x-axis tick positions for
            both subplots. If ``None``, matplotlib chooses them.
        y_ticks (array-like, optional): Explicit y-axis tick positions for
            both subplots. If ``None``, matplotlib chooses them.
        lags (int): Number of lags to display in both ACF and PACF.
            Defaults to 50.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(8, 4)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax1, ax2))`` with the matplotlib figure and the
        two axes (ACF on the left, PACF on the right).

    Raises:
        TypeError: If ``series`` has no ``dropna`` attribute, if any of
            ``titles``, ``pads``, ``figsize`` is not a tuple/list, or if
            any of ``xlim``, ``ylim`` is not a tuple/list of length 2.
        ValueError: If ``series`` is empty, if ``titles`` or ``pads`` do
            not contain the expected number of elements, if ``titles``
            contains non-string elements, or if ``lags`` is not a
            positive integer.
    """
    if not hasattr(series, "dropna"):
        raise TypeError(
            "'series' must be a pandas Series or similar object with a "
            "'dropna' method."
        )

    series = series.dropna()

    if len(series) == 0:
        raise ValueError("'series' is empty after dropping NaNs.")

    if not isinstance(titles, (tuple, list)):
        raise TypeError("'titles' must be a tuple or list of two strings.")

    if len(titles) != 2:
        raise ValueError(
            f"'titles' must contain exactly 2 elements, got {len(titles)}."
        )

    if not all(isinstance(t, str) for t in titles):
        raise TypeError("All elements of 'titles' must be strings.")

    _validate_pads(pads)

    if xlim is not None:
        if not isinstance(xlim, (tuple, list)) or len(xlim) != 2:
            raise TypeError("'xlim' must be None or a tuple/list of length 2.")

    _validate_ylim(ylim)
    _validate_figsize(figsize)

    if not isinstance(lags, int) or lags <= 0:
        raise ValueError("'lags' must be a positive integer.")

    pad_x, pad_y, pad_title = pads

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    common_kwargs = dict(
        lags=lags,
        zero=False,
        color="black",
        vlines_kwargs={"color": "black", "linewidth": 0.8},
    )

    plot_acf(series, ax=ax1, title=titles[0], **common_kwargs)
    plot_pacf(series, ax=ax2, title=titles[1], **common_kwargs)

    for ax in (ax1, ax2):
        _style_axes(ax, tick_size)

        for collection in ax.collections:
            collection.set_facecolor("lightgray")
            collection.set_alpha(0.3)

        for line in ax.lines:
            line.set_markerfacecolor("black")
            line.set_markeredgecolor("black")
            line.set_markersize(1.8)
            line.set_linewidth(0.8)

        if ylim is not None:
            ax.set_ylim(*ylim)
        if xlim is not None:
            ax.set_xlim(*xlim)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

        ax.set_title(ax.get_title(), fontsize=title_size, pad=pad_title)
        ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)

    ax1.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
    ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax1, ax2)


def univariate_ts_diagnostics(
    series,
    lags=(10, 20, 30),
    column_titles=("Prueba", "Estadístico", "p-valor", "Conclusión (α = .05)"),
    acronym=False,
    p_adjust=None,
    p_adjust_title="p-ajustado",
    include_arch=True,
    alpha=0.05,
    adf_autolag="AIC",
    kpss_regression="c",
    kpss_nlags="auto",
    decimals=4,
    render=True,
):
    """Run a diagnostic battery on a univariate time series and tabulate it.

    Runs ADF, KPSS, Ljung-Box and (optionally) ARCH-LM tests on a single
    univariate series, formats the p-values in APA style, and returns
    both the plain and the styled DataFrame. Optionally renders the
    styled table as centered HTML (useful in Jupyter notebooks).

    When ``p_adjust`` is provided, an extra column (named
    ``p_adjust_title``) is added with adjusted p-values computed **by
    block**. Each block is identified by a string prefix that is matched
    against the beginning of the test-name column (e.g. ``"Ljung-Box"``
    matches every row whose test name starts with ``"Ljung-Box"``). The
    correction is applied only within each block, so ADF and KPSS (whose
    null hypotheses are complementary, not redundant) are not lumped
    together. The conclusion column is computed using the adjusted
    p-value whenever a row belongs to an adjusted block, and falls back
    to the raw p-value otherwise.

    Args:
        series (pandas.Series): Input time series. NaNs are dropped
            before running the tests.
        lags (tuple or list of int): Lags at which the Ljung-Box (and, if
            ``include_arch`` is ``True``, the ARCH-LM) tests are
            evaluated. Defaults to ``(10, 20, 30)``.
        column_titles (tuple or list of str): Column names for the output
            table in the order ``(test, statistic, p_value, conclusion)``.
            Must contain exactly four strings. Defaults to
            ``("Prueba", "Estadístico", "p-valor",
            "Conclusión (α = .05)")``.
        acronym (bool): If ``True``, the ADF and KPSS test names are
            shown as short acronyms (``"ADF"``, ``"KPSS"``). If ``False``
            (default), the full names are used with the acronym in
            parentheses. Defaults to ``False``.
        p_adjust (tuple or list of tuple, optional): Sequence of
            ``(block_prefix, method)`` tuples defining which rows get a
            multiple-testing correction and how. ``block_prefix`` is
            matched against the beginning of the test-name column
            (case-sensitive). Supported ``method`` values follow
            ``statsmodels.stats.multitest.multipletests`` plus aliases
            ``"bh" -> "fdr_bh"``, ``"by" -> "fdr_by"``,
            ``"tsbh" -> "fdr_tsbh"``, ``"tsbky" -> "fdr_tsbky"``,
            ``"bonf" -> "bonferroni"``. If ``None`` (default), no
            adjustment is applied and the extra column is not created.
        p_adjust_title (str): Name of the column that will store the
            adjusted p-values. Only used when ``p_adjust`` is not
            ``None``. Must not collide with any element of
            ``column_titles``. Defaults to ``"p-ajustado"``.
        include_arch (bool): If ``True``, add ARCH-LM tests for each lag
            in ``lags``. If ``False``, only ADF, KPSS and Ljung-Box are
            run. Defaults to ``True``.
        alpha (float): Significance level used to build the conclusion
            column. Must be in ``(0, 1)``. Defaults to 0.05.
        adf_autolag (str or None): Lag selection criterion passed to
            ``adfuller`` (``"AIC"``, ``"BIC"``, ``"t-stat"`` or
            ``None``). Defaults to ``"AIC"``.
        kpss_regression (str): Null hypothesis type for ``kpss``
            (``"c"`` for level-stationary, ``"ct"`` for trend-stationary).
            Defaults to ``"c"``.
        kpss_nlags (int or str): Number of lags for ``kpss``; ``"auto"``
            uses the default heuristic. Defaults to ``"auto"``.
        decimals (int): Number of decimals used to format the statistic
            column. Defaults to 4.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain (unstyled) results table, including
        the adjusted p-value column when ``p_adjust`` is provided. The
        styled version is built internally and, when ``render=True``,
        rendered as HTML — it is not returned.

    Raises:
        TypeError: If ``series`` has no ``dropna`` method, if ``lags`` or
            ``column_titles`` is not a tuple/list, if ``column_titles``
            elements are not strings, if ``acronym`` is not a boolean,
            if ``render`` is not a boolean, if ``p_adjust`` is not a
            tuple/list of ``(prefix, method)`` tuples of strings, or if
            ``p_adjust_title`` is not a string.
        ValueError: If ``series`` is empty after dropping NaNs, if
            ``lags`` is empty or contains non-positive integers, if
            ``column_titles`` does not contain exactly four elements, if
            ``alpha`` is not in ``(0, 1)``, if ``p_adjust`` is empty, if
            any ``method`` in ``p_adjust`` is unsupported, or if
            ``p_adjust_title`` collides with a name in ``column_titles``.
    """
    if not hasattr(series, "dropna"):
        raise TypeError(
            "'series' must be a pandas Series or similar object with a "
            "'dropna' method."
        )

    series = series.dropna()

    if len(series) == 0:
        raise ValueError("'series' is empty after dropping NaNs.")

    if not isinstance(lags, (tuple, list)):
        raise TypeError("'lags' must be a tuple or list of integers.")

    if len(lags) == 0:
        raise ValueError("'lags' must contain at least one lag.")

    if not all(isinstance(l, int) and l > 0 for l in lags):
        raise ValueError("All elements of 'lags' must be positive integers.")

    if not isinstance(column_titles, (tuple, list)):
        raise TypeError(
            "'column_titles' must be a tuple or list of four strings."
        )

    if len(column_titles) != 4:
        raise ValueError(
            "'column_titles' must contain exactly 4 elements "
            "(test, statistic, p_value, conclusion)."
        )

    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    if not isinstance(acronym, bool):
        raise TypeError("'acronym' must be a boolean.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
        raise ValueError("'alpha' must be a number in the interval (0, 1).")

    if p_adjust is not None:
        if not isinstance(p_adjust, (tuple, list)):
            raise TypeError(
                "'p_adjust' must be None or a tuple/list of "
                "(block_prefix, method) tuples."
            )
        if len(p_adjust) == 0:
            raise ValueError(
                "'p_adjust' must contain at least one (block_prefix, "
                "method) tuple, or be None."
            )
        for item in p_adjust:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(
                    "Each element of 'p_adjust' must be a "
                    "(block_prefix, method) tuple."
                )
            prefix, method = item
            if not isinstance(prefix, str) or not isinstance(method, str):
                raise TypeError(
                    "Both 'block_prefix' and 'method' in 'p_adjust' "
                    "must be strings."
                )
            _resolve_p_adjust_method(method)

        if not isinstance(p_adjust_title, str):
            raise TypeError("'p_adjust_title' must be a string.")

        if p_adjust_title in column_titles:
            raise ValueError(
                f"'p_adjust_title' ('{p_adjust_title}') collides with one "
                f"of the names in 'column_titles'."
            )

    col_test, col_stat, col_p, col_concl = column_titles
    lags = list(lags)

    adf_res = adfuller(series, autolag=adf_autolag)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_res = kpss(series, regression=kpss_regression, nlags=kpss_nlags)

    lb_res = acorr_ljungbox(series, lags=lags, return_df=True)

    adf_label = "ADF" if acronym else "Augmented Dickey-Fuller (ADF)"
    kpss_label = "KPSS" if acronym else "Kwiatkowski-Phillips (KPSS)"

    rows = [
        {col_test: adf_label, col_stat: adf_res[0], col_p: adf_res[1]},
        {col_test: kpss_label, col_stat: kpss_res[0], col_p: kpss_res[1]},
    ]

    for lag in lags:
        rows.append({
            col_test: f"Ljung-Box (Lag {lag})",
            col_stat: lb_res.loc[lag, "lb_stat"],
            col_p: lb_res.loc[lag, "lb_pvalue"],
        })

    if include_arch:
        for lag in lags:
            arch_res = het_arch(series, nlags=lag)
            rows.append({
                col_test: f"ARCH-LM (Lag {lag})",
                col_stat: arch_res[0],
                col_p: arch_res[1],
            })

    df_inferencial = pd.DataFrame(rows)
    df_inferencial["_p_raw_num"] = pd.to_numeric(
        df_inferencial[col_p], errors="coerce"
    )

    if p_adjust is not None:
        df_inferencial[p_adjust_title] = float("nan")
        for prefix, method in p_adjust:
            method_key = _resolve_p_adjust_method(method)
            mask = df_inferencial[col_test].str.startswith(prefix)
            if not mask.any():
                continue
            raw_p = df_inferencial.loc[mask, "_p_raw_num"].values
            _, p_adj, _, _ = multipletests(raw_p, method=method_key)
            df_inferencial.loc[mask, p_adjust_title] = p_adj

        df_inferencial["_p_adj_num"] = pd.to_numeric(
            df_inferencial[p_adjust_title], errors="coerce"
        )
    else:
        df_inferencial["_p_adj_num"] = float("nan")

    conclusions = []
    for _, row in df_inferencial.iterrows():
        p_adj = row["_p_adj_num"]
        p_raw = row["_p_raw_num"]
        p_used = p_adj if pd.notna(p_adj) else p_raw
        conclusions.append(_conclusion_for(row[col_test], p_used, alpha))

    df_inferencial[col_concl] = conclusions

    df_inferencial[col_p] = df_inferencial["_p_raw_num"].apply(_format_apa_pvalue)

    if p_adjust is not None:
        df_inferencial[p_adjust_title] = df_inferencial["_p_adj_num"].apply(
            _format_apa_pvalue
        )

    df_inferencial = df_inferencial.drop(columns=["_p_raw_num", "_p_adj_num"])

    ordered_cols = [col_test, col_stat, col_p]
    if p_adjust is not None:
        ordered_cols.append(p_adjust_title)
    ordered_cols.append(col_concl)
    df_inferencial = df_inferencial[ordered_cols]

    format_dict_inf = {
        col_stat: lambda x: _smart_format(x, f"{{:,.{decimals}f}}"),
    }

    styled_inf = (
        df_inferencial.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules())
        .format(format_dict_inf)
    )

    if render:
        _render_styled(styled_inf)

    return df_inferencial


def plot_series_and_annual_boxplot(
    series,
    titles=("Time series", "Annual distribution"),
    xlab=("Date", "Year"),
    ylab="Value",
    pads=(9, 9, 12),
    label_size=12,
    title_size=12,
    tick_size=11,
    line_color="black",
    line_width=1.0,
    box_width=0.6,
    box_facecolor="white",
    box_xpad=0.8,
    year_locator_interval=2,
    year_formatter="%Y",
    box_tick_step=2,
    ylim=None,
    y_step=None,
    y_format=None,
    figsize=(8, 4),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot a univariate series over time alongside its annual boxplot.

    Draws the raw series as a line plot on the left and the distribution
    of the series grouped by calendar year as a boxplot on the right,
    applying a consistent, publication-ready style (black lines, thin
    black spines, no y-axis on the right subplot). The figure is
    optionally passed to ``render_figure`` for inline rendering.

    Args:
        series (pandas.Series): Input time series. Must have a
            ``DatetimeIndex``. NaNs are dropped before plotting.
        titles (tuple or list of str): Titles for the left (line) and
            right (boxplot) subplots, respectively. Must contain exactly
            two strings. Defaults to ``("Time series",
            "Annual distribution")``.
        xlab (tuple or list of str): X-axis labels for the left (line)
            and right (boxplot) subplots. Must contain exactly two
            strings. Defaults to ``("Date", "Year")``.
        ylab (str): Y-axis label for the left subplot. The right subplot
            hides its y-axis. Defaults to ``"Value"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis labels, the y-axis label and the
            titles, respectively. Defaults to ``(9, 9, 12)``.
        label_size (float): Font size for the axis labels. Defaults to 12.
        title_size (float): Font size for the subplot titles. Defaults to
            12.
        tick_size (float): Font size for the tick labels. Defaults to 11.
        line_color (str): Color of the line in the left subplot. Defaults
            to ``"black"``.
        line_width (float): Width of the line in the left subplot.
            Defaults to 1.0.
        box_width (float): Width of each box in the right subplot.
            Defaults to 0.6.
        box_facecolor (str): Face color of the boxes. Defaults to
            ``"white"``.
        box_xpad (float): Padding added to the left and right x-limits of
            the boxplot subplot, in year units. Defaults to 0.8.
        year_locator_interval (int): Interval, in years, between major
            x-axis ticks on the left subplot. Defaults to 2.
        year_formatter (str): ``strftime`` format string for the left
            subplot's x-axis. Defaults to ``"%Y"``.
        box_tick_step (int): Show every ``box_tick_step``-th year as a
            tick label on the right subplot (starting from the first
            year present in the data). Defaults to 2.
        ylim (tuple of float, optional): ``(ymin, ymax)`` limits shared
            by both subplots. If ``None``, matplotlib defaults are used.
        y_step (float, optional): If provided, sets a ``MultipleLocator``
            with this step on the y-axis of the left subplot (shared with
            the right one because of ``sharey=True``). If ``None``,
            matplotlib chooses the ticks. Defaults to ``None``.
        y_format (str, optional): If provided, a ``printf``-style format
            string (e.g. ``"%.2f"``) applied to the y-axis ticks via
            ``FormatStrFormatter``. If ``None``, matplotlib default
            formatting is used. Defaults to ``None``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(8, 4)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax1, ax2))`` with the matplotlib figure and the
        two axes (line plot on the left, boxplot on the right).

    Raises:
        TypeError: If ``series`` has no ``dropna`` method, if its index
            is not a ``DatetimeIndex``, if ``titles``, ``xlab`` or
            ``figsize`` is not a tuple/list of the right length, if
            ``ylim`` is not ``None`` or a tuple/list of length 2, or if
            ``y_format`` is not ``None`` or a string.
        ValueError: If ``series`` is empty after dropping NaNs, if
            ``titles`` or ``xlab`` does not contain exactly two strings,
            if ``year_locator_interval`` or ``box_tick_step`` is not a
            positive integer, or if ``y_step`` is not ``None`` or a
            positive number.
    """
    if not hasattr(series, "dropna"):
        raise TypeError(
            "'series' must be a pandas Series or similar object with a "
            "'dropna' method."
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("'series' must have a pandas DatetimeIndex.")

    series = series.dropna()

    if len(series) == 0:
        raise ValueError("'series' is empty after dropping NaNs.")

    if not isinstance(titles, (tuple, list)) or len(titles) != 2:
        raise TypeError("'titles' must be a tuple or list of two strings.")

    if not all(isinstance(t, str) for t in titles):
        raise TypeError("All elements of 'titles' must be strings.")

    if not isinstance(xlab, (tuple, list)) or len(xlab) != 2:
        raise TypeError("'xlab' must be a tuple or list of two strings.")

    if not all(isinstance(t, str) for t in xlab):
        raise TypeError("All elements of 'xlab' must be strings.")

    _validate_pads(pads)
    _validate_ylim(ylim)
    _validate_y_step(y_step)
    _validate_y_format(y_format)
    _validate_figsize(figsize)

    if not isinstance(year_locator_interval, int) or year_locator_interval <= 0:
        raise ValueError("'year_locator_interval' must be a positive integer.")

    if not isinstance(box_tick_step, int) or box_tick_step <= 0:
        raise ValueError("'box_tick_step' must be a positive integer.")

    pad_x, pad_y, pad_title = pads
    title_left, title_right = titles
    xlab_left, xlab_right = xlab

    years = sorted(series.index.year.unique())
    box_data = [series[series.index.year == yr].values for yr in years]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax in (ax1, ax2):
        _style_axes(ax, tick_size)

    ax1.plot(series.index, series.values, color=line_color, linewidth=line_width)
    ax1.set_xlabel(xlab_left, fontsize=label_size, labelpad=pad_x)
    ax1.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
    ax1.set_title(title_left, fontsize=title_size, pad=pad_title)
    _apply_year_axis(ax1, year_locator_interval, year_formatter)

    ax2.boxplot(
        box_data,
        positions=years,
        **_boxplot_kwargs(box_facecolor, box_width),
    )

    tick_years = years[::box_tick_step]
    ax2.set_xticks(tick_years)
    ax2.set_xticklabels(tick_years, rotation=0)
    ax2.set_xlim(min(years) - box_xpad, max(years) + box_xpad)
    ax2.set_xlabel(xlab_right, fontsize=label_size, labelpad=pad_x)
    ax2.set_title(title_right, fontsize=title_size, pad=pad_title)

    _hide_yaxis(ax2)
    _apply_yaxis_settings(ax1, ylim=ylim, y_step=y_step, y_format=y_format)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax1, ax2)


def plot_ts_boxplots_by_calendar(
    series,
    groups,
    ylab="Value",
    pads=(9, 9, 12),
    label_size=12,
    title_size=12,
    tick_size=11,
    box_width=0.6,
    box_facecolor="white",
    box_xpad=0.8,
    ylim=None,
    y_step=None,
    y_format=None,
    sharey=False,
    figsize=(8, 4),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot two calendar-based boxplots of a univariate time series.

    Splits a univariate time series into two calendar groupings (e.g. by
    calendar month and by day of the week), draws one boxplot per group
    on each panel, and applies a consistent, publication-ready style
    (white boxes, black edges, thin black spines). The right panel hides
    its y-axis. The figure is optionally passed to ``render_figure`` for
    inline rendering.

    Args:
        series (pandas.Series): Input time series. Must have a
            ``DatetimeIndex``. NaNs are dropped before plotting.
        groups (tuple or list of dict): Exactly two grouper specs, one
            per panel. Each spec must be a dict with the keys ``"by"``
            (callable or array-like), ``"positions"``, ``"labels"``,
            ``"xlabel"`` and ``"title"``.

            Example::

                groups=(
                    {
                        "by": lambda idx: idx.month,
                        "positions": list(range(1, 13)),
                        "labels": [str(m) for m in range(1, 13)],
                        "xlabel": "Mes",
                        "title": "Distribución mensual",
                    },
                    {
                        "by": lambda idx: idx.dayofweek,
                        "positions": [0, 1, 2, 3, 4],
                        "labels": ["Lun", "Mar", "Mié", "Jue", "Vie"],
                        "xlabel": "Día de la semana",
                        "title": "Distribución por día de la semana",
                    },
                )
        ylab (str): Y-axis label for the left panel. The right panel
            hides its y-axis. Defaults to ``"Value"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis labels, the y-axis label and the
            titles, respectively. Defaults to ``(9, 9, 12)``.
        label_size (float): Font size for the axis labels. Defaults to 12.
        title_size (float): Font size for the panel titles. Defaults to
            12.
        tick_size (float): Font size for the tick labels. Defaults to 11.
        box_width (float): Width of each box. Defaults to 0.6.
        box_facecolor (str): Face color of the boxes. Defaults to
            ``"white"``.
        box_xpad (float): Padding added to the left and right x-limits of
            each panel, in position units. Defaults to 0.8.
        ylim (tuple of float, optional): ``(ymin, ymax)`` limits for the
            left panel. If ``None``, matplotlib defaults are used.
        y_step (float, optional): If provided, sets a ``MultipleLocator``
            with this step on the y-axis of the left panel.
        y_format (str, optional): If provided, a ``printf``-style format
            string (e.g. ``"%.2f"``) applied to the y-axis ticks of the
            left panel.
        sharey (bool): If ``True``, both panels share the y-axis (the
            right panel still hides its tick labels). Defaults to
            ``False``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(8, 4)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, (ax1, ax2))`` with the matplotlib figure and the
        two axes.

    Raises:
        TypeError: If ``series`` has no ``dropna`` method, if its index
            is not a ``DatetimeIndex``, if ``groups`` is not a tuple/list
            of two dicts, if any spec is missing a required key, if
            ``figsize`` is not a tuple/list of length 2, if ``ylim`` is
            not ``None`` or a tuple/list of length 2, or if ``y_format``
            is not ``None`` or a string.
        ValueError: If ``series`` is empty after dropping NaNs, if
            ``y_step`` is not ``None`` or a positive number, or if a
            grouper's ``by`` array does not match the series length.
    """
    if not hasattr(series, "dropna"):
        raise TypeError(
            "'series' must be a pandas Series or similar object with a "
            "'dropna' method."
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("'series' must have a pandas DatetimeIndex.")

    series = series.dropna()

    if len(series) == 0:
        raise ValueError("'series' is empty after dropping NaNs.")

    if not isinstance(groups, (tuple, list)) or len(groups) != 2:
        raise TypeError(
            "'groups' must be a tuple or list of exactly two grouper "
            "specs (one per panel)."
        )

    required_keys = ("by", "positions", "labels", "xlabel", "title")
    for i, spec in enumerate(groups):
        if not isinstance(spec, dict):
            raise TypeError(f"'groups[{i}]' must be a dict.")
        missing = [k for k in required_keys if k not in spec]
        if missing:
            raise ValueError(
                f"'groups[{i}]' is missing required keys: {missing}."
            )

    _validate_pads(pads)
    _validate_ylim(ylim)
    _validate_y_step(y_step)
    _validate_y_format(y_format)
    _validate_figsize(figsize)

    pad_x, pad_y, pad_title = pads

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=sharey)

    for ax in (ax1, ax2):
        _style_axes(ax, tick_size)

    for ax, spec in zip((ax1, ax2), groups):
        by = spec["by"]
        keys = np.asarray(by(series.index)) if callable(by) else np.asarray(by)

        if len(keys) != len(series):
            raise ValueError(
                "A grouper's 'by' array must have the same length as the "
                f"series after dropping NaNs (got {len(keys)} vs "
                f"{len(series)})."
            )

        positions = list(spec["positions"])
        box_data = [series.values[keys == p] for p in positions]

        ax.boxplot(
            box_data,
            positions=positions,
            **_boxplot_kwargs(box_facecolor, box_width),
        )

        ax.set_xticks(positions)
        ax.set_xticklabels(spec["labels"], rotation=0, fontsize=tick_size)
        ax.set_xlim(min(positions) - box_xpad, max(positions) + box_xpad)
        ax.set_xlabel(spec["xlabel"], fontsize=label_size, labelpad=pad_x)
        ax.set_title(spec["title"], fontsize=title_size, pad=pad_title)

    ax1.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
    _hide_yaxis(ax2)
    _apply_yaxis_settings(ax1, ylim=ylim, y_step=y_step, y_format=y_format)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, (ax1, ax2)


def dataframe_integrity_table(
    df,
    column_titles=("Variable", "Dtype", "Missing Values"),
    decimals=0,
    render=True,
):
    """Summarize a DataFrame's structure, dtypes and missing values.

    Builds a small integrity table with one row per column in ``df``,
    reporting the column name, its pandas dtype as a string, and the
    number of missing values. Applies a consistent, publication-ready
    style (centered cells, thin black borders, left-aligned first
    column). Optionally renders the styled table as centered HTML
    (useful in Jupyter notebooks).

    Args:
        df (pandas.DataFrame): Input DataFrame. Must have a ``dtypes``
            attribute and an ``isnull`` method.
        column_titles (tuple or list of str): Column names for the output
            table in the order ``(variable, dtype, missing)``. Must
            contain exactly three strings. Defaults to
            ``("Variable", "Dtype", "Missing Values")``.
        decimals (int): Number of decimals used to format the missing
            values column. Since counts are integers, the default of
            ``0`` is usually what you want. Defaults to 0.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain (unstyled) results table. The styled
        version is built internally and, when ``render=True``, rendered
        as HTML — it is not returned.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame, if
            ``column_titles`` is not a tuple/list of three strings, if
            ``decimals`` is not an int, or if ``render`` is not a
            boolean.
        ValueError: If ``column_titles`` does not contain exactly three
            elements, if any element of ``column_titles`` is not a
            string, or if ``decimals`` is negative.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")

    if not isinstance(column_titles, (tuple, list)):
        raise TypeError(
            "'column_titles' must be a tuple or list of three strings."
        )

    if len(column_titles) != 3:
        raise ValueError(
            "'column_titles' must contain exactly 3 elements "
            "(variable, dtype, missing)."
        )

    if not all(isinstance(c, str) for c in column_titles):
        raise TypeError("All elements of 'column_titles' must be strings.")

    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("'decimals' must be a non-negative integer.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    col_var, col_dtype, col_missing = column_titles

    df_integrity = pd.DataFrame(
        {
            col_var: df.columns,
            col_dtype: [str(dtype) for dtype in df.dtypes],
            col_missing: df.isnull().sum().values,
        }
    )

    format_dict_integrity = {
        col_missing: lambda x: _smart_format(x, f"{{:,.{decimals}f}}"),
    }

    styled_integrity = (
        df_integrity.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules())
        .format(format_dict_integrity)
    )

    if render:
        _render_styled(styled_integrity)

    return df_integrity


def descriptive_stats_table(
    df,
    column_labels=None,
    param_col_title="Parameter",
    row_labels=_DEFAULT_ROW_LABELS,
    int_row_indices=(0,),
    int_format="{:,.0f}",
    float_format="{:,.4f}",
    round_threshold=5e-5,
    render=True,
):
    """Build a descriptive-statistics table for every numeric column.

    Computes a set of summary statistics (count, mean, standard
    deviation, minimum, quartiles, median, maximum and IQR) for each
    column of ``df``, reshapes the result into a long table with one row
    per statistic and one column per input variable, and applies a
    consistent, publication-ready style. Horizontal alignment follows
    the shared style rules (first column left-aligned, every other
    column centered); every cell is additionally centered vertically.
    Small absolute values are rounded to zero to avoid floating-point
    noise, and the count row is formatted as integers while the rest use
    fixed decimals.

    Args:
        df (pandas.DataFrame): Input DataFrame with numeric columns.
            Non-numeric columns will raise a pandas error during
            aggregation.
        column_labels (tuple or list of str, optional): Display names for
            the DataFrame columns, in the same order as ``df.columns``.
            Must have the same length as the number of columns in ``df``.
            If ``None``, the original column names are used.
        param_col_title (str): Header of the first column, which lists
            the statistic names. Defaults to ``"Parameter"``.
        row_labels (tuple or list of str): Display names for the nine
            statistics, in the order ``(count, mean, std, min, q1,
            median, q3, max, iqr)``. Must contain exactly nine strings.
            Defaults to ``("Count", "Mean", "Std. Dev.", "Min", "Q1",
            "Median", "Q3", "Max", "IQR")``.
        int_row_indices (tuple or list of int): Indices (into
            ``row_labels``) of the rows that should be formatted as
            integers. Defaults to ``(0,)`` (the count row).
        int_format (str): Format string used for the rows listed in
            ``int_row_indices``. Defaults to ``"{:,.0f}"``.
        float_format (str): Format string used for all other rows.
            Defaults to ``"{:,.4f}"``.
        round_threshold (float): Absolute threshold below which values
            are replaced by ``0.0`` before formatting. Set to ``0`` to
            disable. Defaults to ``5e-5``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The long-format results table (unstyled). The
        styled version is built internally and, when ``render=True``,
        rendered as HTML — it is not returned.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame, if
            ``column_labels`` is not ``None`` or a tuple/list of strings,
            if ``param_col_title`` is not a string, if ``row_labels`` is
            not a tuple/list of nine strings, if ``int_row_indices`` is
            not a tuple/list of ints, if ``int_format`` or
            ``float_format`` is not a string, or if ``render`` is not a
            boolean.
        ValueError: If ``column_labels`` does not match the number of
            columns in ``df``, if ``row_labels`` does not contain exactly
            nine elements, if ``int_row_indices`` contains an index
            outside ``range(9)``, or if ``round_threshold`` is negative.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")

    if column_labels is not None:
        if not isinstance(column_labels, (tuple, list)):
            raise TypeError(
                "'column_labels' must be None or a tuple/list of strings."
            )
        if len(column_labels) != df.shape[1]:
            raise ValueError(
                f"'column_labels' must have {df.shape[1]} elements to "
                f"match 'df.columns', got {len(column_labels)}."
            )
        if not all(isinstance(c, str) for c in column_labels):
            raise TypeError(
                "All elements of 'column_labels' must be strings."
            )

    if not isinstance(param_col_title, str):
        raise TypeError("'param_col_title' must be a string.")

    if not isinstance(row_labels, (tuple, list)) or len(row_labels) != 9:
        raise ValueError(
            "'row_labels' must be a tuple/list of exactly 9 elements "
            "(count, mean, std, min, q1, median, q3, max, iqr)."
        )

    if not all(isinstance(r, str) for r in row_labels):
        raise TypeError("All elements of 'row_labels' must be strings.")

    if not isinstance(int_row_indices, (tuple, list)):
        raise TypeError("'int_row_indices' must be a tuple or list of ints.")

    if not all(isinstance(i, int) and 0 <= i < 9 for i in int_row_indices):
        raise ValueError(
            "'int_row_indices' must contain integers in the range [0, 9)."
        )

    if not isinstance(int_format, str) or not isinstance(float_format, str):
        raise TypeError("'int_format' and 'float_format' must be strings.")

    if not isinstance(round_threshold, (int, float)) or round_threshold < 0:
        raise ValueError("'round_threshold' must be a non-negative number.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    def _summarize(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return pd.Series({
            row_labels[0]: len(series),
            row_labels[1]: series.mean(),
            row_labels[2]: series.std(),
            row_labels[3]: series.min(),
            row_labels[4]: q1,
            row_labels[5]: series.median(),
            row_labels[6]: q3,
            row_labels[7]: series.max(),
            row_labels[8]: q3 - q1,
        })

    df_summary = df.apply(_summarize)

    if column_labels is not None:
        df_summary.columns = list(column_labels)

    df_audit = df_summary.reset_index().rename(
        columns={"index": param_col_title}
    )

    numeric_cols = df_audit.columns[1:]

    if round_threshold > 0:
        df_audit[numeric_cols] = df_audit[numeric_cols].map(
            lambda x: 0.0 if abs(x) < round_threshold else x
        )

    int_labels = [row_labels[i] for i in int_row_indices]
    mask_int = df_audit[param_col_title].isin(int_labels)

    table_styles = _table_style_rules() + [
        {
            "selector": "th",
            "props": [("vertical-align", "middle")],
        },
        {
            "selector": "td",
            "props": [("vertical-align", "middle")],
        },
    ]

    styled_audit = (
        df_audit.style
        .hide(axis="index")
        .set_table_styles(table_styles)
        .format(int_format, subset=pd.IndexSlice[mask_int, numeric_cols])
        .format(float_format, subset=pd.IndexSlice[~mask_int, numeric_cols])
    )

    if render:
        _render_styled(styled_audit)

    return df_audit


def plot_wavelet_decomposition(
    coeffs,
    index,
    levels,
    panel_titles,
    ylab,
    wavelet="db4",
    n_obs=None,
    xlab="Date",
    pads=(9, 9, 12),
    label_size=12,
    title_size=12,
    tick_size=11,
    line_width=1.0,
    title_loc="left",
    year_locator_interval=2,
    year_formatter="%Y",
    ylim=None,
    y_step=None,
    y_format=None,
    yticks=None,
    right_yaxis=False,
    figsize=None,
    hspace=0.30,
    wspace=0.08,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot a multi-level wavelet decomposition from pre-computed coefficients.

    Takes the output of ``pywt.wavedec`` (plus the original DatetimeIndex
    and, optionally, the original length) and reconstructs one or more
    coefficient bands, plotting one band per panel. Each entry of
    ``levels`` describes one panel:

    - an ``int`` ``k`` means the panel reconstructs the signal using only
      ``coeffs[k]`` (a single approximation or detail level);
    - a ``tuple`` of ints ``(a, b, ...)`` means the panel collapses those
      coefficients together (reconstructs using only those details).

    For example, ``levels=(0, 1)`` yields two panels (approximation and
    first detail), ``levels=((5, 6, 7),)`` yields a single panel that
    collapses the three finest details, and ``levels=(0, 1, 2, (3, 4))``
    yields four panels. The figure is optionally passed to
    ``render_figure`` for inline rendering.

    Args:
        coeffs (list or tuple of numpy.ndarray): Output of
            ``pywt.wavedec``. Typically
            ``[cA_n, cD_n, cD_{n-1}, ..., cD_1]``.
        index (pandas.DatetimeIndex): Datetime index of the original
            series, used as the x-axis. Must have the same length as the
            original series (``n_obs``).
        levels (tuple or list): One entry per panel. Each entry is either
            an ``int`` (single coefficient index) or a ``tuple`` of ints
            (collapse those indices into one panel).
        panel_titles (tuple or list of str): Titles for each panel. Must
            have the same length as ``levels``.
        ylab (str or tuple/list of str): Y-axis label. If a string, it is
            applied to every panel. If a tuple/list, must have the same
            length as ``levels``.
        wavelet (str): Wavelet name, needed by ``pywt.waverec`` to
            reconstruct. Defaults to ``"db4"``.
        n_obs (int, optional): Original length of the series, used to
            truncate each reconstruction to the same length. If ``None``,
            defaults to ``len(index)``.
        xlab (str): X-axis label applied only to the bottom-most panels.
            Defaults to ``"Date"``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis labels, the y-axis labels and the
            titles, respectively. Defaults to ``(9, 9, 12)``.
        label_size (float): Font size for the axis labels. Defaults to 12.
        title_size (float): Font size for the panel titles. Defaults to
            12.
        tick_size (float): Font size for the tick labels. Defaults to 11.
        line_width (float): Width of the plotted lines. Defaults to 1.0.
        title_loc (str): Title alignment: ``"left"``, ``"center"`` or
            ``"right"``. Defaults to ``"left"``.
        year_locator_interval (int): Interval, in years, between major
            x-axis ticks. Defaults to 2.
        year_formatter (str): ``strftime`` format string for the x-axis.
            Defaults to ``"%Y"``.
        ylim (tuple or list of tuples, optional): Y-axis limits. A single
            ``(ymin, ymax)`` tuple applies to all panels. A list of
            tuples applies per panel. If ``None``, matplotlib defaults
            are used.
        y_step (float or tuple/list, optional): Step for a
            ``MultipleLocator`` on the y-axis. Same broadcasting rules as
            ``ylim``.
        y_format (str or tuple/list, optional): ``printf``-style format
            string for the y-axis ticks. Same broadcasting rules as
            ``ylim``.
        yticks (array-like or tuple/list, optional): Explicit y-axis tick
            positions. A flat sequence applies to all panels. A list of
            sequences applies per panel.
        right_yaxis (bool): If ``True`` and the layout has two columns,
            moves the y-axis label and ticks of the right column to the
            right side. Defaults to ``False``.
        figsize (tuple of float, optional): Figure size in inches. If
            ``None``, a default is chosen based on the number of panels.
        hspace (float): Vertical space between panels. Defaults to 0.30.
            For the 4-panel 2x2 grid this value is automatically adjusted
            so that the absolute vertical gap matches the one produced by
            the 2-panel stack.
        wspace (float): Horizontal space between panels. Defaults to 0.08.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, axes)`` where ``fig`` is the matplotlib figure and
        ``axes`` is a flat numpy array of ``Axes``, one per panel.

    Raises:
        TypeError: If ``coeffs`` is not a list/tuple of numpy arrays, if
            ``index`` is not a ``DatetimeIndex``, if ``levels`` contains
            an element that is neither an int nor a tuple/list of ints,
            if ``panel_titles`` or ``ylab`` (when a sequence) has the
            wrong length or contains non-string elements, or if
            ``figsize`` is not ``None`` or a tuple/list of length 2.
        ValueError: If ``levels`` is empty, if a coefficient index is
            negative or exceeds ``len(coeffs) - 1``, if ``n_obs`` is not
            ``None`` or a positive integer, or if ``n_obs`` exceeds the
            length of ``index``.
    """
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
        raise TypeError(
            "'coeffs' must be a non-empty list or tuple of numpy arrays "
            "(the output of pywt.wavedec)."
        )

    if not all(isinstance(c, np.ndarray) for c in coeffs):
        raise TypeError("All elements of 'coeffs' must be numpy arrays.")

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("'index' must be a pandas DatetimeIndex.")

    if len(index) == 0:
        raise ValueError("'index' is empty.")

    if n_obs is None:
        n_obs = len(index)
    elif (
        not isinstance(n_obs, int)
        or isinstance(n_obs, bool)
        or n_obs <= 0
    ):
        raise ValueError("'n_obs' must be None or a positive integer.")
    elif n_obs > len(index):
        raise ValueError(
            f"'n_obs' ({n_obs}) cannot exceed len(index) ({len(index)})."
        )

    if not isinstance(levels, (tuple, list)) or len(levels) == 0:
        raise TypeError("'levels' must be a non-empty tuple or list.")

    n_coeffs = len(coeffs)
    for item in levels:
        if isinstance(item, int) and not isinstance(item, bool):
            if item < 0:
                raise ValueError(
                    "Coefficient indices in 'levels' must be >= 0."
                )
            if item >= n_coeffs:
                raise ValueError(
                    f"Coefficient index {item} in 'levels' exceeds the "
                    f"available range [0, {n_coeffs - 1}]."
                )
        elif isinstance(item, (tuple, list)):
            if len(item) == 0:
                raise ValueError(
                    "Empty tuples are not valid entries in 'levels'."
                )
            for k in item:
                if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                    raise ValueError(
                        "Collapsed coefficient indices must be "
                        "non-negative integers."
                    )
                if k >= n_coeffs:
                    raise ValueError(
                        f"Coefficient index {k} in 'levels' exceeds the "
                        f"available range [0, {n_coeffs - 1}]."
                    )
        else:
            raise TypeError(
                "Each element of 'levels' must be an int or a tuple/list "
                "of ints."
            )

    n_panels = len(levels)

    if not isinstance(panel_titles, (tuple, list)):
        raise TypeError("'panel_titles' must be a tuple or list of strings.")
    if len(panel_titles) != n_panels:
        raise ValueError(
            f"'panel_titles' must have {n_panels} elements to match "
            f"'levels', got {len(panel_titles)}."
        )
    if not all(isinstance(t, str) for t in panel_titles):
        raise TypeError("All elements of 'panel_titles' must be strings.")

    if isinstance(ylab, str):
        ylab_list = [ylab] * n_panels
    elif isinstance(ylab, (tuple, list)):
        if len(ylab) != n_panels:
            raise ValueError(
                f"'ylab' must have {n_panels} elements to match 'levels', "
                f"got {len(ylab)}."
            )
        if not all(isinstance(y, str) for y in ylab):
            raise TypeError("All elements of 'ylab' must be strings.")
        ylab_list = list(ylab)
    else:
        raise TypeError("'ylab' must be a string or a tuple/list of strings.")

    if not isinstance(wavelet, str):
        raise TypeError("'wavelet' must be a string.")

    _validate_pads(pads)
    _validate_year_locator_interval(year_locator_interval)

    if figsize is not None:
        _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    ylim_list = _broadcast_param(ylim, n_panels, _is_two_numbers, "ylim")
    y_step_list = _broadcast_param(y_step, n_panels, _is_number, "y_step")

    if y_format is None:
        y_format_list = [None] * n_panels
    elif isinstance(y_format, str):
        y_format_list = [y_format] * n_panels
    elif isinstance(y_format, (tuple, list)) and len(y_format) == n_panels:
        y_format_list = list(y_format)
    else:
        raise ValueError(
            f"'y_format' must be a single string or a list of {n_panels} "
            f"strings."
        )

    if yticks is None:
        yticks_list = [None] * n_panels
    elif isinstance(yticks, (tuple, list)):
        if len(yticks) == n_panels and all(
            isinstance(v, (tuple, list)) for v in yticks
        ):
            yticks_list = list(yticks)
        else:
            yticks_list = [yticks] * n_panels
    else:
        raise TypeError(
            "'yticks' must be None, a flat sequence, or a list of "
            "sequences."
        )

    reconstructed = []
    for item in levels:
        c_zero = [np.zeros_like(c) for c in coeffs]
        if isinstance(item, int):
            c_zero[item] = coeffs[item]
        else:
            for k in item:
                c_zero[k] = coeffs[k]
        rec = pywt.waverec(c_zero, wavelet)[:n_obs]
        reconstructed.append(rec)

    if n_panels == 1:
        layout = (1, 1)
    elif n_panels == 2:
        layout = (2, 1)
    elif n_panels == 3:
        layout = (3, 1)
    elif n_panels == 4:
        layout = (2, 2)
    else:
        layout = ((n_panels + 1) // 2, 2)

    n_rows, n_cols = layout

    if figsize is None:
        default_figsizes = {
            1: (8, 3),
            2: (8, 6),
            3: (8, 9),
            4: (10, 8),
        }
        figsize = default_figsizes.get(n_panels, (10, 3 * n_rows))

    if n_panels == 4:
        ref_gap = 0.30 * 6.0 / (2 + 0.30)
        hspace = ref_gap * n_rows / (figsize[1] - ref_gap * (n_rows - 1))

    sharex = n_cols == 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        sharex=sharex,
        squeeze=False,
    )

    axes_flat = axes.flatten()

    pad_x, pad_y, pad_title = pads

    for ax in axes_flat:
        _style_axes(ax, tick_size)

    bottom_indices = set()
    for col in range(n_cols):
        bottom_indices.add((n_rows - 1) * n_cols + col)

    for idx, ax in enumerate(axes_flat):
        if idx >= n_panels:
            ax.set_visible(False)
            continue

        ax.plot(
            index, reconstructed[idx], color="black", linewidth=line_width
        )
        ax.set_title(
            panel_titles[idx],
            fontsize=title_size,
            pad=pad_title,
            loc=title_loc,
        )
        ax.set_ylabel(ylab_list[idx], fontsize=label_size, labelpad=pad_y)

        _apply_yaxis_settings(
            ax,
            ylim=ylim_list[idx],
            y_step=y_step_list[idx],
            y_format=y_format_list[idx],
            yticks=yticks_list[idx],
        )
        _apply_year_axis(ax, year_locator_interval, year_formatter)

        if idx in bottom_indices:
            ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

    if right_yaxis and n_cols == 2:
        for row in range(n_rows):
            idx = row * n_cols + (n_cols - 1)
            if idx < n_panels:
                ax = axes_flat[idx]
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

    plt.tight_layout()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, axes_flat[:n_panels]


def plot_rolling_smoothing_panels(
    series,
    windows=(5, 10, 21),
    titles=None,
    ylab=None,
    xlab="Date",
    center=True,
    pads=(9, 9, 12),
    label_size=12,
    title_size=11,
    tick_size=11,
    line_width=1.0,
    title_loc="center",
    year_locator_interval=2,
    year_formatter="%Y",
    ylim=None,
    y_step=None,
    y_format=None,
    yticks=None,
    right_yaxis=True,
    figsize=(9, 6),
    hspace=0.50,
    wspace=0.08,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot a series plus its rolling means in a 2x2 grid.

    Draws the raw series in the top-left panel and one centered rolling
    mean per window in the remaining panels. The number of panels is
    fixed at four, so ``windows`` must contain exactly three values
    (window for the top-right, bottom-left and bottom-right panels, in
    that order). The figure is optionally passed to ``render_figure`` for
    inline rendering.

    Args:
        series (pandas.Series): Input time series. Must have a
            ``DatetimeIndex``. NaNs are dropped before plotting.
        windows (tuple or list of int): Three rolling window sizes, one
            per smoothed panel. Must contain exactly three positive
            integers. Defaults to ``(5, 10, 21)``.
        titles (tuple or list of str, optional): Titles for the four
            panels in the order ``(raw, w0, w1, w2)``. Must contain
            exactly four strings. If ``None``, titles are auto-generated
            from ``ylab``.
        ylab (str): Y-axis label, applied to every panel. Also used to
            auto-generate titles when ``titles`` is ``None``. Must be
            provided if ``titles`` is ``None``.
        xlab (str): X-axis label applied only to the bottom row of
            panels. Defaults to ``"Date"``.
        center (bool): If ``True``, rolling means use centered windows
            (``center=True`` in ``pandas.Series.rolling``). Defaults to
            ``True``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis labels, the y-axis labels and the
            titles, respectively. Defaults to ``(9, 9, 12)``.
        label_size (float): Font size for the axis labels. Defaults to 12.
        title_size (float): Font size for the panel titles. Defaults to
            11.
        tick_size (float): Font size for the tick labels. Defaults to 11.
        line_width (float): Width of the plotted lines. Defaults to 1.0.
        title_loc (str): Title alignment: ``"left"``, ``"center"`` or
            ``"right"``. Defaults to ``"center"``.
        year_locator_interval (int): Interval, in years, between major
            x-axis ticks. Defaults to 2.
        year_formatter (str): ``strftime`` format string for the x-axis.
            Defaults to ``"%Y"``.
        ylim (tuple or list of tuples, optional): Y-axis limits. A single
            ``(ymin, ymax)`` tuple applies to all panels. A list of four
            tuples applies per panel. If ``None``, matplotlib defaults
            are used.
        y_step (float or tuple/list, optional): Step for a
            ``MultipleLocator`` on the y-axis. Same broadcasting rules as
            ``ylim``.
        y_format (str or tuple/list, optional): ``printf``-style format
            string for the y-axis ticks. Same broadcasting rules as
            ``ylim``.
        yticks (array-like or tuple/list, optional): Explicit y-axis tick
            positions. A flat sequence applies to all panels. A list of
            four sequences (each a tuple, list, numpy array or ``None``)
            applies per panel.
        right_yaxis (bool): If ``True``, moves the y-axis label and ticks
            of the right column to the right side. Defaults to ``True``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(9, 6)``.
        hspace (float): Vertical space between the two rows. Defaults to
            0.50.
        wspace (float): Horizontal space between the two columns.
            Defaults to 0.08.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, axes)`` where ``fig`` is the matplotlib figure and
        ``axes`` is a flat numpy array of four ``Axes``, in row-major
        order (raw, w0, w1, w2).

    Raises:
        TypeError: If ``series`` has no ``dropna`` method, if its index
            is not a ``DatetimeIndex``, if ``windows`` is not a
            tuple/list of three ints, if ``titles`` (when provided) is
            not a tuple/list of four strings, if ``ylab`` is not a string
            when ``titles`` is ``None``, if ``figsize`` is not a
            tuple/list of length 2, or if ``y_format`` is not ``None`` or
            a string.
        ValueError: If ``series`` is empty after dropping NaNs, if any
            window is not a positive integer, if any window is larger
            than the series length, or if ``y_step`` is not ``None`` or a
            positive number.
    """
    if not hasattr(series, "dropna"):
        raise TypeError(
            "'series' must be a pandas Series or similar object with a "
            "'dropna' method."
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("'series' must have a pandas DatetimeIndex.")

    series = series.dropna()

    if len(series) == 0:
        raise ValueError("'series' is empty after dropping NaNs.")

    if not isinstance(windows, (tuple, list)) or len(windows) != 3:
        raise TypeError(
            "'windows' must be a tuple or list of exactly three ints."
        )

    if not all(
        isinstance(w, int) and not isinstance(w, bool) and w > 0
        for w in windows
    ):
        raise ValueError("All elements of 'windows' must be positive ints.")

    if any(w > len(series) for w in windows):
        raise ValueError(
            "Each window in 'windows' must be <= len(series) "
            f"({len(series)})."
        )

    if titles is None:
        if not isinstance(ylab, str):
            raise TypeError(
                "'ylab' must be a string when 'titles' is None "
                "(used to auto-generate panel titles)."
            )
        titles = (
            ylab,
            f"{ylab} (suavizado {windows[0]} días)",
            f"{ylab} (suavizado {windows[1]} días)",
            f"{ylab} (suavizado {windows[2]} días)",
        )
    else:
        if not isinstance(titles, (tuple, list)) or len(titles) != 4:
            raise TypeError(
                "'titles' must be None or a tuple/list of four strings."
            )
        if not all(isinstance(t, str) for t in titles):
            raise TypeError("All elements of 'titles' must be strings.")

    if not isinstance(ylab, str):
        raise TypeError("'ylab' must be a string.")

    _validate_pads(pads)
    _validate_year_locator_interval(year_locator_interval)
    _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    n_panels = 4

    ylim_list = _broadcast_param(ylim, n_panels, _is_two_numbers, "ylim")
    y_step_list = _broadcast_param(y_step, n_panels, _is_number, "y_step")

    if y_format is None:
        y_format_list = [None] * n_panels
    elif isinstance(y_format, str):
        y_format_list = [y_format] * n_panels
    elif isinstance(y_format, (tuple, list)) and len(y_format) == n_panels:
        y_format_list = list(y_format)
    else:
        raise ValueError(
            "'y_format' must be a single string or a list of four strings."
        )

    if yticks is None:
        yticks_list = [None] * n_panels
    elif isinstance(yticks, (tuple, list)):
        if len(yticks) == n_panels and all(
            v is None or isinstance(v, (tuple, list, np.ndarray))
            for v in yticks
        ):
            yticks_list = list(yticks)
        else:
            yticks_list = [yticks] * n_panels
    else:
        raise TypeError(
            "'yticks' must be None, a flat sequence, or a list of "
            "sequences."
        )

    smoothed = [
        series.rolling(window=w, center=center).mean()
        for w in windows
    ]
    panels = [series] + smoothed

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    pad_x, pad_y, pad_title = pads

    for ax in axes_flat:
        _style_axes(ax, tick_size)

    bottom_indices = {2, 3}

    for idx, ax in enumerate(axes_flat):
        ax.plot(
            panels[idx].index,
            panels[idx].values,
            color="black",
            linewidth=line_width,
        )
        ax.set_title(
            titles[idx],
            fontsize=title_size,
            pad=pad_title,
            loc=title_loc,
        )
        ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)

        _apply_yaxis_settings(
            ax,
            ylim=ylim_list[idx],
            y_step=y_step_list[idx],
            y_format=y_format_list[idx],
            yticks=yticks_list[idx],
        )
        _apply_year_axis(ax, year_locator_interval, year_formatter)

        if idx in bottom_indices:
            ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

    if right_yaxis:
        for idx in (1, 3):
            axes_flat[idx].yaxis.set_label_position("right")
            axes_flat[idx].yaxis.tick_right()

    plt.tight_layout()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, axes_flat


def correlation_windows_table(
    series_x,
    series_y,
    windows=(5, 10, 21),
    window_labels=None,
    column_titles=None,
    methods=("pearson", "spearman"),
    star_levels=((0.001, "***"), (0.01, "**"), (0.05, "*")),
    bonferroni=True,
    decimals=3,
    render=True,
):
    """Correlate two series across rolling-mean windows and tabulate results.

    Computes pairwise correlations between ``series_x`` and ``series_y``,
    both on the raw series and on centered rolling means of each at the
    requested windows, and returns a publication-ready table with one row
    per window and one column per correlation method. Correlations are
    formatted with significance stars, optionally Bonferroni-adjusted
    across the number of rows (windows). Optionally renders the styled
    table as centered HTML (useful in Jupyter notebooks).

    Args:
        series_x (pandas.Series): First input series.
        series_y (pandas.Series): Second input series.
        windows (tuple or list of int): Rolling window sizes. Defaults to
            ``(5, 10, 21)``.
        window_labels (tuple or list of str, optional): Row labels, one
            for the raw (unsmoothed) series plus one per window. Must
            contain ``len(windows) + 1`` strings. If ``None``, defaults
            to ``("Unsmoothed", "w=5", "w=10", ...)``.
        column_titles (tuple or list of str, optional): Column names, one
            for the window label plus one per method. Must contain
            ``1 + len(methods)`` strings. If ``None``, defaults to
            ``("Window", "Pearson", "Spearman", ...)`` (methods
            capitalized).
        methods (tuple or list of str): Correlation methods to compute.
            Supported values are ``"pearson"``, ``"spearman"`` and
            ``"kendall"``. Defaults to ``("pearson", "spearman")``.
        star_levels (tuple or list of (float, str) pairs): Significance
            thresholds and their associated stars, in any order. Each
            pair is ``(threshold, stars)`` with ``threshold`` in
            ``(0, 1)``. Defaults to
            ``((0.001, "***"), (0.01, "**"), (0.05, "*"))``.
        bonferroni (bool): If ``True``, each threshold in ``star_levels``
            is divided by the number of rows (number of windows tested,
            including the raw one) before evaluating significance.
            Defaults to ``True``.
        decimals (int): Number of decimals used to format the correlation
            coefficients. Defaults to ``3``.
        render (bool): If ``True``, the styled table is rendered as
            centered HTML via ``IPython.display``. Defaults to ``True``.

    Returns:
        pandas.DataFrame: The plain (unstyled) results table. The styled
        version is built internally and, when ``render=True``, rendered
        as HTML — it is not returned.

    Raises:
        TypeError: If ``series_x`` or ``series_y`` is not a pandas
            Series, if ``windows`` or ``methods`` is not a non-empty
            tuple/list, if ``window_labels`` or ``column_titles`` (when
            provided) has the wrong length or non-string elements, if
            ``star_levels`` is not a tuple/list of (number, string)
            pairs, if ``bonferroni`` or ``render`` is not a boolean, or
            if any of the pair elements is of the wrong type.
        ValueError: If any window is not a positive integer, if any
            method is not supported, if any threshold in ``star_levels``
            is not in ``(0, 1)``, or if ``decimals`` is not a
            non-negative integer.
    """
    if not isinstance(series_x, pd.Series):
        raise TypeError("'series_x' must be a pandas Series.")

    if not isinstance(series_y, pd.Series):
        raise TypeError("'series_y' must be a pandas Series.")

    if not isinstance(windows, (tuple, list)) or len(windows) == 0:
        raise TypeError("'windows' must be a non-empty tuple or list of ints.")

    if not all(
        isinstance(w, int) and not isinstance(w, bool) and w > 0
        for w in windows
    ):
        raise ValueError("All elements of 'windows' must be positive ints.")

    n_rows = len(windows) + 1

    if window_labels is None:
        window_labels = ("Unsmoothed",) + tuple(f"w={w}" for w in windows)
    else:
        if (
            not isinstance(window_labels, (tuple, list))
            or len(window_labels) != n_rows
        ):
            raise TypeError(
                f"'window_labels' must be a tuple/list of {n_rows} strings."
            )
        if not all(isinstance(l, str) for l in window_labels):
            raise TypeError(
                "All elements of 'window_labels' must be strings."
            )

    if not isinstance(methods, (tuple, list)) or len(methods) == 0:
        raise TypeError(
            "'methods' must be a non-empty tuple or list of strings."
        )

    for m in methods:
        if m not in _METHOD_FUNCS:
            raise ValueError(
                f"Unsupported correlation method: '{m}'. "
                f"Valid methods: {sorted(_METHOD_FUNCS)}."
            )

    n_cols = 1 + len(methods)

    if column_titles is None:
        column_titles = ("Window",) + tuple(m.capitalize() for m in methods)
    else:
        if (
            not isinstance(column_titles, (tuple, list))
            or len(column_titles) != n_cols
        ):
            raise TypeError(
                f"'column_titles' must be a tuple/list of {n_cols} strings."
            )
        if not all(isinstance(c, str) for c in column_titles):
            raise TypeError(
                "All elements of 'column_titles' must be strings."
            )

    if not isinstance(star_levels, (tuple, list)) or len(star_levels) == 0:
        raise TypeError(
            "'star_levels' must be a non-empty tuple/list of "
            "(threshold, stars) pairs."
        )

    for item in star_levels:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(
                "Each element of 'star_levels' must be a "
                "(threshold, stars) pair."
            )
        thr, stars = item
        if (
            not isinstance(thr, (int, float))
            or isinstance(thr, bool)
            or not (0 < thr < 1)
        ):
            raise ValueError(
                "Each 'threshold' in 'star_levels' must be a number "
                "in the interval (0, 1)."
            )
        if not isinstance(stars, str):
            raise TypeError(
                "Each 'stars' in 'star_levels' must be a string."
            )

    if not isinstance(bonferroni, bool):
        raise TypeError("'bonferroni' must be a boolean.")

    if (
        not isinstance(decimals, int)
        or isinstance(decimals, bool)
        or decimals < 0
    ):
        raise ValueError("'decimals' must be a non-negative integer.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    sorted_levels = sorted(star_levels, key=lambda x: x[0])

    if bonferroni:
        adj_levels = tuple(
            (thr / n_rows, stars) for thr, stars in sorted_levels
        )
    else:
        adj_levels = tuple(sorted_levels)

    pairs = [(series_x, series_y)]
    for w in windows:
        pairs.append((
            series_x.rolling(window=w, center=True).mean(),
            series_y.rolling(window=w, center=True).mean(),
        ))

    def _format_stars(r, p, levels, dec):
        for thr, stars in levels:
            if p < thr:
                return f"{r:.{dec}f}{stars}"
        return f"{r:.{dec}f}"

    rows = []
    for label, (x, y) in zip(window_labels, pairs):
        pair_df = pd.DataFrame({"x": x, "y": y}).dropna()
        row = {column_titles[0]: label}
        for i, method in enumerate(methods):
            r, p = _METHOD_FUNCS[method](pair_df["x"], pair_df["y"])
            row[column_titles[1 + i]] = _format_stars(
                r, p, adj_levels, decimals
            )
        rows.append(row)

    df_corr_table = pd.DataFrame(rows, columns=list(column_titles))

    styled_corr = (
        df_corr_table.style
        .hide(axis="index")
        .set_table_styles(_table_style_rules())
    )

    if render:
        _render_styled(styled_corr)

    return df_corr_table


def plot_ccf_panels(
    series_x,
    series_y,
    windows=(5, 10, 21),
    max_lag=20,
    normalize=False,
    titles=None,
    ylab="Cross-correlation",
    xlab="Lag",
    center=True,
    pads=(11, 11, 12),
    label_size=12,
    title_size=13,
    tick_size=11,
    stem_width=1.0,
    zero_line_width=0.5,
    conf_color="blue",
    conf_linestyle="--",
    conf_linewidth=1.0,
    title_loc="center",
    xlim=None,
    ylim=(-0.001, 1.00),
    y_step=0.1,
    y_format=None,
    yticks=None,
    hide_right_yaxis=True,
    figsize=(10, 8),
    hspace=0.30,
    wspace=0.08,
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot cross-correlation functions in a 2x2 grid.

    Draws the cross-correlation function (CCF) between two series for
    four variants: the raw (unsmoothed) pair and three centered
    rolling-mean versions at the requested windows. Each panel shows the
    stem plot of the CCF, a horizontal zero line and symmetric
    confidence bounds at ``1.96 / sqrt(n)``. The top-right, bottom-left
    and bottom-right panels use the smoothed variants in the order given
    by ``windows``. The figure is optionally passed to ``render_figure``
    for inline rendering.

    Args:
        series_x (pandas.Series): First input series.
        series_y (pandas.Series): Second input series.
        windows (tuple or list of int): Three rolling window sizes, one
            per smoothed panel. Must contain exactly three positive
            integers. Defaults to ``(5, 10, 21)``.
        max_lag (int): Maximum absolute lag shown on the x-axis. Positive
            integer. Defaults to 20.
        normalize (bool): If ``True``, the CCF values are normalized by
            the product of the two series' standard deviations, yielding
            values in ``[-1, 1]``. If ``False`` (default), raw
            cross-covariance values are shown, matching the unnormalized
            ``statsmodels.tsa.stattools.ccf`` output.
        titles (tuple or list of str, optional): Titles for the four
            panels in the order ``(raw, w0, w1, w2)``. Must contain
            exactly four strings. If ``None``, titles are auto-generated
            from ``windows``.
        ylab (str): Y-axis label applied to the left column. The right
            column hides its y-axis by default. Defaults to
            ``"Cross-correlation"``.
        xlab (str): X-axis label applied only to the bottom row of
            panels. Defaults to ``"Lag"``.
        center (bool): If ``True``, rolling means use centered windows.
            Defaults to ``True``.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis labels, the y-axis labels and the
            titles, respectively. Defaults to ``(11, 11, 12)``.
        label_size (float): Font size for the axis labels. Defaults to 12.
        title_size (float): Font size for the panel titles. Defaults to
            13.
        tick_size (float): Font size for the tick labels. Defaults to 11.
        stem_width (float): Width of the vertical stems. Defaults to 1.0.
        zero_line_width (float): Width of the horizontal zero line.
            Defaults to 0.5.
        conf_color (str): Color of the confidence-bound dashed lines.
            Defaults to ``"blue"``.
        conf_linestyle (str): Line style of the confidence bounds.
            Defaults to ``"--"``.
        conf_linewidth (float): Width of the confidence-bound lines.
            Defaults to 1.0.
        title_loc (str): Title alignment: ``"left"``, ``"center"`` or
            ``"right"``. Defaults to ``"center"``.
        xlim (tuple of float, optional): ``(xmin, xmax)`` limits for the
            x-axis of all panels. If ``None``, matplotlib defaults are
            used.
        ylim (tuple of float, optional): ``(ymin, ymax)`` limits for the
            y-axis of all panels. Defaults to ``(-0.001, 1.00)``.
        y_step (float, optional): If provided, sets a ``MultipleLocator``
            with this step on the y-axis. Defaults to 0.1.
        y_format (str, optional): If provided, a ``printf``-style format
            string for the y-axis ticks. If ``None``, default formatting
            is used.
        yticks (array-like, optional): Explicit y-axis tick positions for
            all panels. If ``None``, matplotlib chooses them (subject to
            ``y_step``).
        hide_right_yaxis (bool): If ``True``, the right column hides its
            y-axis (ticks and label), matching the reference style. If
            ``False``, both columns show their y-axes. Defaults to
            ``True``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(10, 8)``.
        hspace (float): Vertical space between the two rows. Defaults to
            0.30.
        wspace (float): Horizontal space between the two columns.
            Defaults to 0.08.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, axes)`` where ``fig`` is the matplotlib figure and
        ``axes`` is a flat numpy array of four ``Axes`` in row-major
        order (raw, w0, w1, w2).

    Raises:
        TypeError: If ``series_x`` or ``series_y`` is not a pandas
            Series, if ``windows`` is not a tuple/list of three ints, if
            ``titles`` (when provided) is not a tuple/list of four
            strings, if ``max_lag`` is not an int, if ``figsize`` is not
            a tuple/list of length 2, or if ``y_format`` is not ``None``
            or a string.
        ValueError: If any window is not a positive integer, if any
            window is larger than the aligned series length, if
            ``max_lag`` is not positive, or if ``y_step`` is not ``None``
            or a positive number.
    """
    if not isinstance(series_x, pd.Series):
        raise TypeError("'series_x' must be a pandas Series.")

    if not isinstance(series_y, pd.Series):
        raise TypeError("'series_y' must be a pandas Series.")

    if not isinstance(windows, (tuple, list)) or len(windows) != 3:
        raise TypeError(
            "'windows' must be a tuple or list of exactly three ints."
        )

    if not all(
        isinstance(w, int) and not isinstance(w, bool) and w > 0
        for w in windows
    ):
        raise ValueError("All elements of 'windows' must be positive ints.")

    if (
        not isinstance(max_lag, int)
        or isinstance(max_lag, bool)
        or max_lag <= 0
    ):
        raise ValueError("'max_lag' must be a positive integer.")

    if titles is None:
        titles = (
            "Unsmoothed",
            f"Smoothed {windows[0]} days",
            f"Smoothed {windows[1]} days",
            f"Smoothed {windows[2]} days",
        )
    else:
        if not isinstance(titles, (tuple, list)) or len(titles) != 4:
            raise TypeError(
                "'titles' must be None or a tuple/list of four strings."
            )
        if not all(isinstance(t, str) for t in titles):
            raise TypeError("All elements of 'titles' must be strings.")

    if not isinstance(ylab, str):
        raise TypeError("'ylab' must be a string.")

    if not isinstance(xlab, str):
        raise TypeError("'xlab' must be a string.")

    _validate_pads(pads)

    if xlim is not None:
        if not isinstance(xlim, (tuple, list)) or len(xlim) != 2:
            raise TypeError("'xlim' must be None or a tuple/list of length 2.")

    _validate_ylim(ylim)
    _validate_y_step(y_step)
    _validate_y_format(y_format)
    _validate_figsize(figsize)

    if not isinstance(hide_right_yaxis, bool):
        raise TypeError("'hide_right_yaxis' must be a boolean.")

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    pairs = [(series_x, series_y)]
    for w in windows:
        pairs.append((
            series_x.rolling(window=w, center=center).mean(),
            series_y.rolling(window=w, center=center).mean(),
        ))

    def _compute_ccf(x, y, max_lag, normalize):
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(df) <= max_lag:
            raise ValueError(
                f"Aligned series length ({len(df)}) must be greater than "
                f"'max_lag' ({max_lag})."
            )
        lags = np.arange(-max_lag, max_lag + 1)
        vals = np.zeros_like(lags, dtype=float)
        vals[lags >= 0] = ccf(df["x"], df["y"], adjusted=False)[:max_lag + 1]
        vals[lags < 0] = ccf(df["y"], df["x"], adjusted=False)[1:max_lag + 1][::-1]
        if normalize:
            denom = df["x"].std() * df["y"].std()
            if denom > 0:
                vals = vals / denom
        return lags, vals, len(df)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    pad_x, pad_y, pad_title = pads

    for ax in axes_flat:
        _style_axes(ax, tick_size)
        ax.grid(False)

    bottom_indices = {2, 3}

    for idx, ax in enumerate(axes_flat):
        lags, vals, n = _compute_ccf(
            pairs[idx][0], pairs[idx][1], max_lag, normalize
        )
        conf_level = 1.96 / np.sqrt(n)

        ax.vlines(
            lags, ymin=0, ymax=vals, color="black", linewidth=stem_width
        )
        ax.axhline(0, color="black", linewidth=zero_line_width)
        ax.axhline(
            conf_level,
            color=conf_color, linestyle=conf_linestyle,
            linewidth=conf_linewidth,
        )
        ax.axhline(
            -conf_level,
            color=conf_color, linestyle=conf_linestyle,
            linewidth=conf_linewidth,
        )

        ax.set_title(
            titles[idx],
            fontsize=title_size,
            pad=pad_title,
            loc=title_loc,
        )

        _apply_yaxis_settings(
            ax, ylim=ylim, y_step=y_step, y_format=y_format, yticks=yticks
        )
        if xlim is not None:
            ax.set_xlim(*xlim)

        if idx in bottom_indices:
            ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        if idx % 2 == 0:
            ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)
        elif hide_right_yaxis:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, axes_flat


def plot_train_test_split(
    df,
    column,
    train_ratio=0.70,
    title="Time series",
    xlab="Date",
    ylab="Value",
    train_label="Train",
    test_label="Test",
    label_y=0.80,
    split_color="black",
    split_linestyle=":",
    split_linewidth=1.0,
    split_alpha=0.8,
    shade_color="lightgray",
    shade_alpha=0.3,
    shade_extend=pd.Timedelta(days=300),
    line_color="black",
    line_width=1.0,
    pads=(9, 9, 12),
    label_size=10,
    title_size=12,
    tick_size=9,
    title_loc="center",
    year_locator_interval=2,
    year_formatter="%Y",
    figsize=(7, 3.5),
    render=True,
    fmt="svg",
    dpi=100,
):
    """Plot a time series split into train and test regions.

    Splits ``df`` at ``train_ratio`` of its length, draws the series as a
    line on the same axis, adds a vertical dashed line at the split
    point, shades the test region in light gray and annotates both
    regions with text labels placed just above the axis. Applies a
    consistent, publication-ready style (black lines, thin black spines).
    The figure is optionally passed to ``render_figure`` for inline
    rendering.

    Args:
        df (pandas.DataFrame): Input DataFrame. Must have a
            ``DatetimeIndex`` and contain ``column`` as one of its
            columns. It is sorted by index internally before splitting.
        column (str): Name of the column to plot.
        train_ratio (float): Fraction of the rows used for training. Must
            be in ``(0, 1)``. The split index is computed as
            ``int(len(df) * train_ratio)``. Defaults to 0.70.
        title (str): Title of the plot. Defaults to ``"Time series"``.
        xlab (str): X-axis label. Defaults to ``"Date"``.
        ylab (str): Y-axis label. Defaults to ``"Value"``.
        train_label (str): Text label for the training region. Defaults
            to ``"Train"``.
        test_label (str): Text label for the test region. Defaults to
            ``"Test"``.
        label_y (float): Vertical position (in axes coordinates) of the
            region labels. Defaults to 0.80.
        split_color (str): Color of the vertical split line. Defaults to
            ``"black"``.
        split_linestyle (str): Line style of the vertical split line.
            Defaults to ``":"``.
        split_linewidth (float): Width of the vertical split line.
            Defaults to 1.0.
        split_alpha (float): Alpha of the vertical split line. Defaults
            to 0.8.
        shade_color (str): Color of the shaded test region. Defaults to
            ``"lightgray"``.
        shade_alpha (float): Alpha of the shaded test region. Defaults
            to 0.3.
        shade_extend (pandas.Timedelta): Extra time added to the right
            edge of the shaded region beyond the last observation.
            Defaults to 300 days.
        line_color (str): Color of the plotted line. Defaults to
            ``"black"``.
        line_width (float): Width of the plotted line. Defaults to 1.0.
        pads (tuple of float): Padding values ``(pad_x, pad_y, pad_title)``
            applied to the x-axis label, the y-axis label and the title,
            respectively. Defaults to ``(9, 9, 12)``.
        label_size (float): Font size for the axis labels. Defaults to
            10.
        title_size (float): Font size for the title. Defaults to 12.
        tick_size (float): Font size for the tick labels. Defaults to 9.
        title_loc (str): Title alignment: ``"left"``, ``"center"`` or
            ``"right"``. Defaults to ``"center"``.
        year_locator_interval (int): Interval, in years, between major
            x-axis ticks. Defaults to 2.
        year_formatter (str): ``strftime`` format string for the x-axis.
            Defaults to ``"%Y"``.
        figsize (tuple of float): Figure size in inches. Defaults to
            ``(7, 3.5)``.
        render (bool): If ``True``, the figure is passed to
            ``render_figure`` for inline rendering. Defaults to ``True``.
        fmt ({"svg", "png"}): Output format forwarded to ``render_figure``
            when ``render=True``. Defaults to ``"svg"`` (vector).
        dpi (int): Resolution forwarded to ``render_figure``; only takes
            effect when ``fmt="png"``. Defaults to ``100``.

    Returns:
        tuple: ``(fig, ax, df_train, df_test)`` where ``fig`` is the
        matplotlib figure, ``ax`` is the ``Axes``, and ``df_train`` and
        ``df_test`` are the two slices of the input DataFrame (with the
        index sorted).

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame, if its index is
            not a ``DatetimeIndex``, if ``column`` is not a string, if
            ``shade_extend`` is not a ``pd.Timedelta``, or if ``figsize``
            is not a tuple/list of length 2.
        ValueError: If ``df`` is empty, if ``column`` is not in ``df``,
            if ``train_ratio`` is not in ``(0, 1)``, if the split
            produces an empty train or test set, or if any of the
            numeric parameters is out of range.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("'df' must have a pandas DatetimeIndex.")

    if len(df) == 0:
        raise ValueError("'df' is empty.")

    if not isinstance(column, str):
        raise TypeError("'column' must be a string.")

    if column not in df.columns:
        raise ValueError(f"'column' ('{column}') is not a column of 'df'.")

    if (
        not isinstance(train_ratio, (int, float))
        or isinstance(train_ratio, bool)
        or not (0 < train_ratio < 1)
    ):
        raise ValueError("'train_ratio' must be a number in the interval (0, 1).")

    if not isinstance(shade_extend, pd.Timedelta):
        raise TypeError("'shade_extend' must be a pandas Timedelta.")

    _validate_pads(pads)
    _validate_year_locator_interval(year_locator_interval)
    _validate_figsize(figsize)

    if not isinstance(render, bool):
        raise TypeError("'render' must be a boolean.")

    df = df.sort_index()
    split_idx = int(len(df) * train_ratio)

    if split_idx == 0 or split_idx == len(df):
        raise ValueError(
            "'train_ratio' produces an empty train or test set."
        )

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    pad_x, pad_y, pad_title = pads

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title, fontsize=title_size, pad=pad_title, loc=title_loc)

    ax.axvline(
        x=df_test.index[0],
        color=split_color,
        linestyle=split_linestyle,
        linewidth=split_linewidth,
        alpha=split_alpha,
    )

    ax.plot(df_train.index, df_train[column],
            color=line_color, linewidth=line_width)
    ax.plot(df_test.index, df_test[column],
            color=line_color, linewidth=line_width)

    xlim_orig = ax.get_xlim()

    ax.fill_betweenx(
        y=[0, 1],
        x1=df_test.index[0],
        x2=df.index[-1] + shade_extend,
        transform=ax.get_xaxis_transform(),
        color=shade_color,
        alpha=shade_alpha,
    )

    ax.set_xlim(xlim_orig)

    mid_train = df_train.index[len(df_train) // 2]
    mid_test = df_test.index[len(df_test) // 2]

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(
        mid_train, label_y, train_label,
        transform=trans, ha="center", va="bottom",
        fontsize=title_size, color="black",
    )
    ax.text(
        mid_test, label_y, test_label,
        transform=trans, ha="center", va="bottom",
        fontsize=title_size, color="black",
    )

    ax.set_xlabel(xlab, fontsize=label_size, labelpad=pad_x)
    ax.set_ylabel(ylab, fontsize=label_size, labelpad=pad_y)

    _style_axes(ax, tick_size)
    _apply_year_axis(ax, year_locator_interval, year_formatter)

    plt.tight_layout()

    if render:
        render_figure(fig, fmt=fmt, dpi=dpi)

    return fig, ax, df_train, df_test


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _smart_format(x, fmt):
    """Apply a format string to a value, leaving strings and NaNs untouched.

    Args:
        x: Value to format.
        fmt (str): Python format string (e.g. ``"{:,.4f}"``).

    Returns:
        The formatted value if ``x`` is numeric, otherwise ``x`` unchanged.
    """
    if pd.isna(x) or isinstance(x, str):
        return x
    try:
        return fmt.format(x)
    except (ValueError, TypeError):
        return x


def _format_apa_pvalue(val):
    """Format a p-value in APA style (no leading zero, ``< .001`` rule).

    Args:
        val (float): p-value.

    Returns:
        str: Formatted p-value, or ``""`` when ``val`` is NaN.
    """
    if pd.isna(val):
        return ""
    if val < 0.001:
        return "< .001"
    formatted = f"{val:.3f}"
    return formatted[1:] if formatted.startswith("0.") else formatted


def _is_number(x):
    """Return ``True`` if ``x`` is a real number (``bool`` is excluded)."""
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float, np.integer, np.floating))


def _is_two_numbers(x):
    """Return ``True`` if ``x`` is a length-2 sequence of real numbers."""
    return (
        isinstance(x, (tuple, list, np.ndarray))
        and len(x) == 2
        and all(_is_number(v) for v in x)
    )


def _broadcast_param(value, n_panels, single_test, name):
    """Broadcast a per-panel parameter to a list of ``n_panels`` entries.

    Args:
        value: Value to broadcast. ``None`` becomes a list of ``None``.
        n_panels (int): Number of panels.
        single_test (callable): Predicate returning ``True`` when
            ``value`` should be applied to every panel.
        name (str): Parameter name used in error messages.

    Returns:
        list: A list of length ``n_panels``.

    Raises:
        ValueError: If ``value`` is neither ``None``, a single value
            accepted by ``single_test``, nor a sequence of length
            ``n_panels``.
    """
    if value is None:
        return [None] * n_panels
    if single_test(value):
        return [value] * n_panels
    if isinstance(value, (tuple, list)) and len(value) == n_panels:
        return list(value)
    raise ValueError(
        f"'{name}' must be a single value or a list of {n_panels} values."
    )


def _validate_pads(pads):
    """Validate the ``(pad_x, pad_y, pad_title)`` padding tuple."""
    if not isinstance(pads, (tuple, list)):
        raise TypeError("'pads' must be a tuple or list of three numbers.")
    if len(pads) != 3:
        raise ValueError(
            f"'pads' must contain exactly 3 elements (pad_x, pad_y, "
            f"pad_title), got {len(pads)}."
        )


def _validate_figsize(figsize):
    """Validate a ``(width, height)`` figure-size tuple."""
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise TypeError("'figsize' must be a tuple/list of length 2.")


def _validate_ylim(ylim):
    """Validate an optional ``(ymin, ymax)`` y-limits tuple."""
    if ylim is not None:
        if not isinstance(ylim, (tuple, list)) or len(ylim) != 2:
            raise TypeError("'ylim' must be None or a tuple/list of length 2.")


def _validate_y_step(y_step):
    """Validate an optional positive y-axis step."""
    if y_step is not None:
        if not _is_number(y_step) or y_step <= 0:
            raise ValueError("'y_step' must be None or a positive number.")


def _validate_y_format(y_format):
    """Validate an optional ``printf``-style y-axis format string."""
    if y_format is not None and not isinstance(y_format, str):
        raise TypeError("'y_format' must be None or a string.")


def _validate_year_locator_interval(interval):
    """Validate a positive integer used as a year-tick interval."""
    if (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError("'year_locator_interval' must be a positive integer.")


def _style_spines(ax, color="black", linewidth=0.8):
    """Apply the toolkit's spine style to every spine of ``ax``."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)


def _style_ticks(ax, tick_size):
    """Apply the toolkit's tick style to the x and y axes of ``ax``."""
    ax.tick_params(
        axis="x", which="both", bottom=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )
    ax.tick_params(
        axis="y", which="both", left=True, labelsize=tick_size,
        direction="out", length=4, width=0.8, colors="black",
    )


def _style_axes(ax, tick_size):
    """Apply the toolkit's standard axis style (spines + ticks)."""
    _style_spines(ax)
    _style_ticks(ax, tick_size)


def _hide_yaxis(ax):
    """Hide the y-axis ticks and labels of ``ax``."""
    ax.tick_params(
        axis="y", which="both",
        left=False, right=False,
        labelleft=False, labelright=False,
    )


def _apply_yaxis_settings(ax, ylim=None, y_step=None, y_format=None, yticks=None):
    """Apply optional y-axis limits, locator, formatter and ticks to ``ax``."""
    if ylim is not None:
        ax.set_ylim(*ylim)
    if y_step is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_step))
    if y_format is not None:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(y_format))
    if yticks is not None:
        ax.set_yticks(yticks)


def _apply_year_axis(ax, interval=2, formatter="%Y"):
    """Set a year locator and formatter on the x-axis of ``ax``."""
    ax.xaxis.set_major_locator(mdates.YearLocator(interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(formatter))


def _boxplot_kwargs(box_facecolor="white", box_width=0.6):
    """Return the shared keyword arguments used by every boxplot call."""
    return dict(
        widths=box_width,
        patch_artist=True,
        flierprops={
            "marker": "o",
            "markersize": 1.5,
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markeredgewidth": 0.5,
        },
        boxprops={
            "facecolor": box_facecolor,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 0.8},
    )


def _table_style_rules():
    """Return the shared HTML style rules used by every styled table."""
    return [
        {
            "selector": "",
            "props": [
                ("margin-left", "auto"),
                ("margin-right", "auto"),
                ("width", "auto"),
                ("border-collapse", "collapse"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("text-align", "center"),
                ("background-color", "#f2f2f2"),
                ("color", "black"),
                ("font-weight", "bold"),
                ("border", "1px solid black"),
                ("padding", "10px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("text-align", "center"),
                ("border", "1px solid black"),
                ("padding", "10px"),
            ],
        },
        {"selector": "th:first-child", "props": [("text-align", "left")]},
        {"selector": "td:first-child", "props": [("text-align", "left")]},
    ]


def _render_styled(styler):
    """Render a pandas ``Styler`` as centered HTML via ``IPython.display``."""
    display(
        HTML(
            "<div style='text-align: center; width: 100%;'>"
            + styler.to_html()
            + "</div>"
        )
    )


def _resolve_p_adjust_method(method):
    """Resolve an alias and validate a p-adjustment method name."""
    key = method.lower()
    if key in _P_ADJUST_ALIASES:
        key = _P_ADJUST_ALIASES[key]
    if key not in _VALID_P_ADJUST_METHODS:
        raise ValueError(
            f"Unsupported p-adjustment method: '{method}'. "
            f"Valid methods: {sorted(_VALID_P_ADJUST_METHODS)} "
            f"(aliases: {sorted(_P_ADJUST_ALIASES)})."
        )
    return key


def _conclusion_for(test_name, p_value, alpha):
    """Return the conclusion string for a given test name and p-value."""
    if test_name.startswith(("Augmented Dickey-Fuller", "ADF")):
        return "No Estacionaria" if p_value > alpha else "Estacionaria"
    if test_name.startswith(("Kwiatkowski-Phillips", "KPSS")):
        return "No Estacionaria" if p_value < alpha else "Estacionaria"
    if test_name.startswith("Ljung-Box"):
        return (
            "Autocorrelación Significativa"
            if p_value < alpha
            else "Sin Autocorrelación Significativa"
        )
    if test_name.startswith("ARCH-LM"):
        return (
            "Efectos ARCH Significativos"
            if p_value < alpha
            else "Sin Efectos ARCH Significativos"
        )
    return ""