from dash import html, dcc, dash_table

import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.cards import metric_card

from services.model_loader import (
    MODELS,
    MODEL_CONFIG,
    ECONOMETRIC,
    normalize_model,
    load_model_results
)


NAVY = "#172B4D"
BLACK = "#111111"
BLUE = "#0072BD"
RED = "#E00000"
SLATE = "#64748B"
WHITE = "#FFFFFF"
BACKGROUND = "#F4F6F9"
BORDER = "#DCE1E8"
GRID = "#E9EDF2"


SECTION_STYLE = {
    "backgroundColor": WHITE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "30px",
    "width": "100%",
    "minWidth": 0,
    "boxSizing": "border-box",
    "display": "flex",
    "flexDirection": "column",
    "gap": "24px"
}


DETAIL_STYLE = {
    "backgroundColor": WHITE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "padding": "24px",
    "width": "100%",
    "minWidth": 0,
    "boxSizing": "border-box"
}


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d"
    ]
}


def format_number(value, decimals=4):

    if pd.isna(value):
        return "--"

    if isinstance(value, str):
        return value

    value = float(value)

    if not np.isfinite(value):
        return "--"

    return f"{value:.{decimals}f}"


def format_pvalue(value):

    if pd.isna(value):
        return "--"

    value = float(value)

    if not np.isfinite(value):
        return "--"

    if value < 0.001:
        return "< .001"

    if value == 1:
        return "1.000"

    return f"{value:.3f}".lstrip("0")


def format_difference(value):

    if pd.isna(value):
        return "--"

    value = float(value)

    if not np.isfinite(value):
        return "--"

    return f"{value:+.4f}"


def section_header(number, category, title, description):

    return html.Div(
        [
            html.Div(
                f"{number} / {category}",
                className="research-label",
                style={
                    "marginBottom": "12px"
                }
            ),

            html.H2(
                title,
                style={
                    "color": NAVY,
                    "fontSize": "24px",
                    "fontWeight": 650,
                    "marginTop": 0,
                    "marginBottom": "14px"
                }
            ),

            html.P(
                description,
                style={
                    "color": SLATE,
                    "margin": 0,
                    "lineHeight": "1.8"
                }
            )
        ]
    )


def section(number, category, title, description, children):

    return html.Section(
        [
            section_header(
                number,
                category,
                title,
                description
            ),

            *children
        ],
        style=SECTION_STYLE.copy()
    )


def detail_card(title, children, note=None):

    content = [
        html.H3(
            title,
            style={
                "fontSize": "18px",
                "fontWeight": 600,
                "color": NAVY,
                "marginTop": 0,
                "marginBottom": "20px"
            }
        )
    ]

    if isinstance(children, list):

        content.extend(children)

    else:

        content.append(children)

    if note:

        content.append(
            html.P(
                note,
                style={
                    "fontSize": "13px",
                    "color": SLATE,
                    "lineHeight": "1.7",
                    "marginTop": "16px",
                    "marginBottom": 0
                }
            )
        )

    return html.Div(
        content,
        style=DETAIL_STYLE.copy()
    )


def responsive_grid(children, minimum=240):

    return html.Div(
        children,
        style={
            "display": "grid",
            "gridTemplateColumns": (
                f"repeat(auto-fit, "
                f"minmax(min(100%, {minimum}px), 1fr))"
            ),
            "gap": "20px",
            "width": "100%",
            "minWidth": 0,
            "margin": 0,
            "padding": 0
        }
    )


def base_figure(
    x_title="Fecha",
    y_title="",
    height=380
):

    fig = go.Figure()

    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,

        font={
            "family": "Inter, Arial, sans-serif",
            "size": 12,
            "color": NAVY
        },

        margin={
            "l": 85,
            "r": 30,
            "t": 65,
            "b": 70
        },

        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.10,
            "yanchor": "bottom",
            "font": {
                "size": 12
            }
        },

        hovermode="x unified",

        hoverlabel={
            "bgcolor": WHITE,
            "font_color": NAVY,
            "bordercolor": BORDER
        }
    )

    axis_style = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "showline": True,
        "mirror": True,
        "linewidth": 1,
        "linecolor": "#A0A7B2",
        "ticks": "outside",
        "ticklen": 5,
        "tickwidth": 1,
        "tickcolor": NAVY,
        "zeroline": False,
        "automargin": True
    }

    fig.update_xaxes(
        title_text=x_title,
        title_standoff=18,
        **axis_style
    )

    fig.update_yaxes(
        title_text=y_title,
        title_standoff=18,
        **axis_style
    )

    return fig


def graph(figure):

    return dcc.Graph(
        figure=figure,
        responsive=True,
        config=GRAPH_CONFIG,
        style={
            "height": f"{figure.layout.height}px",
            "minHeight": f"{figure.layout.height}px",
            "width": "100%"
        }
    )


def data_table(df, column_widths=None):

    column_widths = column_widths or {}

    column_styles = [
        {
            "if": {
                "column_id": df.columns[0]
            },
            "textAlign": "left"
        }
    ]

    for column, width in column_widths.items():

        column_styles.append(
            {
                "if": {
                    "column_id": column
                },
                "width": width,
                "minWidth": width
            }
        )

    return html.Div(
        dash_table.DataTable(
            data=df.to_dict("records"),

            columns=[
                {
                    "name": col,
                    "id": col
                }
                for col in df.columns
            ],

            sort_action="none",
            page_action="none",

            style_table={
                "overflowX": "auto",
                "width": "100%"
            },

            style_cell={
                "fontFamily": "Inter, Arial, sans-serif",
                "fontSize": "13px",
                "color": NAVY,
                "padding": "14px 16px",
                "textAlign": "center",
                "whiteSpace": "normal",
                "height": "auto",
                "minWidth": "90px",
                "border": f"1px solid {BORDER}"
            },

            style_cell_conditional=column_styles,

            style_header={
                "backgroundColor": "#EEF1F5",
                "fontWeight": 700,
                "textAlign": "center",
                "border": f"1px solid {BORDER}",
                "color": NAVY
            },

            style_data={
                "backgroundColor": WHITE,
                "border": f"1px solid {BORDER}"
            },

            style_data_conditional=[
                {
                    "if": {
                        "row_index": "odd"
                    },
                    "backgroundColor": "#F7F8FA"
                }
            ],

            css=[
                {
                    "selector": (
                        "td:first-child, th:first-child"
                    ),
                    "rule": (
                        f"border-left: 1px solid "
                        f"{BORDER} !important;"
                    )
                }
            ]
        ),

        style={
            "border": f"1px solid {BORDER}",
            "width": "100%",
            "overflow": "hidden",
            "boxSizing": "border-box"
        }
    )


def description_section(result):

    config = result["config"]
    params = config["params"]
    model = result["model"]

    is_econometric = model in ECONOMETRIC

    if model in ("garch", "egarch"):

        context_card = metric_card(
            "ORDEN DEL MODELO",
            f'({params["p"]}, {params["q"]})',
            "Orden autorregresivo y de varianza"
        )

    elif model in ("har", "wtd_har"):

        context_card = metric_card(
            "VENTANAS TEMPORALES",
            f'1 / {params["lag_weekly"]} / {params["lag_monthly"]}',
            "Componentes diario, semanal y mensual"
        )

    else:

        context_card = metric_card(
            "VENTANA DE CONTEXTO",
            str(params["look_back"]),
            "Observaciones hist\u00f3ricas"
        )

    cards = [
        context_card,

        metric_card(
            "HORIZONTE",
            "1",
            "Sesi\u00f3n burs\u00e1til"
        ),

        metric_card(
            "FAMILIA",
            "Econom\u00e9trico" if is_econometric else "DL",
            "Series de tiempo" if is_econometric else "Deep Learning"
        ),

        metric_card(
            "EJECUCI\u00d3N",
            config["execution"].replace("exec_", "") or "Final",
            "Configuraci\u00f3n seleccionada"
        )
    ]

    return section(
        "01",
        "DESCRIPCI\u00d3N DEL MODELO",
        result["name"],
        config["description"],
        [
            responsive_grid(cards)
        ]
    )


def learning_figure(history, best_epoch, model=None):

    fig = base_figure(
        x_title="Época",
        y_title="Quasi-Likelihood (QLIKE)",
        height=420
    )

    train_loss = pd.to_numeric(
        history["train_loss"],
        errors="coerce"
    )

    val_loss = pd.to_numeric(
        history["val_loss"],
        errors="coerce"
    )

    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=train_loss,
            mode="lines",
            name="Entrenamiento",
            line={
                "color": BLUE,
                "width": 1.8
            },
            hovertemplate=(
                "Entrenamiento: %{y:.6f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["epoch"],
            y=val_loss,
            mode="lines",
            name="Validación",
            line={
                "color": RED,
                "width": 1.8
            },
            hovertemplate=(
                "Validación: %{y:.6f}"
                "<extra></extra>"
            )
        )
    )

    all_losses = pd.concat(
        [
            train_loss,
            val_loss
        ]
    )

    all_losses = all_losses.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    lower = min(
        0,
        float(all_losses.min())
    )

    upper = float(
        all_losses.quantile(0.95)
    )

    upper = upper + max(
        (upper - lower) * 0.06,
        1e-8
    )

    fig.add_trace(
        go.Scatter(
            x=[best_epoch, best_epoch],
            y=[None, None],
            mode="lines",
            name="Mejores pesos",
            line={
                "color": BLACK,
                "width": 1.2,
                "dash": "dot"
            },
            hoverinfo="skip"
        )
    )

    best = history.loc[
        history["epoch"] == best_epoch,
        "val_loss"
    ]

    if not best.empty:

        fig.add_trace(
            go.Scatter(
                x=[best_epoch],
                y=[float(best.iloc[0])],
                mode="markers",
                showlegend=False,
                marker={
                    "size": 7,
                    "color": BLACK
                },
                hovertemplate=(
                    "Mejor época: %{x}<br>"
                    "QLIKE: %{y:.6f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_yaxes(
        type="linear",
        range=[lower, upper]
    )

    fig.update_xaxes(
        dtick=5,
        range=[
            0,
            int(history["epoch"].max()) + 1
        ]
    )

    fig.update_layout(
        legend={
            "orientation": "v",
            "x": 0.97,
            "xanchor": "right",
            "y": 0.97,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.90)",
            "font": {
                "size": 12
            }
        }
    )

    if model == "tcn":

        fig.update_yaxes(
            type="linear",
            autorange=False,
            range=[0, 5],
            tickmode="linear",
            tick0=0,
            dtick=0.5
        )

        fig.update_xaxes(
            autorange=False,
            range=[0, 44],
            tickmode="linear",
            tick0=0,
            dtick=4
        )

    fig.add_vline(
        x=best_epoch,
        line_color=BLACK,
        line_width=1.2,
        line_dash="dot"
    )

    return fig


def learning_card(result):

    return detail_card(
        "Curva de aprendizaje",
        graph(
            learning_figure(
                result["history"],
                result["best_epoch"], result["model"]
            )
        ),
        note=(
            "Evolución de QLIKE durante el entrenamiento "
            "y la validación. La línea vertical discontinua "
            "identifica la época con menor pérdida "
            "de validación."
        )
    )


def econometric_training_section(result):

    optimization = result["optimization"]
    parameters = result["parameters"].copy()

    cards = [
        metric_card(
            "OPTIMIZACI\u00d3N",
            "Optuna",
            "B\u00fasqueda de hiperpar\u00e1metros"
        ),

        metric_card(
            "REGISTROS DE VALIDACI\u00d3N",
            str(len(optimization)),
            "Resultados experimentales"
        ),

        metric_card(
            "CRITERIO PRINCIPAL",
            "QLIKE",
            "Evaluaci\u00f3n predictiva"
        ),

        metric_card(
            "VALIDACI\u00d3N",
            "Temporal",
            "Walk-forward"
        )
    ]

    return section(
        "02",
        "PROCESO DE ENTRENAMIENTO",
        "Optimizaci\u00f3n y estimaci\u00f3n",
        (
            "Configuraci\u00f3n seleccionada mediante "
            "validaci\u00f3n temporal y estimaci\u00f3n "
            "de los par\u00e1metros del modelo."
        ),
        [
            responsive_grid(cards),

            detail_card(
                "Hiperpar\u00e1metros del modelo",
                data_table(
                    parameters,
                    column_widths={
                        "Hiperpar\u00e1metro": "40%",
                        "Identificador": "35%",
                        "Valor": "25%"
                    }
                )
            )
        ]
    )


def training_section(result):

    if result["model"] in ECONOMETRIC:
        return econometric_training_section(result)

    history = result["history"]

    best_epoch = result["best_epoch"]

    best_loss = history["val_loss"].min()

    cards = [
        metric_card(
            "ÉPOCAS REGISTRADAS",
            str(len(history)),
            "Historial de entrenamiento"
        ),

        metric_card(
            "MEJOR ÉPOCA",
            str(best_epoch),
            "Mínima pérdida de validación"
        ),

        metric_card(
            "PÉRDIDA DE VALIDACIÓN",
            format_number(best_loss),
            "Mínimo registrado"
        ),

        metric_card(
            "OPTIMIZACIÓN",
            "Optuna",
            "Búsqueda de hiperparámetros"
        )
    ]

    parameters = result["parameters"].copy()

    return section(
        "02",
        "PROCESO DE ENTRENAMIENTO",
        "Optimización y convergencia",
        (
            "Configuración seleccionada y evolución "
            "del aprendizaje durante el entrenamiento "
            "de la arquitectura."
        ),
        [
            responsive_grid(cards),

            detail_card(
                "Hiperparámetros del modelo",
                data_table(
                    parameters,
                    column_widths={
                        "Hiperparámetro": "40%",
                        "Identificador": "35%",
                        "Valor": "25%"
                    }
                )
            ),

            learning_card(result)
        ]
    )


def evaluation_indicators(partition):

    metrics = partition["metrics"]

    predictions = partition["predictions"]

    cards = [
        metric_card(
            "OBSERVACIONES",
            f"{len(predictions):,}",
            "Predicciones evaluadas"
        ),

        metric_card(
            "QLIKE",
            format_number(
                metrics["QLIKE"]
            ),
            "Criterio principal"
        ),

        metric_card(
            "RMSE",
            format_number(
                metrics["RMSE"],
                decimals=6
            ),
            "Raíz del error cuadrático medio"
        ),

        metric_card(
            "MAE",
            format_number(
                metrics["MAE"],
                decimals=6
            ),
            "Error absoluto medio"
        )
    ]

    return responsive_grid(cards)


def volatility_figure(df):

    fig = base_figure(
        x_title="Fecha",
        y_title="Volatilidad diaria",
        height=420
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["vol_realized"],
            mode="lines",
            name="Volatilidad realizada",
            line={
                "color": "#808080",
                "width": 1.25
            },
            opacity=0.8,
            hovertemplate=(
                "Realizada: %{y:.6f}"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["vol_predicted"],
            mode="lines",
            name="Volatilidad predicha",
            line={
                "color": "#0072BD",
                "width": 1.7
            },
            hovertemplate=(
                "Predicha: %{y:.6f}"
                "<extra></extra>"
            )
        )
    )

    fig.update_xaxes(
        hoverformat="%d/%m/%Y"
    )

    fig.update_yaxes(
        rangemode="tozero",
        tickformat=".3f"
    )

    return fig


def prediction_card(df, partition):

    title = (
        "Volatilidad realizada y predicha en entrenamiento"
        if partition == "train"
        else "Volatilidad realizada y predicha en prueba"
    )

    return detail_card(
        title,
        graph(
            volatility_figure(df)
        )
    )


def residual_diagnostics_card(diagnostics):

    df = diagnostics.copy()

    df["Prueba Estadística"] = (
        df["Prueba Estadística"]
        .str.replace(
            "Rezago",
            "Lag",
            regex=False
        )
        .str.replace(
            "Dimensión 2",
            "Dim 2",
            regex=False
        )
    )

    df["Estadístico"] = df[
        "Estadístico"
    ].apply(
        lambda value: format_number(
            value,
            decimals=4
        )
    )

    df["p-valor"] = df[
        "p-valor"
    ].apply(format_pvalue)

    df["p-adj"] = df[
        "p-adj"
    ].apply(format_pvalue)

    return detail_card(
        "Diagnóstico estadístico de residuos",
        data_table(
            df,
            column_widths={
                "Prueba Estadística": "26%",
                "Estadístico": "16%",
                "p-valor": "12%",
                "p-adj": "12%",
                "Conclusión (α = 0.05)": "34%"
            }
        ),
        note=(
            "Nivel de significancia del 5 %. "
            "Corrección de Bonferroni aplicada "
            "a las familias Ljung–Box y ARCH-LM. "
            "Las conclusiones no significativas indican "
            "que no se rechaza la hipótesis nula."
        )
    )


def comparison_card(result):

    df = result["comparison"].copy()

    df["Métrica"] = (
        df["Métrica"]
        .str.replace(
            "τ = ",
            "τ=",
            regex=False
        )
    )

    df["Entrenamiento"] = df[
        "Entrenamiento"
    ].apply(
        lambda value: format_number(
            value,
            decimals=4
        )
    )

    df["Prueba"] = df[
        "Prueba"
    ].apply(
        lambda value: format_number(
            value,
            decimals=4
        )
    )

    df["Diferencia"] = df[
        "Diferencia"
    ].apply(format_difference)

    return detail_card(
        "Comparación del desempeño predictivo",
        data_table(
            df,
            column_widths={
                "Métrica": "28%",
                "Entrenamiento": "24%",
                "Prueba": "24%",
                "Diferencia": "24%"
            }
        ),
        note=(
            "La diferencia corresponde al valor de la "
            "métrica en prueba menos su valor en "
            "entrenamiento. Los cálculos se realizan "
            "antes del redondeo."
        )
    )


def evaluation_section(result, partition_name):

    partition = result[partition_name]

    df = partition["predictions"]

    if partition_name == "train":

        return section(
            "03",
            "EVALUACIÓN EN ENTRENAMIENTO",
            "Desempeño dentro de muestra",
            (
                "Evaluación del ajuste del modelo "
                "sobre el conjunto de entrenamiento."
            ),
            [
                evaluation_indicators(partition),

                prediction_card(
                    df,
                    partition_name
                ),

                residual_diagnostics_card(
                    partition["diagnostics"]
                )
            ]
        )

    return section(
        "04",
        "EVALUACIÓN EN PRUEBA",
        "Desempeño fuera de muestra",
        (
            "Evaluación de la capacidad predictiva "
            "del modelo sobre el conjunto de prueba "
            "mediante actualización secuencial."
        ),
        [
            evaluation_indicators(partition),

            prediction_card(
                df,
                partition_name
            ),

            residual_diagnostics_card(
                partition["diagnostics"]
            ),

            comparison_card(result)
        ]
    )


def pending_model_layout(model):

    name = MODELS.get(
        model,
        model
    )

    return section(
        "01",
        "MODELO",
        name,
        (
            "Los resultados de esta arquitectura "
            "se incorporarán posteriormente."
        ),
        [
            dbc.Alert(
                "Resultados pendientes de integración.",
                color="secondary",
                style={
                    "margin": 0
                }
            )
        ]
    )


def model_results_layout(model):

    model = normalize_model(model)

    if model not in MODEL_CONFIG:

        return pending_model_layout(
            model
        )

    try:

        result = load_model_results(
            model
        )

    except Exception as exc:

        return dbc.Alert(
            [
                html.H4(
                    "No fue posible cargar los resultados",
                    className="alert-heading"
                ),

                html.P(
                    str(exc),
                    style={
                        "margin": 0
                    }
                )
            ],
            color="danger"
        )

    return html.Div(
        [
            description_section(result),

            training_section(result),

            evaluation_section(
                result,
                "train"
            ),

            evaluation_section(
                result,
                "test"
            )
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "24px",
            "width": "100%",
            "minWidth": 0,
            "margin": 0,
            "padding": 0
        }
    )
