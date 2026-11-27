from dash import html, dcc
import plotly.graph_objects as go

from components.cards import metric_card


NAVY = "#172B4D"
BLUE = "#6485A8"
SLATE = "#64748B"
BLACK = "#111111"
BORDER = "#DCE1E8"
BACKGROUND = "#F4F6F9"
WHITE = "#FFFFFF"

N_TRAIN = 2171
N_TEST = 953
N_SPLITS = 5
WARMUP = 1200
EMBARGO = 1


def get_folds(
    n_samples=N_TRAIN,
    n_splits=N_SPLITS,
    warmup=WARMUP,
    embargo=EMBARGO
):

    val_size = (n_samples - warmup) // n_splits

    folds = []

    for i in range(n_splits):

        val_start = warmup + i * val_size

        val_end = (
            val_start + val_size
            if i < n_splits - 1
            else n_samples
        )

        train_end = val_start - embargo

        folds.append(
            (0, train_end, val_start, val_end)
        )

    return folds


def validation_figure():

    folds = get_folds()

    total = N_TRAIN + N_TEST

    fig = go.Figure()

    segments = [
        (
            "Entrenamiento",
            NAVY,
            [f[1] - f[0] for f in folds],
            [f[0] for f in folds]
        ),
        (
            "Embargo",
            BLACK,
            [f[2] - f[1] for f in folds],
            [f[1] for f in folds]
        ),
        (
            "Validación",
            BLUE,
            [f[3] - f[2] for f in folds],
            [f[2] for f in folds]
        ),
        (
            "Observaciones futuras",
            BACKGROUND,
            [N_TRAIN - f[3] for f in folds],
            [f[3] for f in folds]
        ),
        (
            "Prueba reservada",
            "#E2E8F0",
            [N_TEST for _ in folds],
            [N_TRAIN for _ in folds]
        )
    ]

    fold_names = [
        f"Pliegue {i + 1}"
        for i in range(len(folds))
    ]

    for name, color, sizes, starts in segments:

        customdata = [
            [start, start + size, size]
            for start, size in zip(starts, sizes)
        ]

        fig.add_trace(
            go.Bar(
                name=name,
                y=fold_names,
                x=sizes,
                base=starts,
                orientation="h",
                width=0.62,
                marker={
                    "color": color,
                    "line": {
                        "width": 0
                    }
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + name
                    + "<br>"
                    "Inicio: %{customdata[0]:,}<br>"
                    "Fin exclusivo: %{customdata[1]:,}<br>"
                    "Observaciones: %{customdata[2]:,}"
                    "<extra></extra>"
                )
            )
        )

    for i, fold in enumerate(folds):

        train_start, train_end, val_start, val_end = fold

        fig.add_annotation(
            x=(train_start + train_end) / 2,
            y=fold_names[i],
            text=f"{train_end - train_start:,}",
            showarrow=False,
            font={
                "size": 12,
                "color": WHITE
            }
        )

        fig.add_annotation(
            x=(val_start + val_end) / 2,
            y=fold_names[i],
            text=f"{val_end - val_start:,}",
            showarrow=False,
            font={
                "size": 11,
                "color": WHITE
            }
        )

        fig.add_shape(
            type="line",
            x0=val_start,
            x1=val_start,
            y0=i - 0.36,
            y1=i + 0.36,
            xref="x",
            yref="y",
            line={
                "color": BLACK,
                "width": 1.5
            }
        )

    fig.add_vline(
        x=N_TRAIN,
        line_width=1.5,
        line_dash="dash",
        line_color=SLATE
    )

    fig.add_annotation(
        x=N_TRAIN / 2,
        y=1.14,
        xref="x",
        yref="paper",
        text="<b>Conjunto de entrenamiento</b>",
        showarrow=False,
        font={
            "size": 13,
            "color": NAVY
        }
    )

    fig.add_annotation(
        x=N_TRAIN + N_TEST / 2,
        y=1.14,
        xref="x",
        yref="paper",
        text="<b>Conjunto de prueba</b>",
        showarrow=False,
        font={
            "size": 13,
            "color": SLATE
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=460,
        autosize=True,
        barmode="overlay",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={
            "family": "Inter, Arial, sans-serif",
            "size": 12,
            "color": NAVY
        },
        margin={
            "l": 120,
            "r": 25,
            "t": 85,
            "b": 100
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.23,
            "yanchor": "top",
            "font": {
                "size": 11
            }
        },
        hoverlabel={
            "bgcolor": WHITE,
            "font_color": NAVY,
            "bordercolor": BORDER
        },
        bargap=0.3
    )

    fig.update_xaxes(
        title_text="Observaciones cronológicas",
        range=[0, total],
        tickmode="array",
        tickvals=[
            0,
            400,
            800,
            1200,
            1600,
            2000,
            2400,
            2800,
            3124
        ],
        tickformat=",d",
        showgrid=True,
        gridcolor="#E9EDF2",
        gridwidth=1,
        showline=True,
        mirror=True,
        linecolor=BORDER,
        linewidth=1,
        zeroline=False,
        ticks="outside",
        automargin=True
    )

    fig.update_yaxes(
        autorange=False,
        categoryorder="array",
        categoryarray=fold_names,
        range=[4.6, -0.6],
        ticklabelstandoff=16,
        showgrid=False,
        showline=True,
        mirror=True,
        linecolor=BORDER,
        linewidth=1,
        ticks="",
        automargin=True
    )

    return fig


def validation_indicators():

    indicators = [
        (
            "VALIDACIÓN CRUZADA",
            "5",
            "Pliegues cronológicos"
        ),
        (
            "WARMUP",
            "1,200",
            "Posición inicial de validación"
        ),
        (
            "EMBARGO",
            "1",
            "Observación excluida"
        ),
        (
            "CRITERIO DE SELECCIÓN",
            "QLIKE",
            "Promedio entre pliegues"
        )
    ]

    return html.Div(
        [
            html.Div(
                metric_card(
                    title,
                    value,
                    description
                ),
                style={
                    "minWidth": 0,
                    "width": "100%",
                    "height": "100%"
                }
            )
            for title, value, description in indicators
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": (
                "repeat(auto-fit, minmax(min(100%, 210px), 1fr))"
            ),
            "gap": "20px",
            "width": "100%",
            "margin": 0,
            "padding": 0
        }
    )


def validation_objective():

    return html.Div(
        [
            html.Div(
                "CRITERIO DE OPTIMIZACIÓN",
                className="research-label",
                style={
                    "marginBottom": "12px"
                }
            ),

            html.H3(
                "Selección de hiperparámetros",
                style={
                    "marginTop": 0,
                    "marginBottom": "12px"
                }
            ),

            html.P(
                "Cada configuración de hiperparámetros "
                "propuesta por Optuna se evalúa mediante "
                "validación cruzada temporal. La función "
                "objetivo corresponde al promedio aritmético "
                "de QLIKE obtenido en los pliegues de validación.",
                style={
                    "marginTop": 0,
                    "marginBottom": "12px",
                    "lineHeight": "1.8"
                }
            ),

            dcc.Markdown(
                r"""
$$
\theta^{*}
=
\underset{\theta\in\Theta}{\operatorname{argmin}}
\left[
\frac{1}{K}
\sum_{k=1}^{K}
\mathrm{QLIKE}_{k}(\theta)
\right],
\qquad K=5
$$
                """,
                mathjax=True,
                style={
                    "textAlign": "center",
                    "fontSize": "18px",
                    "padding": "20px 0",
                    "overflowX": "auto"
                }
            ),

            html.P(
                "Adicionalmente, se utiliza MedianPruner "
                "para interrumpir configuraciones con "
                "rendimiento intermedio poco prometedor. "
                "El mecanismo considera cinco trials iniciales "
                "y un periodo de calentamiento de dos pasos "
                "de evaluación.",
                style={
                    "margin": 0,
                    "lineHeight": "1.8"
                }
            )
        ],
        style={
            "borderTop": f"1px solid {BORDER}",
            "paddingTop": "24px",
            "width": "100%",
            "minWidth": 0
        }
    )


def cross_validation_layout():

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "VALIDACIÓN CRUZADA TEMPORAL",
                        className="research-label",
                        style={
                            "marginBottom": "12px"
                        }
                    ),

                    html.H3(
                        "Expanding Window Cross-Validation",
                        style={
                            "marginTop": 0,
                            "marginBottom": "12px"
                        }
                    ),

                    html.P(
                        "La selección de hiperparámetros se realiza "
                        "mediante validación cruzada con ventanas "
                        "expansivas, preservando el orden cronológico "
                        "de las observaciones. En cada pliegue, el conjunto "
                        "de entrenamiento incorpora progresivamente "
                        "información histórica, mientras que la "
                        "validación se efectúa sobre un bloque "
                        "temporal posterior.",
                        style={
                            "marginTop": 0,
                            "marginBottom": "12px",
                            "lineHeight": "1.8"
                        }
                    ),

                    html.P(
                        "Se establece un embargo temporal entre "
                        "ambas particiones para reducir el riesgo "
                        "de fuga de información. El conjunto de "
                        "prueba permanece excluido del proceso "
                        "de selección de hiperparámetros.",
                        style={
                            "margin": 0,
                            "lineHeight": "1.8"
                        }
                    )
                ]
            ),

            validation_indicators(),

            html.Div(
                [
                    dcc.Graph(
                        figure=validation_figure(),
                        responsive=True,
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "displayModeBar": False
                        },
                        style={
                            "height": "460px",
                            "minHeight": "460px",
                            "width": "100%"
                        }
                    ),

                    html.P(
                        "Esquema de referencia generado a partir "
                        "del algoritmo de partición de "
                        "WalkForwardTrainer, utilizando sus "
                        "valores predeterminados de warmup, "
                        "embargo y número de pliegues. Los valores "
                        "sobre las barras indican el número de "
                        "observaciones de cada partición. "
                        "El embargo se resalta mediante una "
                        "línea negra para facilitar su visualización.",
                        style={
                            "fontSize": "12px",
                            "color": SLATE,
                            "margin": 0,
                            "lineHeight": "1.7"
                        }
                    )
                ],
                style={
                    "width": "100%",
                    "minWidth": 0
                }
            ),

            validation_objective()
        ],
        className="research-detail",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "24px",
            "width": "100%",
            "minWidth": 0,
            "boxSizing": "border-box",
            "margin": 0,
            "padding": "24px",
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "backgroundColor": WHITE
        }
    )
