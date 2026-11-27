from components.forecast import forecast_card
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from components.research import scientific_context

from components.cards import (
    metric_card,
    panel_heading,
    data_row
)

from components.charts import (
    market_price_chart,
    candlestick_chart,
    partition_chart
)

from services.data_loader import (
    load_raw,
    get_market_snapshot,
    get_dataset_summary
)


GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d"
    ]
}


def hero_section(snapshot):

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "INVESTIGACIÓN CUANTITATIVA EN ENERGÍA / 01",
                        className="hero-eyebrow"
                    ),
                    html.H1(
                        [
                            "Pronosticar la incertidumbre.",
                            html.Br(),
                            "Comprender los mercados energéticos."
                        ],
                        className="hero-title"
                    ),
                    html.P(
                        "Petróleo Brent · Econometría · Aprendizaje profundo",
                        className="hero-description"
                    )
                ],
                className="hero-content"
            ),
            html.Div(
                [
                    html.Div(
                        "DATOS HISTÓRICOS",
                        className="hero-eyebrow"
                    ),
                    html.Div(
                        snapshot["date"].strftime("%d/%m/%Y").upper(),
                        className="hero-date"
                    ),
                    html.Div(
                        "Última observación disponible",
                        className="hero-description"
                    )
                ],
                className="hero-aside"
            )
        ],
        className="research-hero"
    )


def market_metrics(snapshot):

    cards = [
        (
            "ÚLTIMO CIERRE",
            f"${snapshot['close']:,.2f}",
            "USD por barril"
        ),
        (
            "RETORNO LOGARÍTMICO DIARIO",
            f"{snapshot['log_return'] * 100:+.2f}%",
            "Última sesión de negociación"
        ),
        (
            "VARIANZA REALIZADA",
            f"{snapshot['variance']:.6f}",
            "Estimador Rogers–Satchell"
        ),
        (
            "VOLATILIDAD DIARIA",
            f"{snapshot['volatility'] * 100:.2f}%",
            "Raíz cuadrada de la varianza realizada"
        )
    ]

    return html.Div(
        [
            metric_card(label, value, caption)
            for label, value, caption in cards
        ] + [
            forecast_card()
        ],
        className="context-metrics-grid mb-4"
    )


def market_overview():

    return html.Div(
        [
            html.Div(
                [
                    panel_heading(
                        "01 / PANORAMA DEL MERCADO",
                        "Futuros del petróleo Brent",
                        "BZ=F · USD por barril · Observaciones diarias"
                    ),
                    dcc.RadioItems(
                        id="context-chart-type",
                        options=[
                            {
                                "label": "Precio",
                                "value": "line"
                            },
                            {
                                "label": "Velas",
                                "value": "candlestick"
                            }
                        ],
                        value="line",
                        className="chart-switch",
                        inputClassName="chart-switch-input",
                        labelClassName="chart-switch-label"
                    )
                ],
                className="chart-header"
            ),
            dcc.Loading(
                dcc.Graph(
                    id="context-market-chart",
                    figure=market_price_chart(load_raw()),
                    config=GRAPH_CONFIG
                ),
                type="circle",
                color="#B79559"
            ),
            html.Div(
                [
                    html.Span(
                        "FUENTE: YAHOO FINANCE / BZ=F"
                    ),
                    html.Span(
                        "DATOS HISTÓRICOS DIARIOS"
                    )
                ],
                className="chart-footer"
            )
        ],
        className="panel"
    )


def research_context():

    return html.Div(
        [
            panel_heading(
                "02 / CONTEXTO DE INVESTIGACIÓN",
                "¿Por qué importa la volatilidad?"
            ),
            html.Div(
                html.P(
                    "La incertidumbre del precio del petróleo afecta la gestión del riesgo financiero, las decisiones de inversión y la planificación económica en los mercados energéticos mundiales."
                ),
                className="research-note"
            ),
            html.Hr(className="section-divider"),
            html.H3(
                "Objetivo de investigación",
                className="subsection-title"
            ),
            html.P(
                "Evaluar el desempeño predictivo de modelos econométricos y de aprendizaje profundo en el pronóstico a un paso de la varianza del petróleo Brent.",
                className="page-description"
            ),
            html.Hr(className="section-divider"),
            html.H3(
                "Marco de pronóstico",
                className="subsection-title"
            ),
            data_row("Activo", "Futuros del Brent"),
            data_row("Frecuencia", "Diaria"),
            data_row("Variable objetivo", "Varianza realizada"),
            data_row("Horizonte de pronóstico", "t + 1"),
            data_row("Tarea", "Regresión de series temporales")
        ],
        className="panel"
    )


def dataset_card(summary):

    return html.Div(
        [
            panel_heading(
                "03 / DOCUMENTACIÓN DE DATOS",
                "Ficha técnica de los datos",
                "Características de los datos y cobertura temporal."
            ),
            html.Div(
                [
                    data_row("Fuente", "Yahoo Finance"),
                    data_row("Símbolo bursátil", "BZ=F"),
                    data_row(
                        "Observaciones originales",
                        f"{summary['raw_observations']:,}"
                    ),
                    data_row(
                        "Observaciones procesadas",
                        f"{summary['processed_observations']:,}"
                    ),
                    data_row(
                        "Variables",
                        str(summary["processed_features"])
                    ),
                    data_row(
                        "Fecha inicial",
                        summary["start_date"].strftime("%d/%m/%Y")
                    ),
                    data_row(
                        "Fecha final",
                        summary["end_date"].strftime("%d/%m/%Y")
                    ),
                    data_row("Frecuencia", "Sesiones diarias de negociación"),
                    data_row("Variable objetivo", "rs_daily")
                ],
                className="mt-3"
            )
        ],
        className="panel"
    )


def validation_card(summary):

    return html.Div(
        [
            panel_heading(
                "04 / DISEÑO EXPERIMENTAL",
                "Validación temporal",
                "Particiones cronológicas de entrenamiento y prueba."
            ),
            dcc.Graph(
                figure=partition_chart(summary),
                config={"displayModeBar": False},
                className="partition-graph"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="partition-dot train-dot"
                            ),
                            html.Span("Entrenamiento")
                        ],
                        className="partition-label"
                    ),
                    html.Strong(
                        f"{summary['train_observations']:,}"
                    )
                ],
                className="data-row"
            ),
            html.P(
                f"{summary['train_start']:%d/%m/%Y} — "
                f"{summary['train_end']:%d/%m/%Y}",
                className="partition-date"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="partition-dot test-dot"
                            ),
                            html.Span("Prueba")
                        ],
                        className="partition-label"
                    ),
                    html.Strong(
                        f"{summary['test_observations']:,}"
                    )
                ],
                className="data-row"
            ),
            html.P(
                f"{summary['test_start']:%d/%m/%Y} — "
                f"{summary['test_end']:%d/%m/%Y}",
                className="partition-date"
            ),
            html.Div(
                html.P(
                    "El orden temporal debe preservarse durante el entrenamiento, la validación y la evaluación de los modelos."
                ),
                className="research-note"
            )
        ],
        className="panel"
    )


def limitations_card():

    return html.Div(
        [
            panel_heading(
                "05 / INVESTIGACIÓN RESPONSABLE",
                "Limitaciones y ética"
            ),
            html.Div(
                [
                    html.H3(
                        "Incertidumbre del mercado",
                        className="subsection-title"
                    ),
                    html.P(
                        "Los acontecimientos geopolíticos y económicos inesperados pueden generar cambios estructurales difíciles de anticipar.",
                        className="page-description"
                    ),
                    html.H3(
                        "Medición indirecta",
                        className="subsection-title mt-3"
                    ),
                    html.P(
                        "La varianza realizada se estima a partir de los precios diarios OHLC, en lugar de observar directamente la volatilidad latente.",
                        className="page-description"
                    ),
                    html.H3(
                        "Uso responsable",
                        className="subsection-title mt-3"
                    ),
                    html.P(
                        "Los pronósticos tienen fines de investigación académica y no constituyen asesoramiento de inversión.",
                        className="page-description"
                    )
                ],
                className="mt-3"
            )
        ],
        className="panel"
    )


def layout():

    try:

        snapshot = get_market_snapshot()
        summary = get_dataset_summary()

        return dbc.Container(
            [
                hero_section(snapshot),

                market_metrics(snapshot),

                dbc.Row(
                    [
                        dbc.Col(
                            market_overview(),
                            xs=12,
                            xl=8
                        ),
                        dbc.Col(
                            research_context(),
                            xs=12,
                            xl=4
                        )
                    ],
                    className="g-3 mb-4"
                ),

                dbc.Row(
                    [
                        dbc.Col(
                            dataset_card(summary),
                            xs=12,
                            lg=4
                        ),
                        dbc.Col(
                            validation_card(summary),
                            xs=12,
                            lg=4
                        ),
                        dbc.Col(
                            limitations_card(),
                            xs=12,
                            lg=4
                        )
                    ],
                    className="g-3 mb-4"
                ),

                scientific_context()
            ],
            fluid=True,
            className="p-0"
        )

    except (FileNotFoundError, ValueError) as error:

        return dbc.Alert(
            [
                html.H4("Datos del mercado no disponibles"),
                html.P(str(error))
            ],
            color="warning"
        )

@callback(
    Output("context-market-chart", "figure"),
    Input("context-chart-type", "value")
)
def update_market_chart(chart_type):

    df = load_raw()

    if chart_type == "candlestick":

        return candlestick_chart(
            df.tail(252)
        )

    return market_price_chart(df)