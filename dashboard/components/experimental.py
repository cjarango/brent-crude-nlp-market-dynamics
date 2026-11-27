from dash import html, dcc
import dash_bootstrap_components as dbc

from components.cards import metric_card, panel_heading
from components.cross_validation import cross_validation_layout


NAVY = "#172B4D"
SLATE = "#64748B"
GOLD = "#B79559"
BORDER = "#DCE1E8"

CARD_GAP = "20px"
SECTION_GAP = "24px"

CARD_STYLE = {
    "width": "100%",
    "minWidth": 0,
    "height": "100%",
    "boxSizing": "border-box",
    "margin": 0,
    "padding": "24px",
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "backgroundColor": "#FFFFFF"
}


def tags(items):

    return html.Div(
        [
            html.Span(item)
            for item in items
        ],
        className="research-tags",
        style={
            "marginTop": "16px"
        }
    )


def formula(expression):

    return dcc.Markdown(
        expression,
        mathjax=True,
        style={
            "textAlign": "center",
            "padding": "20px 0",
            "overflowX": "auto",
            "fontSize": "17px",
            "margin": 0
        }
    )


def label(text):

    return html.Div(
        text,
        className="research-label",
        style={
            "marginBottom": "12px"
        }
    )


def paragraph(text, last=False):

    return html.P(
        text,
        style={
            "marginTop": 0,
            "marginBottom": 0 if last else "12px",
            "lineHeight": "1.8"
        }
    )


def card(children, style=None):

    return html.Div(
        children,
        className="research-detail",
        style={
            **CARD_STYLE,
            **(style or {})
        }
    )


def stack(children, gap=CARD_GAP):

    return html.Div(
        children,
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": gap,
            "width": "100%",
            "minWidth": 0,
            "margin": 0,
            "padding": 0
        }
    )


def grid(children, columns=2):

    return html.Div(
        children,
        style={
            "display": "grid",
            "gridTemplateColumns": (
                f"repeat(auto-fit, "
                f"minmax(min(100%, "
                f"max(240px, calc((100% - "
                f"{columns - 1} * 20px) / "
                f"{columns}))), 1fr))"
            ),
            "gap": CARD_GAP,
            "alignItems": "stretch",
            "width": "100%",
            "minWidth": 0,
            "margin": 0,
            "padding": 0,
            "boxSizing": "border-box"
        }
    )


def detail(
    label_text,
    title,
    description,
    items=None
):

    children = [
        label(label_text),

        html.H3(
            title,
            style={
                "marginTop": 0,
                "marginBottom": "12px"
            }
        ),

        paragraph(
            description,
            last=True
        )
    ]

    if items:
        children.append(tags(items))

    return card(children)


def section(
    index,
    title,
    subtitle,
    children
):

    return html.Section(
        [
            html.Div(
                panel_heading(
                    index,
                    title,
                    subtitle
                ),
                style={
                    "margin": 0,
                    "padding": 0,
                    "width": "100%"
                }
            ),

            stack(children)
        ],
        className="panel research-section",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": SECTION_GAP,
            "width": "100%",
            "minWidth": 0,
            "margin": 0,
            "boxSizing": "border-box"
        }
    )


def indicators_grid(indicators):

    return grid(
        [
            html.Div(
                metric_card(
                    title,
                    value,
                    caption
                ),
                style={
                    "width": "100%",
                    "minWidth": 0,
                    "height": "100%"
                }
            )
            for title, value, caption in indicators
        ],
        columns=4
    )


def protocol_indicators():

    indicators = [
        (
            "MODELOS EVALUADOS",
            "11",
            "Arquitecturas predictivas"
        ),
        (
            "MODELOS ESTADÍSTICOS",
            "4",
            "Referencias econométricas e híbridas"
        ),
        (
            "DEEP LEARNING",
            "7",
            "Arquitecturas neuronales"
        ),
        (
            "HORIZONTE",
            "t + 1",
            "Una sesión de negociación"
        )
    ]

    return indicators_grid(indicators)


def model_inventory():

    statistical = [
        "Modelo GARCH",
        "Modelo EGARCH",
        "Modelo HAR",
        "Modelo WTD-HAR"
    ]

    deep_learning = [
        "Gated Recurrent Unit",
        "Temporal Convolutional Network",
        "Modelo BiGRU-TCN",
        "Causal Transformer",
        "Convolutional Transformer",
        "Band Causal TCN",
        "Markov TCN"
    ]

    def inventory(
        label_text,
        models,
        offset
    ):

        return card(
            [
                label(label_text),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    f"{i + offset:02d}",
                                    style={
                                        "color": SLATE,
                                        "fontFamily": "monospace",
                                        "fontSize": "12px"
                                    }
                                ),

                                html.Span(
                                    model,
                                    style={
                                        "fontWeight": "600",
                                        "color": NAVY
                                    }
                                )
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "16px",
                                "padding": "12px 0",
                                "borderBottom": (
                                    f"1px solid {BORDER}"
                                    if i < len(models) - 1
                                    else "none"
                                )
                            }
                        )
                        for i, model in enumerate(models)
                    ],
                    style={
                        "margin": 0,
                        "padding": 0
                    }
                )
            ]
        )

    return grid(
        [
            inventory(
                "MODELOS ECONOMÉTRICOS Y ESTADÍSTICOS",
                statistical,
                1
            ),

            inventory(
                "MODELOS DE DEEP LEARNING",
                deep_learning,
                5
            )
        ],
        columns=2
    )


def general_framework():

    return section(
        "01 / MARCO GENERAL",
        "Diseño del benchmark",
        "Evaluación comparativa de once arquitecturas "
        "bajo un protocolo experimental común.",
        [
            protocol_indicators(),
            model_inventory()
        ]
    )


def temporal_partition():

    return card(
        [
            label("PARTICIÓN CRONOLÓGICA"),

            html.Div(
                [
                    html.Div(
                        "69.5 %",
                        style={
                            "width": "69.5%",
                            "backgroundColor": NAVY,
                            "color": "#FFFFFF",
                            "padding": "18px 8px",
                            "textAlign": "center",
                            "fontWeight": "600",
                            "boxSizing": "border-box"
                        }
                    ),

                    html.Div(
                        "30.5 %",
                        style={
                            "width": "30.5%",
                            "backgroundColor": GOLD,
                            "color": "#FFFFFF",
                            "padding": "18px 8px",
                            "textAlign": "center",
                            "fontWeight": "600",
                            "boxSizing": "border-box"
                        }
                    )
                ],
                style={
                    "display": "flex",
                    "overflow": "hidden",
                    "borderRadius": "4px",
                    "marginTop": "20px",
                    "marginBottom": "24px",
                    "width": "100%"
                }
            ),

            grid(
                [
                    html.Div(
                        [
                            html.H4(
                                "Entrenamiento",
                                style={
                                    "marginBottom": "8px"
                                }
                            ),

                            html.H2(
                                "2,171",
                                style={
                                    "color": NAVY,
                                    "marginBottom": "8px",
                                    "fontFamily": "monospace"
                                }
                            ),

                            paragraph(
                                "03/01/2014 — 03/11/2022",
                                last=True
                            )
                        ],
                        style={
                            "textAlign": "left"
                        }
                    ),

                    html.Div(
                        [
                            html.H4(
                                "Prueba",
                                style={
                                    "marginBottom": "8px"
                                }
                            ),

                            html.H2(
                                "953",
                                style={
                                    "color": NAVY,
                                    "marginBottom": "8px",
                                    "fontFamily": "monospace"
                                }
                            ),

                            paragraph(
                                "07/11/2022 — 21/08/2026",
                                last=True
                            )
                        ],
                        className="text-md-end"
                    )
                ],
                columns=2
            )
        ]
    )


def optimization_protocol():

    indicators = [
        (
            "OPTIMIZACIÓN",
            "Optuna",
            "Búsqueda de hiperparámetros"
        ),
        (
            "PRESUPUESTO MÁXIMO",
            "50",
            "Trials por búsqueda"
        ),
        (
            "EARLY STOPPING",
            "15",
            "Trials consecutivos sin mejora"
        ),
        (
            "FUNCIÓN OBJETIVO",
            "QLIKE",
            "Criterio de selección predictiva"
        )
    ]

    return card(
        [
            label(
                "OPTIMIZACIÓN DE HIPERPARÁMETROS"
            ),

            html.H3(
                "Presupuesto experimental común",
                style={
                    "marginTop": 0,
                    "marginBottom": "12px"
                }
            ),

            paragraph(
                "Todos los modelos, tanto econométricos como "
                "de Deep Learning, fueron sometidos a un "
                "procedimiento de optimización de hiperparámetros "
                "mediante Optuna, con un presupuesto máximo de "
                "50 trials y un criterio de parada anticipada "
                "de 15 trials consecutivos sin mejora en la "
                "función objetivo."
            ),

            html.Div(
                indicators_grid(indicators),
                style={
                    "marginTop": "20px",
                    "marginBottom": "20px"
                }
            ),

            html.P(
                "La asignación de un presupuesto de búsqueda "
                "común permite controlar las diferencias en "
                "las oportunidades de optimización entre "
                "arquitecturas.",
                style={
                    "fontSize": "13px",
                    "color": SLATE,
                    "margin": 0,
                    "lineHeight": "1.8"
                }
            )
        ]
    )


def parameter_optimization():

    return grid(
        [
            detail(
                "DEEP LEARNING",
                "Adaptive Moment Estimation",
                "Las arquitecturas neuronales utilizaron "
                "el optimizador Adam, basado en estimaciones "
                "adaptativas de los momentos de primer y "
                "segundo orden del gradiente. Este mecanismo "
                "permite ajustar individualmente la tasa "
                "efectiva de aprendizaje de los parámetros "
                "durante el entrenamiento.",
                [
                    "Adam",
                    "Gradientes adaptativos",
                    "Optimización neuronal"
                ]
            ),

            detail(
                "CONDICIONES EXPERIMENTALES",
                "Recursos computacionales",
                "Los modelos de Deep Learning fueron "
                "entrenados bajo condiciones computacionales "
                "equivalentes y un presupuesto común de "
                "optimización, manteniendo criterios "
                "homogéneos de búsqueda y selección "
                "de hiperparámetros.",
                [
                    "Mismos recursos",
                    "Presupuesto común",
                    "Comparabilidad experimental"
                ]
            )
        ],
        columns=2
    )


def walk_forward_protocol():

    steps = [
        (
            "PASO 01",
            "Información histórica",
            "Utilizar las observaciones disponibles "
            "hasta el origen del pronóstico."
        ),
        (
            "PASO 02",
            "Pronóstico",
            "Estimar la varianza correspondiente "
            "a la siguiente sesión."
        ),
        (
            "PASO 03",
            "Evaluación",
            "Contrastar la predicción con "
            "la observación realizada."
        ),
        (
            "PASO 04",
            "Actualización",
            "Incorporar la nueva observación "
            "y actualizar los parámetros del modelo."
        )
    ]

    return card(
        [
            label(
                "EVALUACIÓN FUERA DE MUESTRA"
            ),

            html.H3(
                "Walk-forward con actualización secuencial",
                style={
                    "marginTop": 0,
                    "marginBottom": "12px"
                }
            ),

            paragraph(
                "Durante la evaluación en prueba, todos los "
                "modelos actualizaron sus parámetros a medida "
                "que se incorporaban nuevas observaciones. "
                "Cada pronóstico fue generado utilizando "
                "exclusivamente la información disponible "
                "hasta su correspondiente origen temporal."
            ),

            html.Div(
                grid(
                    [
                        detail(
                            label_text,
                            title,
                            description
                        )
                        for label_text, title, description
                        in steps
                    ],
                    columns=4
                ),
                style={
                    "marginTop": "20px",
                    "marginBottom": "20px"
                }
            ),

            html.P(
                "La actualización secuencial reproduce un "
                "escenario operativo de pronóstico, en el "
                "que los modelos se adaptan progresivamente "
                "a la dinámica del mercado sin acceder a "
                "observaciones futuras.",
                style={
                    "fontSize": "13px",
                    "color": SLATE,
                    "margin": 0,
                    "lineHeight": "1.8"
                }
            )
        ]
    )


def temporal_validation():

    return section(
        "02 / ENTRENAMIENTO Y VALIDACIÓN",
        "Protocolo de entrenamiento",
        "Optimización de hiperparámetros, validación "
        "cronológica y actualización secuencial.",
        [
            temporal_partition(),
            optimization_protocol(),
            parameter_optimization(),
            cross_validation_layout(),
            walk_forward_protocol()
        ]
    )


def metrics_table(metrics):

    return html.Div(
        dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Métrica"),
                            html.Th("Descripción")
                        ]
                    )
                ),

                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    name,
                                    style={
                                        "fontWeight": "600",
                                        "whiteSpace": "nowrap"
                                    }
                                ),

                                html.Td(description)
                            ]
                        )
                        for name, description in metrics
                    ]
                )
            ],
            bordered=True,
            striped=True,
            hover=True,
            size="sm",
            style={
                "fontSize": "13px",
                "verticalAlign": "middle",
                "margin": 0,
                "width": "100%",
                "borderCollapse": "collapse"
            }
        ),
        style={
            "width": "100%",
            "overflowX": "auto",
            "border": f"1px solid {BORDER}",
            "boxSizing": "border-box",
            "marginTop": "16px"
        }
    )


def evaluation_metrics():

    point_metrics = [
        (
            "QLIKE",
            "Pérdida asimétrica para pronósticos de varianza."
        ),
        (
            "MSE",
            "Error cuadrático medio."
        ),
        (
            "RMSE",
            "Raíz del error cuadrático medio."
        ),
        (
            "MAE",
            "Error absoluto medio."
        ),
        (
            "HMSE",
            "Error cuadrático medio heterocedástico."
        ),
        (
            "HMAE",
            "Error absoluto medio heterocedástico."
        )
    ]

    probabilistic_metrics = [
        (
            "Pinball Loss",
            "Precisión de los cuantiles predictivos."
        ),
        (
            "CRPS",
            "Calidad de la distribución predictiva."
        )
    ]

    return section(
        "03 / EVALUACIÓN PREDICTIVA",
        "Métricas de evaluación",
        "Evaluación del desempeño puntual y probabilístico "
        "de los pronósticos.",
        [
            card(
                [
                    label("MÉTRICA PRINCIPAL"),

                    html.H2(
                        "QLIKE",
                        style={
                            "fontFamily": "monospace",
                            "fontWeight": "600",
                            "color": NAVY,
                            "marginTop": 0,
                            "marginBottom": "12px"
                        }
                    ),

                    paragraph(
                        "QLIKE constituye el criterio principal "
                        "de evaluación y comparación predictiva "
                        "de los modelos, complementado por "
                        "métricas de error puntual y probabilístico.",
                        last=True
                    )
                ]
            ),

            grid(
                [
                    card(
                        [
                            label(
                                "EVALUACIÓN PUNTUAL"
                            ),

                            html.H3(
                                "Métricas de error",
                                style={
                                    "marginTop": 0,
                                    "marginBottom": "12px"
                                }
                            ),

                            metrics_table(
                                point_metrics
                            )
                        ]
                    ),

                    card(
                        [
                            label(
                                "EVALUACIÓN PROBABILÍSTICA"
                            ),

                            html.H3(
                                "Métricas de distribución",
                                style={
                                    "marginTop": 0,
                                    "marginBottom": "12px"
                                }
                            ),

                            metrics_table(
                                probabilistic_metrics
                            ),

                            html.P(
                                "Complementan la evaluación "
                                "puntual mediante el análisis "
                                "de la incertidumbre asociada "
                                "a los pronósticos.",
                                style={
                                    "marginTop": "20px",
                                    "marginBottom": 0,
                                    "fontSize": "13px",
                                    "color": SLATE,
                                    "lineHeight": "1.8"
                                }
                            )
                        ]
                    )
                ],
                columns=2
            )
        ]
    )


def generalization():

    return section(
        "04 / CAPACIDAD DE GENERALIZACIÓN",
        "Diagnóstico de sobreajuste",
        "Comparación del rendimiento entre "
        "entrenamiento y prueba.",
        [
            card(
                [
                    label(
                        "BRECHA DE GENERALIZACIÓN"
                    ),

                    formula(
                        r"$$\Delta_{\mathrm{abs}}"
                        r"=M_{\mathrm{test}}"
                        r"-M_{\mathrm{train}}$$"
                    ),

                    paragraph(
                        "El protocolo establece umbrales "
                        "de deterioro predictivo para "
                        "identificar posibles problemas "
                        "de generalización.",
                        last=True
                    )
                ]
            ),

            grid(
                [
                    card(
                        [
                            label(
                                "UMBRAL QLIKE"
                            ),

                            html.H2(
                                "0.05",
                                style={
                                    "fontFamily": "monospace",
                                    "color": NAVY,
                                    "marginTop": 0,
                                    "marginBottom": "12px"
                                }
                            ),

                            formula(
                                r"$$\Delta_{\mathrm{QLIKE}}"
                                r">0.05$$"
                            ),

                            paragraph(
                                "Incremento absoluto de la "
                                "pérdida fuera de muestra.",
                                last=True
                            )
                        ]
                    ),

                    card(
                        [
                            label(
                                "MÉTRICAS RESTANTES"
                            ),

                            html.H2(
                                "5 %",
                                style={
                                    "fontFamily": "monospace",
                                    "color": NAVY,
                                    "marginTop": 0,
                                    "marginBottom": "12px"
                                }
                            ),

                            formula(
                                r"$$M_{\mathrm{test}}"
                                r">1.05\,M_{\mathrm{train}}$$"
                            ),

                            paragraph(
                                "Tolerancia relativa para "
                                "las métricas de error y "
                                "las métricas probabilísticas.",
                                last=True
                            )
                        ]
                    )
                ],
                columns=2
            ),

            html.P(
                "Estos umbrales son criterios operativos "
                "del protocolo. Una brecha elevada también "
                "puede estar asociada a cambios en la "
                "distribución temporal y no constituye "
                "por sí sola una demostración de sobreajuste.",
                style={
                    "fontSize": "13px",
                    "color": SLATE,
                    "margin": 0,
                    "lineHeight": "1.8"
                }
            )
        ]
    )


def statistical_comparison():

    methods = [
        (
            "COMPARACIÓN PAREADA",
            "Diebold–Mariano",
            "Contraste de diferencias en la capacidad "
            "predictiva entre pares de modelos mediante "
            "bootstrap de bloques estacionarios.",
            [
                "Comparación predictiva",
                "Stationary Bootstrap"
            ]
        ),
        (
            "MAGNITUD DEL EFECTO",
            "Delta de Cliff",
            "Medida no paramétrica de la magnitud "
            "de las diferencias de desempeño entre "
            "los modelos evaluados.",
            [
                "Tamaño del efecto",
                "Relevancia práctica"
            ]
        ),
        (
            "SELECCIÓN DE MODELOS",
            "Model Confidence Set",
            "Identificación del conjunto superior de "
            "modelos mediante un procedimiento de "
            "selección estadística sobre la pérdida QLIKE.",
            [
                "MCS",
                "QLIKE",
                "Conjunto superior"
            ]
        )
    ]

    return section(
        "05 / COMPARACIÓN ESTADÍSTICA",
        "Contraste entre arquitecturas",
        "Evaluación de diferencias predictivas, magnitud "
        "del efecto y selección estadística de modelos.",
        [
            card(
                [
                    label(
                        "MARCO COMPARATIVO"
                    ),

                    paragraph(
                        "El análisis integra métricas puntuales "
                        "y probabilísticas de evaluación, pruebas "
                        "de comparación predictiva por pares "
                        "mediante la prueba de Diebold–Mariano "
                        "con bootstrap de bloques estacionarios, "
                        "medidas de magnitud del efecto a través "
                        "del delta de Cliff y procedimientos de "
                        "selección del conjunto superior de modelos "
                        "(Model Confidence Set) sobre la pérdida "
                        "QLIKE."
                    ),

                    paragraph(
                        "Este marco comparativo permite determinar "
                        "si las diferencias de desempeño entre "
                        "modelos reflejan ventajas predictivas "
                        "consistentes o variaciones asociadas "
                        "a los regímenes de volatilidad del mercado.",
                        last=True
                    )
                ]
            ),

            grid(
                [
                    detail(
                        label_text,
                        title,
                        description,
                        items
                    )
                    for label_text, title, description, items
                    in methods
                ],
                columns=3
            )
        ]
    )


def experimental_layout():

    return html.Div(
        [
            general_framework(),
            temporal_validation(),
            evaluation_metrics(),
            generalization(),
            statistical_comparison(),

            html.Div(
                [
                    html.Span(
                        "FUENTE: ",
                        style={
                            "fontWeight": "600"
                        }
                    ),

                    html.Span(
                        "Protocolo experimental "
                        "de la investigación."
                    )
                ],
                style={
                    "fontSize": "12px",
                    "color": SLATE,
                    "padding": "0 0 20px",
                    "margin": 0
                }
            )
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": SECTION_GAP,
            "width": "100%",
            "minWidth": 0,
            "paddingTop": "20px",
            "boxSizing": "border-box"
        }
    )
