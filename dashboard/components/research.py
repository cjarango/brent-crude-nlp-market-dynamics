from dash import html, dcc
import dash_bootstrap_components as dbc

from components.cards import metric_card, panel_heading


def literature_section():

    indicators = [
        ("REGISTROS IDENTIFICADOS", "365", "Búsqueda bibliográfica inicial"),
        ("ESTUDIOS INCLUIDOS", "16", "Corpus final de investigación"),
        ("ARQUITECTURAS DE APRENDIZAJE PROFUNDO", "10", "Modelos seleccionados para comparación"),
        ("MODELOS DE REFERENCIA", "6", "Pronósticos econométricos e ingenuos")
    ]

    return html.Section(
        [
            panel_heading(
                "06 / EVIDENCIA CIENTÍFICA",
                "Revisión sistemática de literatura",
                "Selección de arquitecturas de pronóstico basada en evidencia."
            ),

            dbc.Row(
                [
                    dbc.Col(
                        metric_card(label, value, caption),
                        xs=12,
                        sm=6,
                        xl=3
                    )
                    for label, value, caption in indicators
                ],
                className="g-3 research-indicators"
            ),

            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    "PROTOCOLO DE REVISIÓN",
                                    className="research-label"
                                ),
                                html.H3("PRISMA 2020"),
                                html.P(
                                    "Identificación, cribado y selección sistemática de estudios relevantes."
                                ),
                                html.Div(
                                    [
                                        html.Span("Scopus"),
                                        html.Span("IEEE Xplore"),
                                        html.Span("Web of Science")
                                    ],
                                    className="research-tags"
                                )
                            ],
                            className="research-detail"
                        ),
                        xs=12,
                        lg=5
                    ),

                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    "BRECHA DE INVESTIGACIÓN",
                                    className="research-label"
                                ),
                                html.H3(
                                    "Comparabilidad y reproducibilidad"
                                ),
                                html.P(
                                    "Las diferencias en las particiones de datos, los protocolos de evaluación y las métricas de pronóstico limitan la comparación directa entre arquitecturas publicadas. Se requiere un marco experimental común para evaluar el desempeño predictivo en condiciones comparables."
                                )
                            ],
                            className="research-detail"
                        ),
                        xs=12,
                        lg=7
                    )
                ],
                className="g-3"
            )
        ],
        className="panel research-section"
    )


def question_card(code, category, question, h0, h1, tests):

    return html.Article(
        [
            html.Div(
                [
                    html.Span(
                        code,
                        className="question-index"
                    ),
                    html.Span(
                        category,
                        className="research-label"
                    )
                ],
                className="question-header"
            ),

            html.H3(
                question,
                className="question-title"
            ),

            html.Div(
                [
                    html.Span(
                        "H0",
                        className="hypothesis-symbol"
                    ),
                    html.P(h0)
                ],
                className="hypothesis-row"
            ),

            html.Div(
                [
                    html.Span(
                        "H1",
                        className="hypothesis-symbol"
                    ),
                    html.P(h1)
                ],
                className="hypothesis-row"
            ),

            html.Div(
                [
                    html.Div(
                        "EVALUACIÓN ESTADÍSTICA",
                        className="research-label"
                    ),
                    html.Div(
                        [html.Span(test) for test in tests],
                        className="research-tags"
                    )
                ],
                className="question-tests"
            )
        ],
        className="question-card"
    )


def research_questions_section():

    questions = [
        (
            "RQ1",
            "ADECUACIÓN DEL MODELO",
            "¿Las arquitecturas evaluadas capturan las dependencias lineales y no lineales de la volatilidad del Brent?",
            "Los residuos conservan estructura lineal o no lineal, lo que indica una captura incompleta de la dinámica de la volatilidad.",
            "Los residuos se comportan como ruido blanco, sin autocorrelación remanente, efectos ARCH ni dependencia no lineal.",
            ["Ljung-Box", "ARCH-LM", "BDS"]
        ),
        (
            "RQ2",
            "DESEMPEÑO PREDICTIVO",
            "¿Las arquitecturas de aprendizaje profundo superan a los modelos econométricos tradicionales bajo un protocolo sin fuga de información?",
            "No existe una diferencia significativa en la pérdida QLIKE entre los modelos de aprendizaje profundo y los econométricos de referencia.",
            "Los modelos de aprendizaje profundo reducen significativamente la pérdida QLIKE frente a los modelos econométricos de referencia.",
            ["Diebold-Mariano", "Corrección HLN", "MCS"]
        ),
        (
            "RQ3",
            "COMPARACIÓN DE ARQUITECTURAS",
            "¿Las arquitecturas híbridas superan a las arquitecturas puras de aprendizaje profundo?",
            "El rango promedio de desempeño de los modelos híbridos no difiere del de las arquitecturas puras.",
            "Las arquitecturas híbridas alcanzan un rango promedio de desempeño significativamente mejor.",
            ["Friedman", "Nemenyi"]
        )
    ]

    return html.Section(
        [
            panel_heading(
                "07 / PREGUNTAS DE INVESTIGACIÓN",
                "Preguntas e hipótesis",
                "Hipótesis estadísticas definidas en el protocolo experimental."
            ),

            dbc.Row(
                [
                    dbc.Col(
                        question_card(*question),
                        xs=12,
                        lg=4
                    )
                    for question in questions
                ],
                className="g-3 mt-2"
            )
        ],
        className="panel research-section"
    )


def methodology_step(number, title, description, tags):

    return html.Div(
        [
            html.Div(
                number,
                className="methodology-number"
            ),

            html.Div(
                [
                    html.H3(title),
                    html.P(description),

                    html.Div(
                        [html.Span(tag) for tag in tags],
                        className="research-tags"
                    )
                ],
                className="methodology-content"
            )
        ],
        className="methodology-step"
    )


def dcc_formula():

    return dcc.Markdown(
        r"""
$$
\mathrm{QLIKE}_{\epsilon} =
\frac{1}{T}\sum_{t=1}^{T}
\left[
\frac{y_t+\epsilon}{\hat{y}_t+\epsilon}
-\ln\left(\frac{y_t+\epsilon}{\hat{y}_t+\epsilon}\right)
-1
\right]
$$

$$
\epsilon = 10^{-8}
$$
        """,
        mathjax=True,
        className="research-formula"
    )


def evaluation_panel():

    return html.Div(
        [
            html.Div(
                "ENTRENAMIENTO Y OPTIMIZACIÓN",
                className="research-label"
            ),

            html.Div(
                [
                    html.Span("PyTorch"),
                    html.Span("Optuna"),
                    html.Span("Parada temprana"),
                    html.Span("Validación progresiva")
                ],
                className="research-tags"
            ),

            html.P(
                "Optimización de hiperparámetros bajo un protocolo común de validación temporal, respetando el orden cronológico y procurando presupuestos computacionales comparables entre arquitecturas.",
                className="page-description mt-3"
            ),

            html.Hr(className="section-divider"),

            html.Div(
                "ESTIMADOR DE VARIANZA REALIZADA",
                className="research-label"
            ),

            html.H3(
                "Rogers-Satchell (RS)",
                className="evaluation-title"
            ),

            dcc.Markdown(
                r"""
$$
\mathrm{RS}_t =
\ln\left(\frac{H_t}{O_t}\right)
\ln\left(\frac{H_t}{C_t}\right)
+
\ln\left(\frac{L_t}{O_t}\right)
\ln\left(\frac{L_t}{C_t}\right)
$$
                """,
                mathjax=True,
                style={
                    "textAlign": "center",
                    "fontSize": "20px",
                    "color": "#172B4D",
                    "padding": "25px 0"
                }
            ),

            html.P(
                "Estimador de varianza diaria basado en los precios de "
                "apertura, m\u00e1ximo, m\u00ednimo y cierre (OHLC). "
                "Permite estimar la varianza sin asumir una deriva nula "
                "en el proceso de precios.",
                className="page-description"
            ),

            html.P(
                "La varianza realizada de Rogers-Satchell constituye "
                "la variable objetivo de las arquitecturas de "
                "pron\u00f3stico desarrolladas en esta investigaci\u00f3n.",
                className="page-description"
            ),

            html.Div(
                [
                    html.Span("Rogers-Satchell (1991)"),
                    html.Span("Datos OHLC"),
                    html.Span("Varianza diaria"),
                    html.Span("Variable objetivo")
                ],
                className="research-tags"
            ),

            html.Hr(className="section-divider"),

            html.Div(
                "FUNCIÓN DE PÉRDIDA PRINCIPAL",
                className="research-label"
            ),

            html.H3(
                "QLIKE suavizada",
                className="evaluation-title"
            ),

            dcc_formula(),

            html.P(
                "Pérdida asimétrica de cuasiverosimilitud para el pronóstico de varianza, con estabilización numérica aditiva. Penaliza más la subestimación que la sobreestimación recíproca.",
                className="page-description"
            ),

            html.Div(
                [
                    html.Span("QLIKELoss v1.1.0"),
                    html.Span("Estabilización numérica"),
                    html.Span("Pronóstico de varianza")
                ],
                className="research-tags"
            ),

            html.Hr(className="section-divider"),

            html.Div(
                "EVALUACIÓN PROBABILÍSTICA",
                className="research-label"
            ),

            html.Div(
                [
                    html.Span("Pérdida Pinball"),
                    html.Span("CRPS")
                ],
                className="research-tags"
            ),

            html.Hr(className="section-divider"),

            html.Div(
                "MÉTRICAS COMPLEMENTARIAS",
                className="research-label"
            ),

            html.Div(
                [
                    html.Span(metric)
                    for metric in [
                        "MSE",
                        "RMSE",
                        "MAE",
                        "HMSE",
                        "HMAE"
                    ]
                ],
                className="research-tags"
            ),

            html.Hr(className="section-divider"),

            html.Div(
                "PROTOCOLO ESTADÍSTICO",
                className="research-label"
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Strong("0.05"),
                            html.Span("Nivel de significancia")
                        ]
                    ),

                    html.Div(
                        [
                            html.Strong("2,000"),
                            html.Span("Iteraciones de bootstrap")
                        ]
                    ),

                    html.Div(
                        [
                            html.Strong("0.33"),
                            html.Span("Umbral de delta de Cliff")
                        ]
                    )
                ],
                className="evaluation-stats"
            )
        ],
        className="evaluation-panel"
    )


def methodology_section():

    steps = [
        (
            "01",
            "Revisión sistemática de literatura",
            "Identificar arquitecturas de pronóstico relevantes y establecer un marco de comparación común.",
            [
                "PRISMA 2020",
                "10 mejores"
            ]
        ),
        (
            "02",
            "Análisis exploratorio de datos",
            "Caracterizar la dependencia temporal, el agrupamiento de la volatilidad y las propiedades distribucionales.",
            [
                "ACF / PACF",
                "Estacionariedad",
                "Ondículas (wavelets)"
            ]
        ),
        (
            "03",
            "Desarrollo de modelos",
            "Desarrollar modelos econométricos y de aprendizaje profundo para pronosticar la varianza realizada a un paso.",
            [
                "Referencias econométricas",
                "Aprendizaje profundo",
                "PyTorch"
            ]
        ),
        (
            "04",
            "Optimización de hiperparámetros",
            "Buscar configuraciones competitivas mediante validación cronológica y parada temprana.",
            [
                "Optuna",
                "Búsqueda de hiperparámetros",
                "Parada temprana"
            ]
        ),
        (
            "05",
            "Evaluación temporal",
            "Evaluar pronósticos a un paso con observaciones cronológicas fuera de muestra y validación progresiva.",
            [
                "Validación progresiva",
                "Inicialización previa",
                "QLIKE suavizada"
            ]
        ),
        (
            "06",
            "Validación estadística",
            "Evaluar la precisión predictiva, la adecuación de los residuos y las diferencias entre modelos.",
            [
                "Ljung-Box",
                "Diebold-Mariano",
                "MCS",
                "Friedman",
                "Nemenyi"
            ]
        ),
        (
            "07",
            "Evaluación extendida",
            "Evaluar la robustez, interpretabilidad y generalización de las arquitecturas de pronóstico.",
            [
                "Perturbaciones",
                "SHAP",
                "Gradientes integrados"
            ]
        )
    ]

    return html.Section(
        [
            panel_heading(
                "08 / DISEÑO EXPERIMENTAL",
                "De la evidencia al pronóstico",
                "Descripción general de la metodología de investigación propuesta."
            ),

            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                methodology_step(*step)
                                for step in steps
                            ],
                            className="methodology-pipeline"
                        ),
                        xs=12,
                        xl=7
                    ),

                    dbc.Col(
                        evaluation_panel(),
                        xs=12,
                        xl=5
                    )
                ],
                className="g-4 mt-2"
            )
        ],
        className="panel research-section"
    )


def scientific_context():

    return html.Div(
        [
            literature_section(),
            research_questions_section(),
            methodology_section()
        ],
        className="scientific-context"
    )