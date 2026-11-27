from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

from config import COLORS

from components.cards import page_heading, panel_heading
from components.experimental import experimental_layout
from components.model_results import model_results_layout

from services.model_loader import MODELS, MODEL_CONFIG


NAVY = COLORS["navy"]


FAMILIES = {
    "statistical": [
        "garch",
        "egarch",
        "har",
        "wtd_har"
    ],
    "deep_learning": [
        "gru",
        "tcn",
        "bigru_tcn",
        "causal_transformer",
        "conv_transformer",
        "band_causal_tcn",
        "markov_tcn"
    ]
}


FAMILY_NAMES = {
    "statistical": "Modelos econométricos y estadísticos",
    "deep_learning": "Deep Learning"
}


TABS = [
    ("protocol", "Protocolo experimental"),
    ("individual", "Modelos individuales"),
    ("comparison", "Comparación global")
]


def layout():

    return html.Div(
        [
            page_heading(
                "INVESTIGACIÓN / MODELADO PREDICTIVO",
                "Laboratorio de pronóstico",
                "Diseño experimental, entrenamiento y evaluación "
                "predictiva de la volatilidad del petróleo Brent."
            ),

            dcc.Tabs(
                id="models-tab",
                value="protocol",
                children=[
                    dcc.Tab(
                        label=label,
                        value=value,
                        className="eda-tab",
                        selected_className="eda-tab-selected"
                    )
                    for value, label in TABS
                ],
                className="eda-tabs"
            ),

            html.Div(
                [
                    html.Div(
                        "SELECCIÓN DE ARQUITECTURA",
                        className="eyebrow"
                    ),

                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Familia de modelos"
                                    ),

                                    dcc.Dropdown(
                                        id="models-family",
                                        options=[
                                            {
                                                "label": label,
                                                "value": value
                                            }
                                            for value, label
                                            in FAMILY_NAMES.items()
                                        ],
                                        value="statistical",
                                        clearable=False,
                                        searchable=False
                                    )
                                ],
                                xs=12,
                                md=6
                            ),

                            dbc.Col(
                                [
                                    html.Label(
                                        "Modelo"
                                    ),

                                    dcc.Dropdown(
                                        id="models-selected",
                                        options=[
                                            {
                                                "label": MODELS[key],
                                                "value": key
                                            }
                                            for key
                                            in FAMILIES["statistical"]
                                        ],
                                        value="garch",
                                        clearable=False,
                                        searchable=False
                                    )
                                ],
                                xs=12,
                                md=6
                            )
                        ],
                        className="g-3"
                    )
                ],
                id="models-selector",
                className="panel eda-filters",
                style={
                    "display": "none"
                }
            ),

            dcc.Loading(
                html.Div(
                    id="models-content"
                ),
                type="circle",
                color=NAVY
            )
        ],
        className="models-page",
        style={
            "width": "100%",
            "minWidth": 0
        }
    )


def pending_section(index, title, subtitle):

    return html.Section(
        [
            panel_heading(
                index,
                title,
                subtitle
            ),

            html.Div(
                [
                    html.Div(
                        "RESULTADOS PENDIENTES DE INTEGRACIÓN",
                        className="research-label"
                    ),

                    html.P(
                        "Los resultados de esta sección se incorporarán "
                        "a partir de los archivos experimentales "
                        "de la arquitectura seleccionada.",
                        style={
                            "color": "#64748B",
                            "fontSize": "13px",
                            "marginBottom": 0
                        }
                    )
                ],
                className="research-detail",
                style={
                    "marginTop": "20px"
                }
            )
        ],
        className="panel research-section",
        style={
            "marginBottom": "24px"
        }
    )


def pending_individual_layout(model):

    family = next(
        key
        for key, members in FAMILIES.items()
        if model in members
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        FAMILY_NAMES[family].upper(),
                        className="eyebrow"
                    ),

                    html.H2(
                        MODELS[model],
                        style={
                            "fontSize": "30px",
                            "fontWeight": "600",
                            "color": NAVY,
                            "margin": "8px 0"
                        }
                    ),

                    html.P(
                        "Modelo predictivo para la estimación "
                        "de la varianza del petróleo Brent.",
                        style={
                            "fontSize": "14px",
                            "color": "#64748B",
                            "marginBottom": 0
                        }
                    )
                ],
                className="panel",
                style={
                    "marginBottom": "24px",
                    "padding": "24px",
                    "borderLeft": f"3px solid {NAVY}"
                }
            ),

            pending_section(
                "01 / ARQUITECTURA",
                "Descripción del modelo",
                "Fundamento matemático y estructura predictiva."
            ),

            pending_section(
                "02 / OPTIMIZACIÓN",
                "Proceso de entrenamiento",
                "Configuración experimental, hiperparámetros "
                "y procedimientos de optimización."
            ),

            pending_section(
                "03 / EVALUACIÓN EN ENTRENAMIENTO",
                "Evaluación del entrenamiento",
                "Rendimiento predictivo y diagnóstico estadístico "
                "sobre el conjunto de entrenamiento."
            ),

            pending_section(
                "04 / EVALUACIÓN EN PRUEBA",
                "Evaluación en prueba",
                "Capacidad predictiva fuera de muestra "
                "y generalización temporal."
            )
        ],
        style={
            "width": "100%",
            "minWidth": 0
        }
    )


def individual_layout(model):

    if model not in MODELS:

        return dbc.Alert(
            "Selecciona un modelo válido.",
            color="warning"
        )

    if model in MODEL_CONFIG:

        content = model_results_layout(model)

    else:

        content = pending_individual_layout(model)

    return html.Div(
        content,
        style={
            "paddingTop": "20px",
            "width": "100%",
            "minWidth": 0,
            "boxSizing": "border-box"
        }
    )


def comparison_layout():

    from components.comparison import comparison_content

    return comparison_content()


@callback(
    Output("models-selector", "style"),
    Input("models-tab", "value")
)
def toggle_model_selector(tab):

    if tab == "individual":

        return {
            "display": "block",
            "marginTop": "20px",
            "width": "100%",
            "boxSizing": "border-box"
        }

    return {
        "display": "none"
    }


@callback(
    Output("models-selected", "options"),
    Output("models-selected", "value"),
    Input("models-family", "value")
)
def update_model_options(family):

    members = FAMILIES.get(
        family,
        FAMILIES["statistical"]
    )

    options = [
        {
            "label": MODELS[key],
            "value": key
        }
        for key in members
    ]

    return options, members[0]


@callback(
    Output("models-content", "children"),
    Input("models-tab", "value"),
    Input("models-selected", "value")
)
def render_models(tab, model):

    if tab == "protocol":

        return experimental_layout()

    if tab == "individual":

        return individual_layout(model)

    if tab == "comparison":

        return comparison_layout()

    return dbc.Alert(
        "Sección no disponible.",
        color="warning"
    )
