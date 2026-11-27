import dash

from dash import Dash, html, dcc, Input, Output

import dash_bootstrap_components as dbc

from config import APP_NAME, APP_SUBTITLE, APP_VERSION


app = Dash(
    __name__,
    use_pages=True,
    pages_folder="",
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title=f"{APP_NAME} | {APP_SUBTITLE}",
    update_title=None
)

server = app.server


from pages.context import layout as context_layout
from pages.eda import layout as eda_layout
from pages.models import layout as models_layout


def header():

    return html.Header(
        [
            dcc.Link(
                [
                    html.Img(
                        src=app.get_asset_url("logo.svg"),
                        className="brand-logo"
                    ),
                    html.Div(
                        [
                            html.Div(
                                APP_NAME,
                                className="brand-name"
                            ),
                            html.Div(
                                APP_SUBTITLE.upper(),
                                className="brand-subtitle"
                            )
                        ]
                    )
                ],
                href="/",
                className="brand-link"
            ),
            html.Nav(
                [
                    dcc.Link(
                        "01  CONTEXTO",
                        href="/",
                        id="nav-context",
                        className="nav-link"
                    ),
                    dcc.Link(
                        "02  EDA",
                        href="/eda",
                        id="nav-eda",
                        className="nav-link"
                    ),
                    dcc.Link(
                        "03  MODELOS",
                        href="/models",
                        id="nav-models",
                        className="nav-link"
                    )
                ],
                className="nav-menu"
            ),
            html.Div(
                [
                    html.Span(
                        className="status-dot"
                    ),
                    html.Span(
                        "INVESTIGACIÓN CUANTITATIVA"
                    )
                ],
                className="system-status"
            )
        ],
        className="terminal-header"
    )


def footer():

    return html.Footer(
        [
            html.Span(
                "BRENT QUANT / INVESTIGACIÓN DE VOLATILIDAD"
            ),
            html.Span(
                "UNIVERSIDAD DEL NORTE"
            ),
            html.Span(
                f"VERSIÓN {APP_VERSION}"
            )
        ],
        className="terminal-footer"
    )


dash.register_page(
    "context",
    path="/",
    name="Contexto",
    layout=context_layout
)

dash.register_page(
    "eda",
    path="/eda",
    name="EDA",
    layout=eda_layout
)

dash.register_page(
    "models",
    path="/models",
    name="Modelos",
    layout=models_layout
)


app.layout = html.Div(
    [
        dcc.Location(
            id="navigation-url",
            refresh=False
        ),
        header(),
        html.Main(
            dash.page_container,
            className="main-content"
        ),
        footer()
    ]
)


@app.callback(
    Output("nav-context", "className"),
    Output("nav-eda", "className"),
    Output("nav-models", "className"),
    Input("navigation-url", "pathname")
)
def update_navigation(pathname):

    paths = ["/", "/eda", "/models"]

    return tuple(
        "nav-link active"
        if pathname == path
        else "nav-link"
        for path in paths
    )


if __name__ == "__main__":

    app.run(
        debug=True,
        dev_tools_ui=False,
        port=8050
    )