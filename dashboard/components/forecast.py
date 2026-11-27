import json
from pathlib import Path

from dash import html, callback, Input, Output
import dash_bootstrap_components as dbc


FORECAST_PATH = (
    Path(__file__).resolve().parents[1]
    / "services"
    / "tcn_forecast.json"
)


def forecast_card():

    return html.Div([

        html.Div(
            "VOLATILIDAD SIGUIENTE SESI\u00d3N",
            className="metric-label"
        ),

        html.Div(
            id="context-forecast-value",
            className="metric-value",
            children=""
        ),

        html.Div(
            id="context-forecast-direction",
            className="metric-caption"
        ),

        dbc.Button(
            "Generar pron\u00f3stico",
            id="context-forecast-button",
            n_clicks=0,
            color="secondary",
            outline=True,
            size="sm",
            style={
                "width": "100%",
                "fontSize": "11px",
                "marginTop": "8px",
                "borderColor": "#CBD5E1",
                "color": "#172B4D",
                "backgroundColor": "#FFFFFF"
            }
        )

    ], className="metric-card")


@callback(
    Output("context-forecast-value", "children"),
    Output("context-forecast-direction", "children"),
    Output("context-forecast-button", "children"),
    Input("context-forecast-button", "n_clicks"),
    prevent_initial_call=True
)
def update_forecast(n_clicks):

    try:

        with FORECAST_PATH.open(
            encoding="utf-8"
        ) as file:

            result = json.load(file)

        pred_vol = result["pred_vol"]
        last_vol = result["last_vol"]

        pct_change = (
            (pred_vol / last_vol - 1) * 100
        )

        if pred_vol > last_vol:

            direction = "SUBE"
            symbol = "\u2191"
            color = "#B91C1C"

        elif pred_vol < last_vol:

            direction = "BAJA"
            symbol = "\u2193"
            color = "#15803D"

        else:

            direction = "ESTABLE"
            symbol = "\u2192"
            color = "#64748B"

        detail = html.Span([

            html.Strong(
                f"{symbol} {direction} "
                f"({pct_change:+.2f}%)",
                style={"color": color}
            ),

            html.Span(
                " | TCN",
                style={"color": "#64748B"}
            )

        ])

        return (
            f"{pred_vol * 100:.4f}%",
            detail,
            "Consultar pron\u00f3stico"
        )

    except Exception:

        return (
            "",
            "Pron\u00f3stico no disponible",
            "Reintentar"
        )
