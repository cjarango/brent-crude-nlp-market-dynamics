from dash import html, dcc
import plotly.graph_objects as go

from services.fhs_loader import load_fhs_bands


def fhs_figure():

    df = load_fhs_bands()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["p_low"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hovertemplate=(
            "Limite inferior: %{y:.2f} USD"
            "<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["p_high"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0,114,178,0.25)",
        name="Banda FHS (95%)",
        hovertemplate=(
            "Limite superior: %{y:.2f} USD"
            "<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["p_true"],
        mode="lines",
        name="Precio real",
        line=dict(
            color="#808080",
            width=1.2
        ),
        hovertemplate=(
            "Precio real: %{y:.2f} USD"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=dict(
            text="Bandas de Predicci\u00f3n en Precio (TCN)",
            x=0.5,
            font=dict(size=16)
        ),
        height=550,
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter, Arial",
            size=12,
            color="#222222"
        ),
        margin=dict(
            l=75,
            r=30,
            t=65,
            b=65
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12)
        ),
        hovermode="x unified"
    )

    fig.update_xaxes(
        title="Fecha",
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="#333333",
        linewidth=1,
        ticks="outside",
        dtick="M6",
        tickformat="%b %Y"
    )

    fig.update_yaxes(
        title="Precio (USD)",
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="#333333",
        linewidth=1,
        ticks="outside"
    )

    return fig


def fhs_content():

    return html.Div([

        html.Div(
            dcc.Graph(
                figure=fhs_figure(),
                config={
                    "displayModeBar": False,
                    "responsive": True
                }
            ),
            style={
                "width": "100%",
                "maxWidth": "1200px",
                "margin": "0 auto"
            }
        ),

        html.P(
            "Las bandas se construyen a partir de los "
            "cuantiles emp\u00edricos de los retornos "
            "estandarizados y las predicciones de "
            "varianza condicional del modelo TCN. "
            "Los intervalos de los retornos se transforman "
            "a precios mediante el precio observado "
            "en la sesi\u00f3n anterior.",
            style={
                "fontSize": "13px",
                "lineHeight": "1.7",
                "color": "#64748B",
                "marginTop": "16px"
            }
        ),

        html.P(
            "Nota: los cuantiles FHS se calibraron sobre "
            "el conjunto de prueba, siguiendo el "
            "procedimiento original. La calibraci\u00f3n "
            "es retrospectiva.",
            style={
                "fontSize": "12px",
                "color": "#64748B",
                "marginBottom": 0
            }
        )

    ])
