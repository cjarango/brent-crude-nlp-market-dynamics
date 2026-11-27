from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from services.comparison_loader import load_benchmark
from components.fhs_results import fhs_content


NAVY = "#172B4D"

NAMES = [
    "GARCH", "EGARCH", "HAR", "WTD-HAR",
    "GRU", "TCN", "Markov TCN", "BiGRU-TCN",
    "Cau. Trans", "Conv. Trans", "Band TCN"
]

CLIFF = [
    [],
    [0.02],
    [None, None],
    [None, None, -0.03],
    [None, None, None, None],
    [-0.06, -0.08, -0.03, -0.0, -0.07],
    [None, None, None, None, -0.03, 0.04],
    [0.05, None, 0.08, 0.11, None, 0.11, 0.07],
    [None, -0.05, None, None, -0.04, 0.03, None, -0.08],
    [-0.04, None, None, 0.02, None, 0.02, -0.02, -0.08, -0.01],
    [None, None, None, None, -0.05, 0.02, None, -0.09, None, 0.0]
]


def section(label, title, description, children):

    return html.Div([
        html.Div(label, style={
            "fontSize": "11px",
            "fontWeight": "600",
            "letterSpacing": "1.5px",
            "color": "#64748B"
        }),
        html.H3(title, style={
            "color": NAVY,
            "fontSize": "23px",
            "marginTop": "8px"
        }),
        html.P(description, style={
            "color": "#64748B",
            "fontSize": "14px",
            "marginBottom": "24px"
        }),
        *children
    ], style={
        "background": "#FFFFFF",
        "border": "1px solid #E2E8F0",
        "borderRadius": "8px",
        "padding": "24px",
        "marginBottom": "24px",
        "minWidth": 0
    })


def benchmark_table():

    df = load_benchmark().copy()

    for col in df.columns[1:]:
        df[col] = df[col].map(lambda x: f"{x:.6f}")

    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[
            {"name": c, "id": c}
            for c in df.columns
        ],
        sort_action="native",
        style_table={
            "overflowX": "auto"
        },
        style_header={
            "backgroundColor": NAVY,
            "color": "white",
            "fontWeight": "600",
            "textAlign": "center",
            "padding": "12px"
        },
        style_cell={
            "fontSize": "13px",
            "padding": "12px",
            "textAlign": "center",
            "border": "1px solid #E2E8F0",
            "fontFamily": "Inter, sans-serif"
        },
        style_cell_conditional=[
            {
                "if": {"column_id": "Modelo"},
                "textAlign": "left",
                "fontWeight": "600"
            }
        ],
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#F8FAFC"
            },
            {
                "if": {"row_index": 0},
                "backgroundColor": "#E8F0F8",
                "fontWeight": "600"
            }
        ]
    )


def cliff_figure():

    fig = go.Figure()

    groups = {
        "positive": {"x": [], "y": [], "text": []},
        "negative": {"x": [], "y": [], "text": []},
        "ns": {"x": [], "y": [], "text": []}
    }

    for i, row in enumerate(CLIFF):

        for j in range(i + 1):

            fig.add_shape(
                type="rect",
                x0=j - 0.5,
                x1=j + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                fillcolor="#F1F1F1" if i == j else "#FFFFFF",
                line=dict(color="#D6D6D6", width=1),
                layer="below"
            )

        fig.add_annotation(
            x=i, y=i,
            text="\u2014",
            showarrow=False,
            font=dict(color="#888888", size=12)
        )

        for j, value in enumerate(row):

            if value is None:
                key = "ns"
                label = ""
            elif value < 0 or str(value).startswith("-"):
                key = "negative"
                label = f"{value:.2f}"
            else:
                key = "positive"
                label = f"{value:.2f}"

            groups[key]["x"].append(j)
            groups[key]["y"].append(i)
            groups[key]["text"].append(label)

    styles = [
        ("positive", "#F6D2C7", 32,
         "\u03b4 > 0 (Modelo A mayor p\u00e9rdida)"),
        ("negative", "#C5E0EB", 32,
         "\u03b4 < 0 (Modelo A menor p\u00e9rdida)"),
        ("ns", "#222222", 8,
         "DM no significativo (IC 95% incluye 0)")
    ]

    for key, color, size, name in styles:

        group = groups[key]

        fig.add_trace(go.Scatter(
            x=group["x"],
            y=group["y"],
            text=group["text"],
            mode="markers+text",
            textposition="middle center",
            textfont=dict(
                size=11,
                color="#111111",
                family="Arial Black"
            ),
            marker=dict(
                size=size,
                color=color,
                line=dict(color="#A0A0A0", width=0.7)
            ),
            name=name,
            hoverinfo="skip"
        ))

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(11)),
        ticktext=NAMES,
        tickangle=-40,
        range=[-0.5, 10.5],
        title="Modelo B",
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="#555555"
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(11)),
        ticktext=NAMES,
        range=[10.5, -0.5],
        title="Modelo A",
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="#555555"
    )

    fig.update_layout(
        title=dict(
            text="Delta de Cliff por pares (Referencia DM - QLIKE)",
            x=0.5,
            font=dict(size=15, color=NAVY)
        ),
        height=650,
        autosize=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial", size=12),
        margin=dict(l=100, r=25, t=55, b=100),
        legend=dict(
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11)
        )
    )

    return fig


def mcs_panel():

    indicators = [
        ("MUESTRAS BOOTSTRAP", "2.000"),
        ("MODELOS EVALUADOS", "11"),
        ("MODELOS RETENIDOS", "11 / 11")
    ]

    return html.Div([
        html.H5(
            "Model Confidence Set (MCS)",
            style={"color": NAVY}
        ),

        dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div(label, style={
                        "fontSize": "11px",
                        "color": "#64748B",
                        "fontWeight": "600"
                    }),
                    html.Div(value, style={
                        "fontSize": "28px",
                        "color": NAVY,
                        "fontWeight": "600"
                    })
                ], style={
                    "padding": "18px",
                    "border": "1px solid #E2E8F0",
                    "borderRadius": "6px"
                }),
                md=4
            )
            for label, value in indicators
        ], className="g-3"),

        html.P(
            "El procedimiento Model Confidence Set, aplicado "
            "sobre la p\u00e9rdida QLIKE mediante 2.000 muestras "
            "bootstrap, mantuvo los once modelos dentro del "
            "conjunto superior. No se obtuvo evidencia "
            "estad\u00edstica suficiente para excluir ninguno "
            "de los modelos evaluados.",
            style={
                "marginTop": "18px",
                "marginBottom": 0,
                "fontSize": "14px",
                "color": "#475569"
            }
        )
    ], style={
        "borderTop": "1px solid #E2E8F0",
        "paddingTop": "24px",
        "marginTop": "24px"
    })


def comparison_content():

    try:
        benchmark = benchmark_table()
    except Exception as error:
        benchmark = dbc.Alert(str(error), color="danger")

    return html.Div([

        section(
            "01 / BENCHMARK",
            "Comparaci\u00f3n del rendimiento predictivo",
            "Evaluaci\u00f3n conjunta de los once modelos sobre "
            "el conjunto de prueba, ordenados por QLIKE.",
            [benchmark]
        ),

        section(
            "02 / INFERENCIA ESTAD\u00cdSTICA",
            "Comparaci\u00f3n estad\u00edstica",
            "Comparaci\u00f3n por pares mediante la prueba de "
            "Diebold-Mariano y el delta de Cliff como medida "
            "de magnitud del efecto.",
            [
                html.Div(
                    dcc.Graph(
                        figure=cliff_figure(),
                        config={
                            "displayModeBar": False,
                            "responsive": True
                        }
                    ),
                    style={
                        "width": "100%",
                        "maxWidth": "800px",
                        "margin": "0 auto"
                    }
                ),
                mcs_panel()
            ]
        ),

        section(
            "03 / PRON\u00d3STICO PROBABIL\u00cdSTICO",
            "Bandas de predicci\u00f3n del precio del Brent",
            "Intervalos predictivos del 95% mediante "
            "Filtered Historical Simulation (FHS), "
            "construidos a partir de los retornos "
            "estandarizados y las estimaciones de "
            "volatilidad condicional del modelo TCN.",
            [fhs_content()]
        )

    ], style={
        "paddingTop": "20px",
        "width": "100%",
        "minWidth": 0
    })
