import pandas as pd
import plotly.graph_objects as go

from config import COLORS, CHART_TEMPLATE, FONT_FAMILY


def temporal_selector(candlestick=False):

    periods = [
        (1, "1M", "month"),
        (3, "3M", "month"),
        (6, "6M", "month"),
        (1, "1A", "year")
    ]

    if not candlestick:
        periods.append((5, "5A", "year"))

    buttons = [
        {
            "count": count,
            "label": label,
            "step": step,
            "stepmode": "backward"
        }
        for count, label, step in periods
    ]

    if not candlestick:
        buttons.append({
            "label": "TODO",
            "step": "all"
        })

    return {
        "x": 0,
        "y": 1.10,
        "bgcolor": COLORS["background"],
        "activecolor": COLORS["gold"],
        "borderwidth": 0,
        "font": {
            "family": FONT_FAMILY,
            "size": 11,
            "color": COLORS["navy"]
        },
        "buttons": buttons
    }

def axis_style():

    return {
        "showline": True,
        "mirror": True,
        "linecolor": "rgba(23, 43, 77, 0.25)",
        "linewidth": 1,
        "ticks": "outside",
        "ticklen": 4,
        "tickwidth": 1,
        "tickcolor": "rgba(23, 43, 77, 0.45)",
        "tickfont": {
            "family": FONT_FAMILY,
            "size": 11,
            "color": COLORS["slate"]
        },
        "zeroline": False,
        "automargin": True
    }


def financial_layout(fig, df, initial_months=None):

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        autosize=True,
        showlegend=False,
        dragmode="zoom",
        hovermode="x unified",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        margin={
            "l": 15,
            "r": 65,
            "t": 55,
            "b": 50
        },
        xaxis={
            **axis_style(),
            "type": "date",
            "title": {
                "text": "Fecha",
                "font": {
                    "size": 11,
                    "color": COLORS["slate"]
                },
                "standoff": 12
            },
            "showgrid": False,
            "nticks": 9,
            "rangeselector": temporal_selector(
   	     candlestick=initial_months is not None
             ),
            "rangeslider": {
                "visible": False
            },
            "tickformatstops": [
                {
                    "dtickrange": [None, 86400000],
                    "value": "%d/%m"
                },
                {
                    "dtickrange": [86400000, "M1"],
                    "value": "%d/%m"
                },
                {
                    "dtickrange": ["M1", "M12"],
                    "value": "%m/%Y"
                },
                {
                    "dtickrange": ["M12", None],
                    "value": "%Y"
                }
            ]
        },
        yaxis={
            **axis_style(),
            "side": "right",
            "title": {
                "text": "Precio (USD/barril)",
                "font": {
                    "size": 11,
                    "color": COLORS["slate"]
                },
                "standoff": 12
            },
            "tickprefix": "$",
            "showgrid": True,
            "gridcolor": "rgba(100, 116, 139, 0.08)",
            "gridwidth": 1,
            "fixedrange": False,
            "nticks": 7
        }
    )

    if initial_months is not None:

        end_date = df.index.max()

        start_date = (
            end_date - pd.DateOffset(months=initial_months)
        )

        fig.update_xaxes(
            range=[start_date, end_date]
        )

        visible_df = df.loc[
            (df.index >= start_date) &
            (df.index <= end_date)
        ]

        if not visible_df.empty:

            lower = visible_df["low"].min()
            upper = visible_df["high"].max()

            padding = (upper - lower) * 0.08

            fig.update_yaxes(
                range=[
                    lower - padding,
                    upper + padding
                ]
            )

    return fig


def market_price_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            name="Petróleo Brent",
            mode="lines",
            line={
                "color": COLORS["navy"],
                "width": 1.6
            },
            hovertemplate=(
                "$%{y:.2f} USD"
                "<extra></extra>"
            )
        )
    )

    return financial_layout(fig, df)


def candlestick_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Petróleo Brent",
            increasing_line_color=COLORS["navy"],
            decreasing_line_color=COLORS["gold"],
            increasing_fillcolor=COLORS["navy"],
            decreasing_fillcolor=COLORS["gold"],
            increasing_line_width=1,
            decreasing_line_width=1
        )
    )

    return financial_layout(
        fig,
        df,
        initial_months=3
    )


def partition_chart(summary):

    fig = go.Figure()

    partitions = [
        (
            "Entrenamiento",
            summary["train_observations"],
            COLORS["navy"]
        ),
        (
            "Prueba",
            summary["test_observations"],
            COLORS["gold"]
        )
    ]

    for name, count, color in partitions:

        percentage = (
            count / summary["processed_observations"] * 100
        )

        fig.add_trace(
            go.Bar(
                x=[count],
                y=["Conjunto de datos"],
                name=name,
                orientation="h",
                marker_color=color,
                text=[f"{percentage:.1f}%"],
                textposition="inside",
                insidetextanchor="middle",
                textfont={
                    "family": FONT_FAMILY,
                    "color": COLORS["surface"],
                    "size": 12
                },
                hovertemplate=(
                    f"{name}: "
                    "%{x:,} observations"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        template=CHART_TEMPLATE,
        barmode="stack",
        height=85,
        showlegend=False,
        margin={
            "l": 0,
            "r": 0,
            "t": 5,
            "b": 5
        },
        xaxis={
            "visible": False,
            "showline": False,
            "ticks": ""
        },
        yaxis={
            "visible": False,
            "showline": False,
            "ticks": ""
        }
    )

    return fig