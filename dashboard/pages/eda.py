from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from inspect import signature
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, lilliefors
from statsmodels.stats.stattools import jarque_bera
import pywt

from config import COLORS, CHART_TEMPLATE
from components.cards import page_heading, panel_heading, metric_card
from services.data_loader import load_processed, filter_data


TABS = [
    ('mercado', 'Dinámica del mercado'),
    ('vol', 'Volatilidad'),
    ('stats', 'Propiedades estadísticas'),
    ('wave', 'Análisis wavelet')
]

VARS = {
    'close': 'Precio de cierre',
    'log_returns': 'Retornos logarítmicos',
    'rs_daily': 'Varianza Rogers–Satchell',
    'squared_returns': 'Retornos cuadrados',
    'volume': 'Volumen'
}

PARTITIONS = {
    'All': 'Conjunto completo',
    'Train': 'Entrenamiento',
    'Test': 'Prueba'
}

ALLOWED_PARTITIONS = {
    'mercado': ['All', 'Train', 'Test'],
    'vol': ['All', 'Train'],
    'stats': ['Train'],
    'wave': ['All', 'Train', 'Test']
}

NAVY = COLORS['navy']
BLACK = '#111111'
BORDER = '#D0D5DD'


def layout():

    try:
        df = load_processed()

    except (ValueError, FileNotFoundError) as exc:
        return dbc.Alert(str(exc), color='warning')

    lo = df.index.min().date()
    hi = df.index.max().date()

    return html.Div([

        page_heading(
            'INVESTIGACIÓN / ANÁLISIS EXPLORATORIO',
            'Análisis exploratorio del mercado',
            'Propiedades estadísticas, dependencias temporales '
            'y dinámica de la volatilidad.'
        ),

        html.Div([

            html.Div(
                'FILTROS GLOBALES',
                className='eyebrow'
            ),

            dbc.Row([

                dbc.Col([

                    html.Label('Partición'),

                    dcc.Dropdown(
                        id='eda-part',
                        options=[
                            {'label': label, 'value': value}
                            for value, label in PARTITIONS.items()
                        ],
                        value='Train',
                        clearable=False
                    )

                ], xs=12, md=3),

                dbc.Col([

                    html.Label('Periodo de análisis'),

                    dcc.DatePickerRange(
                        id='eda-dates',
                        min_date_allowed=lo,
                        max_date_allowed=hi,
                        start_date=lo,
                        end_date=hi,
                        display_format='DD/MM/YYYY'
                    )

                ], xs=12, md=4),

                dbc.Col([

                    html.Label('Variable de análisis'),

                    dcc.Dropdown(
                        id='eda-var',
                        options=[
                            {'label': label, 'value': value}
                            for value, label in VARS.items()
                        ],
                        value='rs_daily',
                        clearable=False
                    )

                ], xs=12, md=3),

                dbc.Col([

                    html.Label(
                        'Rezagos',
                        id='eda-lags-label'
                    ),

                   dcc.Slider(
    			id='eda-lags',
   			min=10,
   			max=60,
    			step=10,
    			value=50,
    			disabled=True,
    			marks={i: str(i) for i in range(10, 61, 10)}
		   )

                ], xs=12, md=2)

            ], className='g-3'),

            html.Div(
                id='eda-filter-note',
                children=(
                    'Por defecto se explora el entrenamiento, '
                    'tal como en el cuadernillo original.'
                ),
                className='eda-note'
            )

        ], className='panel eda-filters'),

        dcc.Tabs(
            id='eda-tab',
            value='mercado',
            children=[
                dcc.Tab(
                    label=label,
                    value=value,
                    className='eda-tab',
                    selected_className='eda-tab-selected'
                )
                for value, label in TABS
            ],
            className='eda-tabs'
        ),

        dcc.Loading(
            html.Div(id='eda-content'),
            type='circle',
            color=NAVY
        )

    ], className='eda-page')


def figure(traces, x='Fecha', y='', height=320):

    f = go.Figure(traces)

    f.update_layout(
        template=CHART_TEMPLATE,
        height=height,
        autosize=True,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        margin=dict(l=65, r=25, t=22, b=55),
        hovermode='closest',
        legend=dict(orientation='h', y=1.12),
        showlegend=len(traces) > 1
    )

    axis_style = dict(
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='rgba(23,43,77,.25)',
        ticks='outside',
        ticklen=4,
        tickwidth=1,
        tickcolor=NAVY,
        showgrid=True,
        gridcolor='rgba(100,116,139,.10)',
        automargin=True,
        zeroline=False
    )

    f.update_xaxes(
        title_text=x,
        **axis_style
    )

    f.update_yaxes(
        title_text=y,
        side='left',
        **axis_style
    )

    return f


def line(s, name, color=NAVY):

    return go.Scatter(
        x=s.index,
        y=s,
        mode='lines',
        name=name,
        line=dict(color=color, width=1.4),
        hovertemplate=(
            '%{y:.6g}'
            '<extra>%{fullData.name}</extra>'
        )
    )


def graph(title, f, note):

    height = f.layout.height or 320

    return html.Div([

        html.H3(
            title,
            className='eda-title'
        ),

        dcc.Graph(
            figure=f,
            config={
                'displaylogo': False,
                'responsive': True
            },
            responsive=True,
            style={
                'width': '100%',
                'height': f'{height}px',
                'minHeight': f'{height}px',
                'flexShrink': 0
            }
        ),

        html.Div(
            note,
            className='eda-note'
        )

    ],

        className='panel eda-card',

        style={
            'height': 'auto',
            'minWidth': 0,
            'display': 'flex',
            'flexDirection': 'column'
        }

    )


def grid(*items):

    return dbc.Row(
        [
            dbc.Col(
                item,
                xs=12,
                xl=12 if len(items) == 1 else 6
            )
            for item in items
        ],
        className='g-3 mb-3'
    )


def table(title, rows, note):

    if not rows:
        return dbc.Alert(
            'No hay resultados disponibles.',
            color='warning'
        )

    return html.Div([

        html.H3(
            title,
            className='eda-title'
        ),

        html.Div([

            dash_table.DataTable(

                data=rows,

                columns=[
                    {'name': key, 'id': key}
                    for key in rows[0]
                ],

                sort_action='native',

                style_table={
                    'overflowX': 'auto',
                    'width': '100%'
                },

                style_cell={
                    'fontFamily': 'Inter, sans-serif',
                    'fontSize': 12,
                    'padding': '12px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'textAlign': 'left',
                    'color': NAVY,
                    'border': f'1px solid {BORDER}'
                },

                style_header={
                    'backgroundColor': COLORS['background'],
                    'fontWeight': '600',
                    'border': f'1px solid {BORDER}'
                },

                style_data={
                    'backgroundColor': '#FFFFFF',
                    'border': f'1px solid {BORDER}'
                },

                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#F7F8FA'
                    }
                ],

                css=[
                    {
                        'selector': (
                            '.dash-spreadsheet-container '
                            '.dash-spreadsheet-inner '
                            'td:first-child'
                        ),
                        'rule': (
                            f'border-left: 1px solid '
                            f'{BORDER} !important;'
                        )
                    },
                    {
                        'selector': (
                            '.dash-spreadsheet-container '
                            '.dash-spreadsheet-inner '
                            'th:first-child'
                        ),
                        'rule': (
                            f'border-left: 1px solid '
                            f'{BORDER} !important;'
                        )
                    }
                ]

            )

        ], style={
            'width': '100%',
            'border': f'1px solid {BORDER}',
            'boxSizing': 'border-box',
            'overflow': 'hidden'
        }),

        html.Div(
            note,
            className='eda-note'
        )

    ],

        className='panel eda-card',

        style={
            'height': 'auto',
            'minWidth': 0,
            'display': 'flex',
            'flexDirection': 'column'
        }

    )


def hist(s, label, color=NAVY):

    return figure(
        [
            go.Histogram(
                x=s,
                nbinsx=60,
                marker=dict(
                    color=color,
                    line=dict(color=color, width=0.3)
                )
            )
        ],
        x=label,
        y='Frecuencia'
    )


def fmt(v):

    if not np.isfinite(v):
        return 'No disponible'

    if abs(v) >= 1:
        return (
            f'{v:,.3f}'
            .rstrip('0')
            .rstrip('.')
        )

    return f'{v:.6g}'


def fmt_p(p):

    if not np.isfinite(p):
        return 'No disponible'

    if p < 0.0001:
        return '< 0.0001'

    return f'{p:.4f}'


def legacy_test(fn, s, **kwargs):

    if 'result_object' in signature(fn).parameters:
        kwargs['result_object'] = False

    return fn(s, **kwargs)[:2]


def diagnostics(s, lags):

    rows = []

    def add(name, fn, null):

        try:

            stat, p = fn()

            if not np.isfinite(p):
                decision = 'No disponible'

            else:
                decision = (
                    'Rechazar H₀'
                    if p < 0.05
                    else 'No rechazar H₀'
                )

            rows.append({
                'Prueba': name,
                'Estadístico': fmt(stat),
                'Valor p': fmt_p(p),
                'Hipótesis nula': null,
                'Decisión (5 %)': decision
            })

        except (
            ValueError,
            np.linalg.LinAlgError,
            ZeroDivisionError
        ):

            rows.append({
                'Prueba': name,
                'Estadístico': '—',
                'Valor p': '—',
                'Hipótesis nula': null,
                'Decisión (5 %)': 'No disponible'
            })

    add(
        'ADF',
        lambda: legacy_test(
            adfuller, s, autolag='AIC'
        ),
        'Raíz unitaria'
    )

    add(
        'KPSS',
        lambda: legacy_test(
            kpss, s, regression='c', nlags='auto'
        ),
        'Estacionariedad en nivel'
    )

    add(
        'Jarque–Bera',
        lambda: jarque_bera(s)[:2],
        'Normalidad'
    )

    add(
        'Lilliefors',
        lambda: lilliefors(s),
        'Normalidad'
    )

    ls = [
        k for k in [10, 20, 30]
        if k <= lags and k < len(s)
    ]

    if ls:

        lb = acorr_ljungbox(
            s,
            lags=ls,
            return_df=True
        )

        for k in ls:

            add(
                f'Ljung–Box ({k})',

                lambda k=k: (
                    lb.loc[k, 'lb_stat'],
                    min(
                        1,
                        lb.loc[k, 'lb_pvalue'] * len(ls)
                    )
                ),

                'Ausencia de autocorrelación'
            )

    return rows


def market_section(df):

    descriptive_rows = [
        {
            'Variable': VARS[k],
            'Media': fmt(df[k].mean()),
            'Desv. estándar': fmt(df[k].std()),
            'Mediana': fmt(df[k].median())
        }
        for k in VARS
    ]

    return [

        grid(

            graph(
                'Precio de cierre',

                figure(
                    [line(df.close, 'Cierre', NAVY)],
                    y='USD/barril'
                ),

                'Los cambios de nivel motivan estudiar '
                'los retornos en lugar de los precios '
                'para caracterizar la dinámica del mercado.'
            ),

            graph(
                'Retornos logarítmicos',

                figure(
                    [line(df.log_returns, 'Retorno', BLACK)],
                    y='Retorno'
                ),

                'Los episodios de amplitud elevada permiten '
                'explorar el agrupamiento de volatilidad.'
            )

        ),

        grid(

            graph(
                'Distribución de retornos',

                hist(
                    df.log_returns,
                    'Retorno',
                    NAVY
                ),

                'Las colas de la distribución permiten '
                'examinar la presencia de eventos extremos.'
            ),

            table(
                'Estadísticos descriptivos',

                descriptive_rows,

                'Las diferencias de escala son relevantes '
                'para la preparación de las entradas de los modelos.'
            )

        )

    ]


def volatility_section(df):

    r = df.rs_daily
    q = df.squared_returns

    return [

        grid(

            graph(
                'Varianza Rogers–Satchell',

                figure(
                    [line(r, 'Rogers–Satchell', NAVY)],
                    y='Varianza'
                ),

                'Estimador basado en precios OHLC '
                'y variable objetivo del pronóstico.'
            ),

            graph(
                'Retornos cuadrados',

                figure(
                    [line(q, 'Retornos cuadrados', BLACK)],
                    y='Retorno²'
                ),

                'Proxy alternativo de varianza basado '
                'en los retornos de cierre.'
            )

        ),

        grid(

            graph(
                'Distribución de Rogers–Satchell',

                hist(r, 'Varianza', NAVY),

                'La asimetría y los valores extremos '
                'motivan evaluar funciones de pérdida '
                'específicas para el pronóstico de varianza.'
            ),

            graph(
                'Comparación de estimadores',

                figure(
                    [
                        go.Scatter(
                            x=q,
                            y=r,
                            mode='markers',
                            marker=dict(
                                color=NAVY,
                                size=4,
                                opacity=0.4
                            ),
                            hovertemplate=(
                                'Retorno²: %{x:.6g}'
                                '<br>Varianza RS: %{y:.6g}'
                                '<extra></extra>'
                            )
                        )
                    ],
                    x='Retorno²',
                    y='Varianza RS'
                ),

                'La asociación entre los estimadores '
                'no implica equivalencia entre sus mediciones.'
            )

        )

    ]


def correlation_plot(values, confidence, name):

    values = np.asarray(values)[1:]
    confidence = np.asarray(confidence)[1:]

    lags = np.arange(1, len(values) + 1)

    lower = confidence[:, 0] - values
    upper = confidence[:, 1] - values

    stem_x = []
    stem_y = []

    for lag, value in zip(lags, values):

        stem_x.extend([lag, lag, None])
        stem_y.extend([0, value, None])

    traces = [

        go.Scatter(
            x=lags,
            y=lower,
            mode='lines',
            line=dict(width=0),
            hoverinfo='skip',
            showlegend=False
        ),

        go.Scatter(
            x=lags,
            y=upper,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(100,100,100,0.12)',
            hoverinfo='skip',
            showlegend=False
        ),

        go.Scatter(
            x=stem_x,
            y=stem_y,
            mode='lines',
            line=dict(
                color='rgba(0,0,0,0.40)',
                width=1
            ),
            hoverinfo='skip',
            showlegend=False
        ),

        go.Scatter(
            x=lags,
            y=values,
            mode='markers',
            name=name,
            marker=dict(
                color=BLACK,
                size=5
            ),
            hovertemplate=(
                'Rezago: %{x}'
                '<br>Correlación: %{y:.4f}'
                '<extra></extra>'
            ),
            showlegend=False
        )

    ]

    f = figure(
        traces,
        x='Rezago (sesiones)',
        y='Autocorrelación',
        height=340
    )

    f.update_layout(
        showlegend=False,
        hovermode='closest'
    )

    f.add_hline(
        y=0,
        line_color='rgba(0,0,0,0.65)',
        line_width=1,
        layer='below'
    )

    f.update_xaxes(
        range=[-1, len(values) + 2],
        tick0=0,
        dtick=10 if len(values) >= 30 else 5,
        showgrid=True,
        gridcolor='rgba(100,116,139,.10)',
        gridwidth=1,
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='#A0A7B2',
        ticks='outside'
    )

    f.update_yaxes(
        range=[-1.1, 1.1],
        tick0=-1,
        dtick=0.25,
        tickformat='.2f',
        showgrid=True,
        gridcolor='rgba(100,116,139,.10)',
        gridwidth=1,
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor='#A0A7B2',
        ticks='outside'
    )

    return f


def statistical_section(s, lags):

    if s.nunique() < 2:

        return [
            dbc.Alert(
                'La serie seleccionada es constante. '
                'No se pueden calcular los diagnósticos.',
                color='warning'
            )
        ]

    k = min(
        int(lags),
        len(s) // 2 - 1
    )

    acf_values, acf_confidence = acf(
        s,
        nlags=k,
        fft=True,
        alpha=0.05
    )

    pacf_values, pacf_confidence = pacf(
        s,
        nlags=k,
        method='ywm',
        alpha=0.05
    )

    return [

        grid(

            graph(
                'Función de autocorrelación',

                correlation_plot(
                    acf_values,
                    acf_confidence,
                    'ACF'
                ),

                'Permite identificar la persistencia temporal '
                'de la serie. La región sombreada representa '
                'los límites de confianza aproximados del 95 %.'
            ),

            graph(
                'Función de autocorrelación parcial',

                correlation_plot(
                    pacf_values,
                    pacf_confidence,
                    'PACF'
                ),

                'Permite identificar dependencias directas '
                'por rezago, controlando los rezagos intermedios.'
            )

        ),

        table(
            'Diagnósticos estadísticos',

            diagnostics(s, k),

            'Ljung–Box con ajuste Bonferroni. '
            'Los valores p de KPSS y Lilliefors pueden estar '
            'acotados por sus procedimientos de aproximación. '
            'No rechazar una hipótesis nula no implica '
            'demostrar que sea verdadera.'
        )

    ]


def wavelet_section(s):

    w = pywt.Wavelet('db4')

    level = 7

    max_level = pywt.dwt_max_level(
        len(s),
        w.dec_len
    )

    if max_level < level:

        return [
            dbc.Alert(
                'El periodo seleccionado no contiene '
                'suficientes observaciones para realizar '
                'la descomposición wavelet de nivel 7. '
                'Amplía el periodo de análisis.',
                color='warning'
            )
        ]

    coeff = pywt.wavedec(
        s.to_numpy(),
        w,
        level=level
    )

    def reconstruct(indices):

        reconstructed = pywt.waverec(
            [
                c if i in indices else np.zeros_like(c)
                for i, c in enumerate(coeff)
            ],
            w
        )[:len(s)]

        return pd.Series(
            reconstructed,
            index=s.index
        )

    groups = [

        (
            'Tendencia general · A7',
            [0],
            'Componente de aproximación que representa '
            'la dinámica de largo plazo de la serie.'
        ),

        (
            'Estacionalidad anualizada · D7',
            [1],
            'Componente 1: fluctuaciones asociadas '
            'a una escala aproximadamente anual.'
        ),

        (
            'Estacionalidad semestral · D6',
            [2],
            'Componente 2: fluctuaciones asociadas '
            'a una escala aproximadamente semestral.'
        ),

        (
            'Estacionalidad trimestral · D5 + D4',
            [3, 4],
            'Reconstrucción conjunta de los componentes '
            '3 y 4, asociados a la escala trimestral.'
        ),

        (
            'Variaciones irregulares · D3 + D2 + D1',
            [5, 6, 7],
            'Reconstrucción conjunta de los componentes '
            '5, 6 y 7, asociados a fluctuaciones '
            'irregulares de corto plazo.'
        )

    ]

    cards = []

    for title, indices, note in groups:

        component = reconstruct(indices)

        cards.append(

            graph(
                title,

                figure(
                    [line(component, title, BLACK)],
                    y='Amplitud'
                ),

                note
            )

        )

    return [

        html.Div(
            'Daubechies db4 · Nivel 7 · '
            'Reconstrucción agrupada de componentes. '
            'Las escalas estacionales son aproximadas. '
            'Descomposición exploratoria no causal.',
            className='eda-note mb-3'
        ),

        grid(cards[0]),

        grid(cards[1], cards[2]),

        grid(cards[3], cards[4])

    ]


@callback(
    Output('eda-part', 'options'),
    Output('eda-part', 'value'),
    Output('eda-part', 'disabled'),
    Output('eda-lags', 'disabled'),
    Output('eda-lags-label', 'style'),
    Output('eda-filter-note', 'children'),

    Input('eda-tab', 'value'),

    State('eda-part', 'value')
)
def update_filters(tab, current_part):

    allowed = ALLOWED_PARTITIONS.get(
        tab,
        ['Train']
    )

    options = [
        {
            'label': PARTITIONS[value],
            'value': value
        }
        for value in allowed
    ]

    selected = (
        current_part
        if current_part in allowed
        else 'Train'
    )

    partition_disabled = len(allowed) == 1

    lags_disabled = tab != 'stats'

    lags_style = {
        'opacity': 0.45 if lags_disabled else 1,
        'transition': 'opacity 0.2s ease'
    }

    notes = {

        'mercado': (
            'Explora la dinámica del mercado en el conjunto '
            'completo o en sus particiones temporales.'
        ),

        'vol': (
            'El análisis de volatilidad está disponible para '
            'el conjunto completo y el entrenamiento.'
        ),

        'stats': (
            'Los diagnósticos estadísticos se realizan '
            'exclusivamente sobre el conjunto de entrenamiento.'
        ),

        'wave': (
            'Selecciona una variable y un periodo para '
            'explorar sus componentes wavelet.'
        )

    }

    return (
        options,
        selected,
        partition_disabled,
        lags_disabled,
        lags_style,
        notes.get(tab, '')
    )


@callback(
    Output('eda-content', 'children'),

    Input('eda-tab', 'value'),
    Input('eda-part', 'value'),
    Input('eda-dates', 'start_date'),
    Input('eda-dates', 'end_date'),
    Input('eda-var', 'value'),
    Input('eda-lags', 'value')
)
def render(tab, part, start, end, var, lags):

    allowed = ALLOWED_PARTITIONS.get(
        tab,
        ['Train']
    )

    if part not in allowed:
        part = 'Train'

    df = filter_data(
        part,
        start,
        end
    ).copy()

    if len(df) < 12:

        return dbc.Alert(
            'Selecciona un periodo con al menos '
            '12 observaciones dentro de la partición permitida.',
            color='warning'
        )

    df['squared_returns'] = (
        df['log_returns'] ** 2
    )

    s = (
        df[var]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(s) < 12:

        return dbc.Alert(
            'La variable seleccionada no contiene '
            'suficientes datos válidos.',
            color='warning'
        )

    metric_values = [

        (
            'OBSERVACIONES',
            f'{len(s):,}',
            'Sesiones válidas seleccionadas'
        ),

        (
            'MEDIA',
            fmt(s.mean()),
            VARS[var]
        ),

        (
            'DESVIACIÓN ESTÁNDAR',
            fmt(s.std()),
            VARS[var]
        ),

        (
            'MÁXIMO',
            fmt(s.max()),
            VARS[var]
        )

    ]

    metrics = dbc.Row(
        [
            dbc.Col(
                metric_card(
                    title,
                    value,
                    subtitle
                ),
                xs=6,
                xl=3
            )
            for title, value, subtitle in metric_values
        ],
        className='g-3 mb-3'
    )

    if tab == 'mercado':

        body = market_section(df)

    elif tab == 'vol':

        body = volatility_section(df)

    elif tab == 'stats':

        body = statistical_section(
            s,
            lags or 50
        )

    elif tab == 'wave':

        body = wavelet_section(s)

    else:

        return dbc.Alert(
            'Sección no disponible.',
            color='warning'
        )

    return [
        metrics,
        *body
    ]
