from dash import html


def metric_card(label, value, caption):

    return html.Div(
        [
            html.Div(
                label,
                className="metric-label"
            ),
            html.Div(
                value,
                className="metric-value"
            ),
            html.Div(
                caption,
                className="metric-caption"
            )
        ],
        className="metric-card"
    )


def page_heading(eyebrow, title, description):

    return html.Div(
        [
            html.Div(
                eyebrow,
                className="eyebrow"
            ),
            html.H1(
                title,
                className="page-title"
            ),
            html.P(
                description,
                className="page-description"
            )
        ],
        className="page-heading"
    )


def panel_heading(eyebrow, title, description=None):

    children = [
        html.Div(
            eyebrow,
            className="eyebrow"
        ),
        html.H2(
            title,
            className="panel-title"
        )
    ]

    if description:

        children.append(
            html.P(
                description,
                className="page-description"
            )
        )

    return html.Div(children)


def data_row(label, value):

    return html.Div(
        [
            html.Span(
                label,
                className="data-label"
            ),
            html.Span(
                value,
                className="data-value"
            )
        ],
        className="data-row"
    )