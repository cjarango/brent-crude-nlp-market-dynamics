from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

APP_NAME = "BRENT QUANT"
APP_SUBTITLE = "Terminal de investigación cuantitativa"
APP_VERSION = "1.0.0"

COLORS = {
    "background": "#F4F6F9",
    "surface": "#FFFFFF",
    "navy": "#172B4D",
    "gold": "#B79559",
    "slate": "#64748B"
}

FONT_FAMILY = "Inter, sans-serif"
FONT_MONO = "IBM Plex Mono, monospace"

CHART_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["surface"],
        "plot_bgcolor": COLORS["surface"],
        "font": {
            "family": FONT_FAMILY,
            "color": COLORS["slate"],
            "size": 12
        },
        "xaxis": {
            "gridcolor": "rgba(100, 116, 139, 0.12)",
            "showgrid": True,
            "zeroline": False
        },
        "yaxis": {
            "gridcolor": "rgba(100, 116, 139, 0.12)",
            "showgrid": True,
            "zeroline": False,
            "side": "right"
        },
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": COLORS["surface"],
            "bordercolor": COLORS["slate"],
            "font_color": COLORS["navy"]
        },
        "margin": {
            "l": 25,
            "r": 60,
            "t": 35,
            "b": 35
        },
        "colorway": [
            COLORS["navy"],
            COLORS["gold"],
            COLORS["slate"]
        ]
    }
}