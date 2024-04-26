from typing import List

# **************************************** Plotly / Visualization
BLACK: str = "rgba(0,0,0,1)"
WHITE: str = "rgba(255,255,255,1)"

AXIS_COLOR = BLACK
AXIS_THK: float = 2.0
GRID_COLOR = "rgba(127,127,127,0.5)"
GRID_THK: float = 0.5
LEGEND_BACKGROUND_COLOR: str = "rgba(255,255,255, 0.8)"
PAPER_BACKGROUND_COLOR: str = "rgba(255,255,255,1)"
PLOT_HEIGHT: int = 500
PLOT_WIDTH: int = 800
FONT_SIZE: int = 19
FONT_FAMILY: str = "Computer Modern"

DEFAULT_LEGEND = dict(bgcolor=LEGEND_BACKGROUND_COLOR,)

DEFAULT_MARGIN = dict(
    l=80,
    r=20,
    t=20,
    b=20,
    pad=0,
)

DEFAULT_FONT = dict(
    family=FONT_FAMILY,
    color=BLACK,
    size=FONT_SIZE,
)

DEFAULT_AXIS = dict(
    # Title
    #title=dict(standoff=0.8),

    # Axis
    showline=True,
    zeroline=True,
    linewidth=AXIS_THK,
    linecolor=AXIS_COLOR,
    mirror=True,
    #side="right",

    # Grid
    showgrid=True,
    gridwidth=GRID_THK,
    gridcolor=GRID_COLOR,

    # Ticks
    showticklabels=True,
    ticks="inside",
    tickangle=0,
    tickfont=dict(
        family=FONT_FAMILY,
        size=FONT_SIZE,
        color=BLACK,
    ),
)

DEFAULT_LAYOUT = dict(
    autosize=True,
    xaxis=DEFAULT_AXIS,
    yaxis=DEFAULT_AXIS,
    height=PLOT_HEIGHT,
    width=PLOT_WIDTH,
    font=DEFAULT_FONT,
    showlegend=True,
    legend=DEFAULT_LEGEND,
    margin=DEFAULT_MARGIN,
    plot_bgcolor=PAPER_BACKGROUND_COLOR,
    paper_bgcolor=PAPER_BACKGROUND_COLOR,
)

PLOTLY_DEFAULT_COLORS: List[str] = [
    "rgba(31, 119, 180, 1)",
    "rgba(255, 127, 14, 1)",
    "rgba(44, 160, 44, 1)",
    "rgba(214, 39, 40, 1)",
    "rgba(148, 103, 189, 1)",
    "rgba(140, 86, 75, 1)",
    "rgba(227, 119, 194, 1)",
    "rgba(127, 127, 127, 1)",
    "rgba(188, 189, 34, 1)",
    "rgba(23, 190, 207, 1)",
]
