from __future__ import annotations

import streamlit as st

PRIMARY = "#2E86AB"
ACCENT = "#5DADE2"
BG = "#0E1117"
BG_SECONDARY = "#1E2530"
TEXT = "#FAFAFA"
GRID = "rgba(250, 250, 250, 0.08)"

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E", "#E5C687", "#5DADE2"]

_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 280px;
    max-width: 280px;
}
[data-testid="stSidebarNav"] a {
    border-radius: 6px;
    padding: 6px 10px;
    margin: 2px 0;
    color: #FAFAFA !important;
    opacity: 0.75;
}
[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(46, 134, 171, 0.18);
    opacity: 1;
}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNav"] li > div:has(> a[aria-current="page"]) {
    background-color: #2E86AB !important;
    border-left: 4px solid #5DADE2 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
</style>
"""


def inject_sidebar_style() -> None:
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)


def apply_plotly_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(t=40, r=20, b=40, l=20),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickcolor=GRID)
    return fig
