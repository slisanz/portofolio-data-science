from __future__ import annotations

import streamlit as st

_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {
    min-width: 280px !important;
    max-width: 280px !important;
    transform: none !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"],
button[kind="header"][aria-label*="sidebar"],
div[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 6px;
    padding: 6px 10px;
    margin: 2px 0;
    color: #1f2937 !important;
    opacity: 0.85;
}
[data-testid="stSidebarNav"] a:hover {
    background-color: rgba(47, 93, 98, 0.12);
    opacity: 1;
}
[data-testid="stSidebarNav"] a[aria-current="page"],
[data-testid="stSidebarNav"] li > div:has(> a[aria-current="page"]) {
    background-color: #2f5d62 !important;
    border-left: 4px solid #1f2937 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}
</style>
"""


def inject_sidebar_style() -> None:
    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
