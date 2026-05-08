from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#2f5d62", "#7d9b76", "#c9a86a", "#b08968", "#5b7e91", "#3a6e5f"]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 120,
        }
    )


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"
