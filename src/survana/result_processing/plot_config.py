from typing import Any

import seaborn as sns
from matplotlib import pyplot as plt


def _set_plt_params(remove_grid=False) -> None:
    """Set parameters and use seaborn theme to plot."""
    sns.set_theme()
    if remove_grid:
        sns.set_style("whitegrid", {"axes.grid": False})
    params: dict[str, Any] = {
        "font.family": "DejaVu Serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
        "axes.titlesize": "large",
        "axes.labelsize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "medium",
        "savefig.dpi": 300,
    }
    plt.rcParams.update(params)
    plt.style.use("tableau-colorblind10")
