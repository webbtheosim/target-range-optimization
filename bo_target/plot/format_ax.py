import json
from pathlib import Path

_STYLE_PATH = Path(__file__).resolve().parents[2] / "plot_style.json"

with open(_STYLE_PATH) as f:
    _format_params = json.load(f)


def format_ax(ax):
    """Apply consistent styling: tick direction, minor ticks, grid."""
    ax.format(**_format_params)
    ax.tick_params(top=True, right=True, bottom=True, left=True, which="both")
    ax.minorticks_on()