"""
Styled HTML / CSS helpers for rich notebook rendering.

Uses IPython.display to inject professional-looking statistical summaries
directly in notebook cells.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import pandas as pd

# Guard import – works outside IPython but display is a no-op.
try:
    from IPython.display import HTML, display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


# ======================================================================
# CSS template
# ======================================================================
_TABLE_CSS = """
<style>
.tsa-summary-table {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 13px;
}
.tsa-summary-table caption {
    font-size: 15px;
    font-weight: 600;
    text-align: left;
    padding: 8px 0;
    color: #2c3e50;
}
.tsa-summary-table th {
    background-color: #2c3e50;
    color: #ecf0f1;
    padding: 8px 12px;
    text-align: left;
}
.tsa-summary-table td {
    padding: 6px 12px;
    border-bottom: 1px solid #ecf0f1;
}
.tsa-summary-table tr:nth-child(even) {
    background-color: #f9f9f9;
}
.tsa-summary-table tr:hover {
    background-color: #eaf2f8;
}
.tsa-metric-card {
    display: inline-block;
    min-width: 140px;
    margin: 6px 8px;
    padding: 12px 16px;
    border-radius: 8px;
    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    color: #fff;
    font-family: 'Segoe UI', sans-serif;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.tsa-metric-card .metric-label {
    font-size: 11px;
    text-transform: uppercase;
    opacity: 0.85;
}
.tsa-metric-card .metric-value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}
</style>
"""


# ======================================================================
# Public API
# ======================================================================
def styled_summary(
    df: pd.DataFrame,
    caption: str = "Statistical Summary",
    show: bool = True,
) -> str:
    """
    Render a DataFrame as a styled HTML table inside a notebook cell.

    Parameters
    ----------
    df : pd.DataFrame
        The data to render (e.g. ``df.describe()``).
    caption : str
        Table caption text.
    show : bool
        When *True* and running in IPython, call ``display(HTML(…))``.

    Returns
    -------
    str
        Raw HTML string.
    """
    rows_html = ""
    # Header
    rows_html += "<tr>" + "".join(f"<th>{c}</th>" for c in [""] + list(df.columns)) + "</tr>\n"
    # Body
    for idx, row in df.iterrows():
        cells = f"<td><b>{idx}</b></td>"
        for val in row:
            if isinstance(val, float):
                cells += f"<td>{val:,.4f}</td>"
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>\n"

    html = (
        _TABLE_CSS
        + f'<table class="tsa-summary-table"><caption>{caption}</caption>\n'
        + rows_html
        + "</table>"
    )

    if show and _HAS_IPYTHON:
        display(HTML(html))
    return html


def metric_cards(
    metrics: Dict[str, Union[float, str]],
    show: bool = True,
) -> str:
    """
    Render key metrics as styled cards in a notebook cell.

    Parameters
    ----------
    metrics : dict[str, float | str]
        Metric label → value pairs.
    show : bool
        Auto-display in IPython.

    Returns
    -------
    str
        Raw HTML string.
    """
    cards = ""
    for label, value in metrics.items():
        if isinstance(value, float):
            formatted = f"{value:,.4f}"
        else:
            formatted = str(value)
        cards += (
            f'<div class="tsa-metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{formatted}</div>'
            f"</div>\n"
        )

    html = _TABLE_CSS + f'<div style="display:flex;flex-wrap:wrap;">{cards}</div>'

    if show and _HAS_IPYTHON:
        display(HTML(html))
    return html
