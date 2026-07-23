"""Panel grid for multi-panel figures: at most 3 per row, last row centred.

A 1x5 strip of panels is unreadable at IEEE column width — the panels end up
about 1.5 in wide once the figure is scaled to fit. Reflowing 5 panels as 3+2
roughly doubles the width each panel gets.

`plt.subplots(2, 3)` cannot do this: it leaves an empty slot at the bottom RIGHT,
so the two bottom panels sit left-aligned under the first two of the top row.
The fix is a GridSpec twice as wide as the panel count — every panel spans two
columns, and a short row is pushed right by the half-panel offset that centres
it. With 6 columns: a 3-panel row starts at 0, a 2-panel row at 1, a 1-panel row
at 2.

    from utils.fig_layout import panel_grid
    fig, axes = panel_grid(len(TASKS))
    for i, (ax, task) in enumerate(zip(axes, TASKS)):
        ...

`axes` is a flat list in reading order, so it drops straight into the `zip(axes,
TASKS)` loops the plot scripts already use.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

_MAX_PER_ROW = 3
_UNITS = 2          # grid columns per panel; the odd offset is what centres a short row


def panel_grid(n_panels: int, *, panel_w: float = 4.6, panel_h: float = 4.3,
               max_per_row: int = _MAX_PER_ROW):
    """Return (fig, axes) with `n_panels` axes, <= max_per_row per row, last row centred.

    axes is a flat list in reading order. Panel size is per-panel, so the figure
    grows with the grid rather than squeezing panels as the count rises.
    """
    if n_panels <= 0:
        raise ValueError("n_panels must be positive")

    rows = [min(max_per_row, n_panels - i * max_per_row)
            for i in range((n_panels + max_per_row - 1) // max_per_row)]
    n_rows, n_cols = len(rows), max_per_row * _UNITS

    fig = plt.figure(figsize=(panel_w * max_per_row, panel_h * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    axes = []
    for r, k in enumerate(rows):
        offset = (n_cols - k * _UNITS) // 2      # 0 for a full row, 1 for a row of 2
        for c in range(k):
            start = offset + c * _UNITS
            axes.append(fig.add_subplot(gs[r, start:start + _UNITS]))
    return fig, axes
