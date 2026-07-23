"""Nature-style panel letters (a, b, c, ...) for multi-panel figures.

Every composite figure in the manuscript is labelled with a bold lowercase
letter at the top-left of each panel, so the text can refer to "Fig. 1c" rather
than "the third panel of Fig. 1".

The letter is drawn in AXES coordinates at a position slightly outside the axes
box, which puts it left of the y-label and above the panel title. Because it
sits outside the axes, the figure MUST be saved with bbox_inches="tight" or the
letters are clipped — every caller in analyze/ already does this.

Deliberately NOT applied to plot_regression_combined.py: that figure's published
PNG is the archived 256-grid version and cannot be regenerated (its raw data was
overwritten by the 1008-grid retrain), so restyling the script would only make
the code disagree with the figure that actually ships.
"""
from __future__ import annotations

from string import ascii_lowercase


def panel_label(ax, index, *, fontsize: int = 15, dx: float = -0.10,
                dy: float = 1.12) -> None:
    """Draw the panel letter for `index` (0 -> 'a') at the top-left of `ax`.

    index may also be a ready-made string, for figures whose panels are numbered
    by something other than their position in a loop.

    dx/dy are in axes-fraction coordinates; the defaults clear a y-label of two
    or three digits and a single-line title. Panels with a long y-label or a
    two-line title need a more negative dx / larger dy.
    """
    letter = ascii_lowercase[index] if isinstance(index, int) else index
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")
