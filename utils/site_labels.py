"""De-identified site labels for figures and tables.

The OneFlorida+ contributor sites are stored on disk as `source_<N>` with
non-contiguous N because the numbering comes from the data owner's original site
index. The paper uses exactly six cohorts: the four-site prior source pool
{source_1, source_4, source_14, source_16}, the internal target (source_15) and
the external target (MIMIC-IV). Printing raw `source_<N>` invites "where did the
others go?", so the manuscript renames them Site A..E.

Mapping rule: the four pool sites are lettered A–D by ASCENDING source number,
mechanically (so the labelling cannot look chosen after seeing the results); the
internal target is Site E. source_3 / source_10 / source_11 were swept in earlier
exploration but are NOT part of the paper's fixed pool and get no letter.

    source_1  → A   source_16 → D
    source_4  → B   source_15 → E  (internal target)
    source_14 → C   MIMIC-IV  → MIMIC-IV (public, keeps its name)

DISPLAY LAYER ONLY. Never use these labels to build a path, a glob, a filename
or a `--hospital` argument: every results directory on disk is still named
`source_<N>`, and rewriting a path through this map would silently produce an
empty result set. Apply `site_label()` at the last moment — axis ticks, plot
titles, LaTeX cells.
"""
from __future__ import annotations

# Raw on-disk cohort name → manuscript label.
# The paper's prior source pool is exactly {source_1, source_4, source_14,
# source_16}; those are lettered A–D by ascending contributor index. Site E is
# the internal target (source_15); MIMIC-IV is the external target. source_3 /
# source_10 / source_11 are NOT part of the paper — they were swept historically
# but never enter the fixed prior pool, so they get no letter (they pass through
# as their raw name, which is a visible signal if one ever appears in a figure).
SITE_LABEL = {
    "source_1":  "Site A",
    "source_4":  "Site B",
    "source_14": "Site C",
    "source_16": "Site D",
    "source_15": "Site E",   # internal target
    "MIMIC-IV":  "MIMIC-IV",
    "MIMIC-III": "MIMIC-III",
}


def site_label(hospital: str) -> str:
    """Manuscript label for a cohort; unknown cohorts pass through unchanged.

    Pass-through (rather than KeyError) is intentional: a new site added mid-
    analysis should still plot, just under its raw name, so a missing mapping
    shows up as an odd axis tick instead of a crashed figure.
    """
    return SITE_LABEL.get(hospital, hospital)
