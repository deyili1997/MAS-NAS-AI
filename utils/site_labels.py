"""De-identified site labels for figures and tables.

The OneFlorida+ contributor sites are stored on disk as `source_<N>` with
non-contiguous N (1, 3, 4, 10, 11, 14, 15, 16) because the numbering comes from
the data owner's original site index, and several sites were dropped during
cohort selection (source_6 empty, source_9 too small, source_12 degenerate
death rate). Printing those raw gaps in a paper invites the question "where did
source_2 go?", so the manuscript renames them Site A..H.

Mapping rule: letters are assigned by ASCENDING source number, mechanically.
This is deliberate — a mapping that happened to put the target cohort last
would look like the labels were chosen after seeing the results.

    source_1 → A   source_10 → D   source_15 → G  (internal target)
    source_3 → B   source_11 → E   source_16 → H
    source_4 → C   source_14 → F   MIMIC-IV  → MIMIC-IV (public, keeps its name)

DISPLAY LAYER ONLY. Never use these labels to build a path, a glob, a filename
or a `--hospital` argument: every results directory on disk is still named
`source_<N>`, and rewriting a path through this map would silently produce an
empty result set. Apply `site_label()` at the last moment — axis ticks, plot
titles, LaTeX cells.
"""
from __future__ import annotations

# Raw on-disk cohort name → manuscript label.
SITE_LABEL = {
    "source_1":  "Site A",
    "source_3":  "Site B",
    "source_4":  "Site C",
    "source_10": "Site D",
    "source_11": "Site E",
    "source_14": "Site F",
    "source_15": "Site G",
    "source_16": "Site H",
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
