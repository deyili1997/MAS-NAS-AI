"""Process-level counter for LLM API calls during a baseline / mas_search run.

Each top-level entry point (`baseline2.main`, `baseline3.main`, `baseline4.main`,
`mas_search.main`) calls `reset()` at startup; every Anthropic API call site
(in `baselines/baseline{2,3,4}.py` and `agents/{proposal,experiment,critic}_agent.py`)
calls `increment()` before invoking `client.messages.create(...)`. At end of run,
the entry point calls `get()` and serializes the count to `search_meta.json`
alongside the search/best CSVs.

Why a module-level counter (instead of threading on `args` or building a
real wrapper):
  - Each baseline / mas_search runs in its OWN Python process (one sbatch =
    one process), so module-level state is per-process. No threading concern.
  - Adding kwarg-passing through every helper would touch ~30 sites; this
    keeps the change to ~10 sites (one-liner increment + 2 lines reset/get).
  - Counter is read-only metadata for the cost-comparison table — it's fine
    if a malicious caller forgets to increment; the result just under-counts.

Usage at LLM call site:
    from utils.llm_counter import increment
    increment()
    response = client.messages.create(...)

Usage at top of main():
    from utils.llm_counter import reset, get as llm_count_get
    reset()
    ...

Usage at end of main():
    meta = {"llm_calls": llm_count_get(), ...}
"""
from __future__ import annotations

_count: int = 0


def reset() -> None:
    """Reset the counter to 0. Call once at the start of `main()`."""
    global _count
    _count = 0


def increment(n: int = 1) -> None:
    """Increment the counter by `n` (default 1, for one LLM API call)."""
    global _count
    _count += n


def get() -> int:
    """Return current count. Call once at the end of `main()`."""
    return _count
