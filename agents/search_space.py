"""Shared search-space definition for all MAS-NAS agents.

256-architecture scalar grid (4 x 4 x 4 x 4) — the search space all reported
results use. Identical to tag v1-orig-256grid, byte-for-byte: the key order and
value order feed the supernet's subnet sampler, so they are part of the
experimental setup, not cosmetics.

NOTE: this MUST stay in sync with the mirror in run_pipeline.py. Changing it
invalidates the supernet checkpoint (the cache compares `choices` and forces a
retrain), and any results produced under a different grid are NOT comparable.
"""

CHOICES = {
    "mlp_ratio": [1, 2, 4, 8],
    "num_heads": [1, 2, 4, 8],
    "embed_dim": [32, 64, 128, 256],
    "depth": [1, 2, 4, 8],
}   # = 256 architectures
