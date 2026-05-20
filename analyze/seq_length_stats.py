"""Offline per-patient sequence-length statistics.

Computes the total token count per patient (summing across all admissions, +1 for
[CLS]) that `utils/dataset.py:PreTrainEHRDataset.__getitem__` would produce.
This is needed to pick a hardcoded `MAX_SEQ_LEN` for the truncation feature
added in `utils/dataset.py` — chosen to cover ~95% of patients un-truncated.

Mimics the EXACT token-count logic from `_transform_pretrain_data` and
`__getitem__` (every code from every admission counted; nothing dropped here).

Per the user's design: keep the dataloader simple. This script is meant to be
run ONCE offline, the resulting recommended MAX_SEQ_LEN is then HARDCODED in
`utils/dataset.py`.

Usage:
    # On HPC (after OneFL+ owner's patient-level LOINC dedup is done):
    python analyze/seq_length_stats.py
    python analyze/seq_length_stats.py --hospitals source_10 MIMIC-IV
    python analyze/seq_length_stats.py --target_coverage 0.95 --round_to 128
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root so we can import utils.paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from utils.paths import get_processed_root  # noqa: E402


# Default hospitals: 8 OneFL+ keeps (per FINAL selection in plan) + MIMIC-III + MIMIC-IV.
# Excludes source_6 (empty file), source_9 (~900 rows too small), source_12 (death% degenerate).
DEFAULT_HOSPITALS = [
    "source_14", "source_4", "source_1", "source_15", "source_3",
    "source_16", "source_11", "source_10",
    "MIMIC-IV", "MIMIC-III",
]

# Columns that contribute tokens to the patient sequence — same order as
# dataset.py:_transform_pretrain_data's `if "diag/med/lab/pro" in token_type` block.
TOKEN_COLS = ["ICD9_CODE", "NDC", "LAB_TEST", "PRO_CODE"]


def _safe_count(value) -> int:
    """Mirror utils/dataset.py:_safe_to_list — count only valid string entries."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return sum(1 for v in value if v is not None and str(v) != "nan")
    if isinstance(value, np.ndarray):
        return sum(1 for v in value.tolist() if v is not None and str(v) != "nan")
    # pandas ArrowStringArray and similar — iterate
    try:
        return sum(1 for v in value if v is not None and str(v) != "nan")
    except TypeError:
        return 0


def patient_seq_lengths(pretrain_df: pd.DataFrame) -> np.ndarray:
    """Compute per-patient total seq length (matches dataset.py token assembly).

    For each SUBJECT_ID, iterate all rows (= admissions) and sum tokens across
    diag/med/lab/pro lists, then add 1 for the leading [CLS] token.
    Returns 1-D numpy array of seq lengths, one entry per unique patient.
    """
    # Group by subject + count tokens per row first (vectorize the inner loop)
    counts_per_row = np.zeros(len(pretrain_df), dtype=np.int64)
    for col in TOKEN_COLS:
        if col in pretrain_df.columns:
            counts_per_row += pretrain_df[col].map(_safe_count).to_numpy()

    # Sum per patient, +1 for [CLS]
    pretrain_df = pretrain_df.copy()
    pretrain_df["_token_count"] = counts_per_row
    per_patient = pretrain_df.groupby("SUBJECT_ID")["_token_count"].sum().to_numpy()
    return per_patient + 1  # +1 for [CLS]


def report_percentiles(name: str, lens: np.ndarray) -> dict:
    """Compute + print percentiles for one hospital. Returns the percentile dict."""
    pcts = {
        "n_patients": len(lens),
        "mean":  float(np.mean(lens)),
        "p50":   float(np.percentile(lens, 50)),
        "p75":   float(np.percentile(lens, 75)),
        "p90":   float(np.percentile(lens, 90)),
        "p95":   float(np.percentile(lens, 95)),
        "p99":   float(np.percentile(lens, 99)),
        "max":   int(np.max(lens)),
    }
    print(
        f"  {name:<12} "
        f"n={pcts['n_patients']:>7,} "
        f"mean={pcts['mean']:>7.1f} "
        f"p50={pcts['p50']:>6.0f} "
        f"p75={pcts['p75']:>6.0f} "
        f"p90={pcts['p90']:>6.0f} "
        f"p95={pcts['p95']:>6.0f} "
        f"p99={pcts['p99']:>6.0f} "
        f"max={pcts['max']:>6}"
    )
    return pcts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--hospitals", nargs="+", default=DEFAULT_HOSPITALS,
        help=f"Hospitals to include. Default: {DEFAULT_HOSPITALS}",
    )
    p.add_argument(
        "--target_coverage", type=float, default=0.95,
        help="Percentile for recommended MAX_SEQ_LEN (default 0.95 = p95).",
    )
    p.add_argument(
        "--round_to", type=int, default=128,
        help="Round recommended MAX_SEQ_LEN UP to a multiple of this (default 128).",
    )
    p.add_argument(
        "--pkl_name", type=str, default="mimic_pretrain.pkl",
        help="Pickle file to load per hospital (default mimic_pretrain.pkl).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"=== Per-patient sequence-length distribution ({args.pkl_name}) ===")
    print(
        f"  {'Hospital':<12} {'n_patients':>10} {'mean':>8} {'p50':>7} "
        f"{'p75':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}"
    )
    print("  " + "-" * 88)

    all_lens = []
    rows = []
    for hosp in args.hospitals:
        try:
            data_root = get_processed_root(hosp)
            pkl_path = data_root / args.pkl_name
            if not pkl_path.exists():
                print(f"  {hosp:<12} ⚠ missing: {pkl_path}")
                continue

            df = pickle.load(open(pkl_path, "rb"))
            if "SUBJECT_ID" not in df.columns:
                print(f"  {hosp:<12} ⚠ no SUBJECT_ID column")
                continue

            lens = patient_seq_lengths(df)
            pcts = report_percentiles(hosp, lens)
            pcts["hospital"] = hosp
            rows.append(pcts)
            all_lens.append(lens)
        except Exception as e:
            print(f"  {hosp:<12} ✗ ERROR: {str(e)[:60]}")

    if not all_lens:
        print("\nNo hospitals processed — abort.")
        sys.exit(1)

    # Aggregate across all hospitals (pool patients)
    print("  " + "-" * 88)
    pooled = np.concatenate(all_lens)
    pooled_pcts = report_percentiles("POOLED", pooled)

    # Recommend MAX_SEQ_LEN
    target_pct_label = f"p{int(args.target_coverage * 100)}"
    raw_recommend = int(np.percentile(pooled, args.target_coverage * 100))
    rounded = math.ceil(raw_recommend / args.round_to) * args.round_to

    print()
    print(f"=== Recommendation ===")
    print(f"  Target coverage: {args.target_coverage:.2%} ({target_pct_label} = {raw_recommend} tokens)")
    print(f"  Rounded UP to multiple of {args.round_to}:  MAX_SEQ_LEN = {rounded}")
    print()
    print(f"  Coverage breakdown @ MAX_SEQ_LEN = {rounded}:")
    for r in rows:
        lens = np.array([])  # will recompute coverage per site
        # Coverage = fraction of patients with seq_len <= MAX_SEQ_LEN
        # Recompute from cached pooled list per site
        pass
    # Just compute pooled coverage
    pooled_cov = (pooled <= rounded).mean()
    print(f"  Pooled patients UN-TRUNCATED: {pooled_cov:.2%} ({(1-pooled_cov)*100:.2f}% will get oldest tokens dropped)")

    # Per-site coverage
    print()
    print(f"  Per-site coverage @ MAX_SEQ_LEN = {rounded}:")
    print(f"  {'Hospital':<12} {'untruncated %':>14} {'max trunc loss':>16}")
    print("  " + "-" * 50)
    cum = 0
    for hosp, lens in zip([r["hospital"] for r in rows], all_lens):
        ut = (lens <= rounded).mean()
        max_loss = max(0, int(np.max(lens)) - rounded)
        print(f"  {hosp:<12} {ut:>13.2%} {max_loss:>16}")
        cum += len(lens)

    # Save CSV
    out_csv = _REPO_ROOT / "analyze" / "seq_length_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print()
    print(f"  ✓ saved per-hospital percentiles to {out_csv}")
    print()
    print(f"=== Next step ===")
    print(f"  Set in utils/dataset.py:")
    print(f"      MAX_SEQ_LEN = {rounded}")


if __name__ == "__main__":
    main()
