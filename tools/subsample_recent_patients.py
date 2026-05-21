"""Subsample a hospital's pkl files to keep only the most-recent N patients.

Used to bring big OneFL+ sites (source_1/4/14, 187K-232K patients) down to a
manageable size (~50K patients, matching source_10/15 scale) so run_pipeline
production fits the 14-day hpg-turin wall limit.

Workflow:
    1. Load 5 input pkls from the existing data_process/<site>/<site>-processed/
       directory (which are symlinks to /blue/.../data_shared/).
    2. Compute "top N patients by latest ADMITTIME" from the merged file
       (mimic.pkl) — gives a consistent patient ID set.
    3. Filter all 5 pkls to keep only those patients (preserves task alignment).
    4. Write filtered pkls to <output_dir>/<site>/<site>-processed/.
    5. (User then re-points data_process symlinks at the new location.)

Notes:
    - Within-patient admission order/dedup is preserved (we only drop entire
      patients, never modify their records).
    - Token vocabulary, LOINC dedup, and any other preprocessing stays the
      same — we're just reducing the patient count.
    - For multilabel finetune splits (mimic_nextdiag_*.pkl), the structure
      `(train, val, test)` is preserved with the same filter applied.

Usage:
    python tools/subsample_recent_patients.py \\
        --site source_14 \\
        --max_patients 50000 \\
        --output_dir /blue/mei.liu/lideyi/MAS-NAS/data_subsampled

    # Batch mode for all 3 big sites:
    for SITE in source_1 source_4 source_14; do
      python tools/subsample_recent_patients.py --site $SITE --max_patients 50000
    done
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root so we can import utils.paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from utils.paths import get_processed_root  # noqa: E402


PKL_NAMES = [
    "mimic.pkl",
    "mimic_pretrain.pkl",
    "mimic_downstream.pkl",
    "mimic_nextdiag_6m.pkl",
    "mimic_nextdiag_12m.pkl",
]


def compute_top_n_patients(merged_df: pd.DataFrame, n: int) -> set:
    """Top N patients by latest ADMITTIME (most recent first).

    Uses the MERGED file (all admissions across all splits) for consistent
    patient ranking. Each patient's "score" is their most recent admission
    timestamp; the N patients with the highest scores are kept.
    """
    if "SUBJECT_ID" not in merged_df.columns:
        raise ValueError("merged_df missing SUBJECT_ID column")
    if "ADMITTIME" not in merged_df.columns:
        raise ValueError("merged_df missing ADMITTIME column — cannot rank by recency")

    # Parse ADMITTIME defensively
    merged_df = merged_df.copy()
    merged_df["ADMITTIME"] = pd.to_datetime(merged_df["ADMITTIME"], errors="coerce")
    n_bad = merged_df["ADMITTIME"].isna().sum()
    if n_bad > 0:
        print(f"  ⚠ {n_bad:,} rows had unparseable ADMITTIME (dropped from ranking)")
        merged_df = merged_df.dropna(subset=["ADMITTIME"])

    n_total = merged_df["SUBJECT_ID"].nunique()
    if n_total <= n:
        print(f"  Already have only {n_total:,} patients (≤ target {n:,}) — keeping all")
        return set(merged_df["SUBJECT_ID"].unique())

    # Latest admission per patient
    latest_per_patient = merged_df.groupby("SUBJECT_ID")["ADMITTIME"].max()
    top_n_ids = set(latest_per_patient.sort_values(ascending=False).head(n).index)

    # Diagnostic
    cutoff = latest_per_patient.sort_values(ascending=False).iloc[n - 1]
    print(f"  Patients before: {n_total:,}")
    print(f"  Patients kept:   {len(top_n_ids):,} (most recent {n:,})")
    print(f"  Cutoff ADMITTIME (oldest kept patient's latest visit): {cutoff}")
    return top_n_ids


def filter_by_patients(obj, keep_ids: set):
    """Filter a loaded pkl object by patient ID. Handles single DataFrame and
    tuple/list of DataFrames (for downstream/nextdiag splits)."""
    if isinstance(obj, pd.DataFrame):
        return obj[obj["SUBJECT_ID"].isin(keep_ids)].reset_index(drop=True)
    if isinstance(obj, (tuple, list)):
        return type(obj)(filter_by_patients(x, keep_ids) for x in obj)
    raise TypeError(f"Cannot filter object of type {type(obj).__name__}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--site", required=True,
                   help="Short site name (e.g., source_14). Reads from data_process/<site>/<site>-processed/")
    p.add_argument("--max_patients", type=int, default=50000,
                   help="Target patient count (default 50000)")
    p.add_argument("--output_dir", type=str,
                   default="/blue/mei.liu/lideyi/MAS-NAS/data_subsampled",
                   help="Output directory root. Creates <out>/<site>/<site>-processed/")
    p.add_argument("--input_dir", type=str, default=None,
                   help="Override input dir (default: utils.paths.get_processed_root(site))")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve input/output dirs
    if args.input_dir:
        in_root = Path(args.input_dir)
    else:
        in_root = get_processed_root(args.site)
    out_root = Path(args.output_dir) / args.site / f"{args.site}-processed"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Subsample {args.site} to {args.max_patients:,} most-recent patients ===")
    print(f"  Input:  {in_root}")
    print(f"  Output: {out_root}")
    print()

    # Step 1: Load merged file to determine patient ranking
    print(f"[1/2] Determining top {args.max_patients:,} patients from mimic.pkl...")
    merged = pickle.load(open(in_root / "mimic.pkl", "rb"))
    keep_ids = compute_top_n_patients(merged, args.max_patients)
    print()

    # Step 2: Filter each pkl
    print(f"[2/2] Filtering 5 pkl files...")
    summary_rows = []
    for name in PKL_NAMES:
        src = in_root / name
        dst = out_root / name
        if not src.exists():
            print(f"  ⚠ skipping (missing): {name}")
            continue
        obj = pickle.load(open(src, "rb"))
        filtered = filter_by_patients(obj, keep_ids)

        # Count rows before/after
        if isinstance(obj, pd.DataFrame):
            before, after = len(obj), len(filtered)
            before_p = obj["SUBJECT_ID"].nunique()
            after_p = filtered["SUBJECT_ID"].nunique()
        elif isinstance(obj, (tuple, list)):
            before = sum(len(df) for df in obj)
            after = sum(len(df) for df in filtered)
            before_p = len(set().union(*[set(df["SUBJECT_ID"].unique()) for df in obj]))
            after_p = len(set().union(*[set(df["SUBJECT_ID"].unique()) for df in filtered]))
        else:
            before = after = before_p = after_p = 0

        pickle.dump(filtered, open(dst, "wb"))
        size_mb = dst.stat().st_size / 1024**2
        ratio = (after / before * 100) if before > 0 else 0
        print(f"  ✓ {name:<28} rows {before:>9,} → {after:>9,} ({ratio:5.1f}%)  "
              f"patients {before_p:>7,} → {after_p:>7,}  size: {size_mb:.1f} MB")
        summary_rows.append({
            "file": name, "rows_before": before, "rows_after": after,
            "patients_before": before_p, "patients_after": after_p,
            "out_size_mb": round(size_mb, 1),
        })

    # Summary CSV for audit trail
    pd.DataFrame(summary_rows).to_csv(out_root / "_subsample_summary.csv", index=False)
    print()
    print(f"=== Done ===")
    print(f"  ✓ Subsampled pkls at: {out_root}")
    print(f"  ✓ Summary saved to:   {out_root}/_subsample_summary.csv")
    print()
    print(f"=== Next step: re-link symlinks ===")
    DST_LINK = f"/blue/mei.liu/lideyi/MAS-NAS/data_process/{args.site}/{args.site}-processed"
    print(f"  mkdir -p {DST_LINK}")
    for name in PKL_NAMES:
        print(f"  ln -sf {out_root}/{name}  {DST_LINK}/{name}")


if __name__ == "__main__":
    main()
