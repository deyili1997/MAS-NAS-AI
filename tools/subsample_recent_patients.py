"""Subsample a hospital's pkl files by ADMITTIME cutoff date.

Used to bring big OneFL+ sites (source_1/4/14, 187K-232K patients) down to a
manageable size (~50K patients, matching source_10/15 scale) so run_pipeline
production fits the 14-day hpg-turin wall limit.

Paper-friendly framing: keep patients whose latest admission is on or after a
SINGLE cutoff date (cleaner than "top N patients" arbitrary count). The cutoff
date is auto-computed from a target patient count and ROUNDED to month-start
for paper-defensibility, OR supplied directly via --cutoff_date.

Workflow:
    1. Load 5 input pkls from data_process/<site>/<site>-processed/ (symlinks).
    2. Determine cutoff date:
         - If --cutoff_date given: use it directly.
         - Else compute from --target_patients: find the N-th patient's latest
           ADMITTIME, then ROUND DOWN to the 1st of that month for clean paper
           framing. Resulting count is approximately N (usually slightly more).
    3. Keep all patients whose LATEST admission ≥ cutoff_date.
    4. Filter all 5 pkls to that patient set (preserves task alignment).
    5. Write filtered pkls + a manifest with the cutoff date for paper writing.

Notes:
    - Within-patient admission order/dedup is preserved (we only drop entire
      patients, never modify their records).
    - Token vocabulary, LOINC dedup, and any other preprocessing stays the
      same — we're just reducing the patient count.
    - For multilabel finetune splits (mimic_nextdiag_*.pkl), the structure
      `(train, val, test)` is preserved with the same filter applied.

Usage:
    # Auto-compute date from target count (rounded to month-start):
    python tools/subsample_recent_patients.py --site source_14 --target_patients 50000

    # Or specify date directly (e.g., shared across sites for paper consistency):
    python tools/subsample_recent_patients.py --site source_14 --cutoff_date 2018-01-01

    # Batch mode (per-site auto-date):
    for SITE in source_1 source_4 source_14; do
      python tools/subsample_recent_patients.py --site $SITE --target_patients 50000
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


def parse_admittime(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Defensive ADMITTIME → datetime parse; drop unparseable rows. Returns
    a copy with ADMITTIME as pd.Timestamp dtype."""
    if "SUBJECT_ID" not in merged_df.columns:
        raise ValueError("merged_df missing SUBJECT_ID column")
    if "ADMITTIME" not in merged_df.columns:
        raise ValueError("merged_df missing ADMITTIME column — cannot rank by recency")
    merged_df = merged_df.copy()
    merged_df["ADMITTIME"] = pd.to_datetime(merged_df["ADMITTIME"], errors="coerce")
    n_bad = merged_df["ADMITTIME"].isna().sum()
    if n_bad > 0:
        print(f"  ⚠ {n_bad:,} rows had unparseable ADMITTIME (dropped from ranking)")
        merged_df = merged_df.dropna(subset=["ADMITTIME"])
    return merged_df


def compute_cutoff_from_target(merged_df: pd.DataFrame, target_n: int) -> pd.Timestamp:
    """Find the ADMITTIME date that, used as ≥ cutoff on per-patient latest
    admission, yields approximately `target_n` patients.

    Algorithm: rank patients by latest admission DESC, find the target_n-th
    patient's latest admission, ROUND DOWN to the 1st of that month for a
    clean paper-friendly cutoff. Actual count will be slightly ≥ target_n
    because rounding extends the window backward.
    """
    latest_per_patient = merged_df.groupby("SUBJECT_ID")["ADMITTIME"].max()
    n_total = len(latest_per_patient)
    if n_total <= target_n:
        cutoff = latest_per_patient.min()
        print(f"  Only {n_total:,} patients total (≤ target {target_n:,}) — using earliest as cutoff")
        return cutoff.replace(day=1)
    raw_cutoff = latest_per_patient.sort_values(ascending=False).iloc[target_n - 1]
    # Round DOWN to month-start
    cutoff = raw_cutoff.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    print(f"  Raw cutoff (target_n={target_n:,}-th patient's latest): {raw_cutoff}")
    print(f"  Rounded to month-start:                                  {cutoff.date()}")
    return cutoff


def patients_after_cutoff(merged_df: pd.DataFrame, cutoff: pd.Timestamp) -> set:
    """Return set of patient IDs whose LATEST admission ≥ cutoff."""
    latest_per_patient = merged_df.groupby("SUBJECT_ID")["ADMITTIME"].max()
    keep_ids = set(latest_per_patient[latest_per_patient >= cutoff].index)
    print(f"  Patients before:  {merged_df['SUBJECT_ID'].nunique():,}")
    print(f"  Cutoff:           {cutoff.date()}  (latest_admission >= this)")
    print(f"  Patients kept:    {len(keep_ids):,}")
    return keep_ids


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
    group = p.add_mutually_exclusive_group()
    group.add_argument("--target_patients", type=int, default=None,
                       help="Auto-compute cutoff date from target patient count (rounded "
                            "to month-start). Default 50000 if neither flag given.")
    group.add_argument("--cutoff_date", type=str, default=None,
                       help="Explicit cutoff date (YYYY-MM-DD). Patients with latest "
                            "admission ≥ this are kept. Mutually exclusive with --target_patients.")
    p.add_argument("--output_dir", type=str,
                   default="/blue/mei.liu/lideyi/MAS-NAS/data_subsampled",
                   help="Output directory root. Creates <out>/<site>/<site>-processed/")
    p.add_argument("--input_dir", type=str, default=None,
                   help="Override input dir (default: utils.paths.get_processed_root(site))")
    return p.parse_args()


def main():
    args = parse_args()

    # Default to target_patients=50000 if neither flag given
    if args.target_patients is None and args.cutoff_date is None:
        args.target_patients = 50000

    # Resolve input/output dirs
    if args.input_dir:
        in_root = Path(args.input_dir)
    else:
        in_root = get_processed_root(args.site)
    out_root = Path(args.output_dir) / args.site / f"{args.site}-processed"
    out_root.mkdir(parents=True, exist_ok=True)

    mode_str = (f"cutoff_date={args.cutoff_date}" if args.cutoff_date
                else f"target_patients={args.target_patients:,} (auto-date)")
    print(f"=== Subsample {args.site} by ADMITTIME cutoff ({mode_str}) ===")
    print(f"  Input:  {in_root}")
    print(f"  Output: {out_root}")
    print()

    # Step 1: Load merged + determine cutoff date
    print(f"[1/2] Loading mimic.pkl + determining cutoff date...")
    merged = pickle.load(open(in_root / "mimic.pkl", "rb"))
    merged = parse_admittime(merged)

    if args.cutoff_date:
        cutoff = pd.Timestamp(args.cutoff_date)
    else:
        cutoff = compute_cutoff_from_target(merged, args.target_patients)

    keep_ids = patients_after_cutoff(merged, cutoff)
    print()

    # Step 2: Filter each pkl
    print(f"[2/2] Filtering 5 pkl files (keeping patients with latest admission ≥ {cutoff.date()})...")
    summary_rows = []
    for name in PKL_NAMES:
        src = in_root / name
        dst = out_root / name
        if not src.exists():
            print(f"  ⚠ skipping (missing): {name}")
            continue
        obj = pickle.load(open(src, "rb"))
        filtered = filter_by_patients(obj, keep_ids)

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

    # Manifest (csv + paper_summary.txt) for audit trail + paper writing
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_root / "_subsample_summary.csv", index=False)
    with open(out_root / "_paper_summary.txt", "w") as f:
        f.write(f"OneFL+ site {args.site} subsampling — for paper Section X\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Cutoff date: {cutoff.date()}\n")
        f.write(f"Criterion: patients whose latest hospital admission was on or after {cutoff.date()}\n\n")
        f.write(f"Total patients kept: {len(keep_ids):,}\n")
        f.write(f"Total patients before: {merged['SUBJECT_ID'].nunique():,}\n")
        f.write(f"Retention rate: {len(keep_ids)/merged['SUBJECT_ID'].nunique()*100:.1f}%\n\n")
        f.write(f"Per-file row counts:\n")
        for r in summary_rows:
            f.write(f"  {r['file']:<28} rows: {r['rows_after']:>9,} (was {r['rows_before']:>9,})\n")

    print()
    print(f"=== Done ===")
    print(f"  ✓ Subsampled pkls at:   {out_root}")
    print(f"  ✓ Audit CSV at:         {out_root}/_subsample_summary.csv")
    print(f"  ✓ Paper summary at:     {out_root}/_paper_summary.txt")
    print()
    print(f"=== Paper-ready statement ===")
    print(f"  \"For {args.site}, we restricted to patients whose latest hospital")
    print(f"   admission occurred on or after {cutoff.date()}, yielding")
    print(f"   {len(keep_ids):,} patients (from {merged['SUBJECT_ID'].nunique():,} total).\"")
    print()
    print(f"=== Next step: re-link symlinks ===")
    DST_LINK = f"/blue/mei.liu/lideyi/MAS-NAS/data_process/{args.site}/{args.site}-processed"
    print(f"  mkdir -p {DST_LINK}")
    for name in PKL_NAMES:
        print(f"  ln -sf {out_root}/{name}  {DST_LINK}/{name}")


if __name__ == "__main__":
    main()
