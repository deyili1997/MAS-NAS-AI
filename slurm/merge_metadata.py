"""Merge per-task metadata CSVs from parallel pipeline runs into one file.

Usage:
    python slurm/merge_metadata.py --hospital source_15
"""
import argparse
import glob
from pathlib import Path
import pandas as pd

TASKS = ["death", "stay", "readmission", "next_diag_6m_pheno", "next_diag_12m_pheno"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hospital", required=True)
    p.add_argument("--tmp_root", default="/blue/mei.liu/lideyi/MAS-NAS/results/pipeline_tmp")
    p.add_argument("--output_root", default="/blue/mei.liu/lideyi/MAS-NAS/results")
    args = p.parse_args()

    dfs = []
    missing = []
    for task in TASKS:
        csv = Path(args.tmp_root) / f"{args.hospital}_{task}" / args.hospital / "metadata.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            print(f"  {task}: {len(df)} rows from {csv}")
            dfs.append(df)
        else:
            missing.append(task)
            print(f"  {task}: MISSING — {csv}")

    if missing:
        print(f"\n⚠ Missing tasks: {missing}. Merge aborted.")
        raise SystemExit(1)

    merged = pd.concat(dfs, ignore_index=True)
    out = Path(args.output_root) / args.hospital / "metadata.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing if present
    if out.exists():
        backup = out.with_suffix(".csv.pre_merge_backup")
        out.rename(backup)
        print(f"\nBacked up existing metadata to {backup}")

    merged.to_csv(out, index=False)
    print(f"\n✅ Merged {len(merged)} rows → {out}")
    print(f"   Tasks: {merged['task'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()
