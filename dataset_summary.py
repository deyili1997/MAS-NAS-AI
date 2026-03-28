"""
Dataset Metadata Summary
========================
Generates a summary table of dataset characteristics for each hospital,
covering both pretraining and fine-tuning training datasets.
Output: one row per hospital with Pretrain_* and Finetune_* columns.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Generate dataset metadata summary")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--hospital", type=str, help="Single hospital name")
    group.add_argument("--hospitals", nargs="+", type=str, help="Multiple hospital names")
    p.add_argument("--output_dir", type=str, default="./results")
    return p.parse_args()


def _to_list(x):
    """Convert various array types to a plain list, empty list for scalars/NaN."""
    if isinstance(x, (list, np.ndarray)):
        return [str(v) for v in x if v is not None and str(v) != "nan"]
    # handle ArrowStringArray and other array-like types
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return [str(v) for v in x if v is not None and str(v) != "nan"]
    return []


def _safe_len(x):
    """Return length of a list-like, 0 for non-list or NaN."""
    return len(_to_list(x))


def _unique_codes(series):
    """Collect all unique codes from a Series of lists."""
    codes = set()
    for x in series:
        codes.update(_to_list(x))
    return codes


def summarize_dataset(df, prefix):
    """
    Compute summary statistics for one dataset (pretrain or finetune train).
    Returns a dict with prefixed keys.
    """
    code_cols = {
        "diag": "ICD9_CODE",
        "med": "NDC",
        "lab": "LAB_TEST",
        "pro": "PRO_CODE",
    }

    # Basic counts
    num_patients = df["SUBJECT_ID"].nunique()

    # Unique code counts
    unique_counts = {}
    for short, col in code_cols.items():
        if col in df.columns:
            unique_counts[short] = len(_unique_codes(df[col]))
        else:
            unique_counts[short] = 0

    # Per-patient aggregation
    per_patient_data = []
    for subj_id, group in df.groupby("SUBJECT_ID"):
        row = {
            "num_encounters": group["HADM_ID"].nunique() if "HADM_ID" in group.columns else len(group),
        }
        for short, col in code_cols.items():
            if col in group.columns:
                row[f"total_{short}"] = sum(_safe_len(x) for x in group[col])
            else:
                row[f"total_{short}"] = 0
        per_patient_data.append(row)

    per_patient_df = pd.DataFrame(per_patient_data)

    result = {
        f"{prefix}_num_samples": num_patients,
        f"{prefix}_diag_num": unique_counts["diag"],
        f"{prefix}_med_num": unique_counts["med"],
        f"{prefix}_lab_num": unique_counts["lab"],
        f"{prefix}_pro_num": unique_counts["pro"],
        f"{prefix}_avg_num_encounters": round(per_patient_df["num_encounters"].mean(), 2),
        f"{prefix}_avg_diag_per_patient": round(per_patient_df["total_diag"].mean(), 2),
        f"{prefix}_avg_med_per_patient": round(per_patient_df["total_med"].mean(), 2),
        f"{prefix}_avg_lab_per_patient": round(per_patient_df["total_lab"].mean(), 2),
        f"{prefix}_avg_pro_per_patient": round(per_patient_df["total_pro"].mean(), 2),
    }
    return result


def process_hospital(hospital, output_dir):
    """Process one hospital: load data, compute stats, save CSV."""
    data_root = Path(f"./data_process/{hospital}/{hospital}-processed")
    pretrain_path = data_root / "mimic_pretrain.pkl"
    downstream_path = data_root / "mimic_downstream.pkl"

    print(f"\n--- {hospital} ---")

    # Load pretrain data
    pretrain_df = pickle.load(open(pretrain_path, "rb"))
    print(f"  Pretrain: {len(pretrain_df)} rows, {pretrain_df['SUBJECT_ID'].nunique()} patients")
    pretrain_stats = summarize_dataset(pretrain_df, "Pretrain")

    # Load finetune train data (first element of the tuple)
    train_df, _, _ = pickle.load(open(downstream_path, "rb"))
    print(f"  Finetune train: {len(train_df)} rows, {train_df['SUBJECT_ID'].nunique()} patients")
    finetune_stats = summarize_dataset(train_df, "Finetune")

    # Merge into one row
    row = {"hospital": hospital, **pretrain_stats, **finetune_stats}

    # Save per-hospital CSV
    out_dir = Path(output_dir) / hospital
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    csv_path = out_dir / "dataset_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return row


def main():
    args = parse_args()
    hospitals = args.hospitals if args.hospitals else [args.hospital]

    all_rows = []
    for hospital in hospitals:
        row = process_hospital(hospital, args.output_dir)
        all_rows.append(row)

    # If multiple hospitals, save combined table
    if len(all_rows) > 1:
        combined = pd.DataFrame(all_rows)
        out_path = Path(args.output_dir) / "dataset_summary_all.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nCombined summary ({len(all_rows)} hospitals): {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    summary_df = pd.DataFrame(all_rows)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
