#!/usr/bin/env python
"""
Build the med_rec (standard drug-recommendation) pkl WITHOUT re-running the full
MIMIC-IV notebook.

The med_rec label only needs each admission's own NDC list, which already lives
in the processed `mimic.pkl`. So instead of re-running the expensive O(n^2)
NEXT_DIAG pipeline, this reads mimic.pkl, adds the med_rec label + filter, makes
a patient-level 40/30/30 split, and writes `mimic_med_rec.pkl`. The 4 existing
downstream pkls are NOT touched.

Single source of truth: the 55-code vocabulary is imported from
utils.task_registry.MED_LABEL_ORDER (identical, in order, to the >=1% document
frequency set), so the pkl can never drift from the registry the model reads.

This reproduces exactly what the MIMIC-IV.ipynb med_rec cells produce (same
vocabulary, same CUR_MED_ATC/CUR_MED_FILTER, same RANDOM_SEED+4 split).

Usage (server):
    cd /home/lideyi/MAS-NAS
    python data_process/MIMIC-IV/build_med_rec.py
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root on sys.path so `utils` imports work regardless of CWD.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from utils.paths import get_processed_root          # noqa: E402
from utils.task_registry import MED_LABEL_ORDER, N_MED  # noqa: E402

RANDOM_SEED = 123          # must match MIMIC-IV.ipynb
SPLIT_SEED = RANDOM_SEED + 4
TRAIN_RATIO, VAL_RATIO = 0.4, 0.3


def main():
    proc = get_processed_root("MIMIC-IV")
    mimic = pickle.load(open(proc / "mimic.pkl", "rb"))
    print(f"[load] mimic.pkl: {mimic.shape}  from {proc}")

    vocab_set = set(MED_LABEL_ORDER)
    code_sets = mimic["NDC"].apply(lambda x: set(x) & vocab_set)
    mimic = mimic.copy()
    mimic["CUR_MED_ATC"] = code_sets.apply(lambda s: [int(c in s) for c in MED_LABEL_ORDER])
    mimic["CUR_MED_FILTER"] = code_sets.apply(lambda s: 1 if len(s) > 0 else float("nan"))

    # Sanity
    sub = mimic[mimic["CUR_MED_FILTER"].notna()]
    lab = np.array(sub["CUR_MED_ATC"].tolist())
    assert lab.shape[1] == N_MED == 55, f"label dim {lab.shape[1]} != {N_MED}"
    print(f"[label] {N_MED} classes | eligible visits {len(sub)}/{len(mimic)} "
          f"({100*len(sub)/len(mimic):.1f}%) | mean prevalence {100*lab.mean():.1f}% "
          f"| positives/visit {lab.sum(1).mean():.2f}")

    # Patient-level 40/30/30 split over med-rec-eligible patients.
    pats = sorted(set(sub["SUBJECT_ID"].values.tolist()))
    n_ft, n_val = int(len(pats) * TRAIN_RATIO), int(len(pats) * VAL_RATIO)
    np.random.seed(SPLIT_SEED)
    np.random.shuffle(pats)
    ft_pat, val_pat, test_pat = pats[:n_ft], pats[n_ft:n_ft + n_val], pats[n_ft + n_val:]

    ft = mimic[mimic["SUBJECT_ID"].isin(set(ft_pat))]
    va = mimic[mimic["SUBJECT_ID"].isin(set(val_pat))]
    te = mimic[mimic["SUBJECT_ID"].isin(set(test_pat))]

    out = proc / "mimic_med_rec.pkl"
    pickle.dump([ft, va, te], open(out, "wb"))
    print(f"[OK] wrote {out}  ({len(ft)}/{len(va)}/{len(te)} rows; "
          f"{len(pats)} eligible patients)")


if __name__ == "__main__":
    main()
