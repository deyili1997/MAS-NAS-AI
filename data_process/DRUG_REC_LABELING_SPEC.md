# OneFlorida Drug-Recommendation Labeling Spec

For labeling the OneFlorida cohorts so the drug-recommendation task is
**consistent with the MIMIC-IV pipeline**. Applies to the two test hospitals
(OneFlorida held-out site + MIMIC-IV, already done) **and** the four prior-pool
source hospitals (`source_1`, `source_4`, `source_14`, `source_16`) — the
cross-hospital prior needs drug labels on the source hospitals too, otherwise the
method's Layer-2 prior is empty on this task.

## 1. Task

Standard **same-visit** medication recommendation (the GAMENet / SafeDrug
formulation), **not** next-visit forecasting.

- **Predict**: the set of medications administered during the **current** admission.
- **Condition on**: the **diagnoses** of the current admission + the patient's full prior history.
- **No time horizon** (it is about the current admission, not a future window).

## 2. Label vocabulary — FIXED, global, identical for every hospital

Predict a fixed set of **55 ATC-4 (4th-level) codes** (drugs present in ≥1% of
MIMIC-IV admissions). This vocabulary is **frozen and shared across all
hospitals** — do NOT compute a per-hospital top-k; the label space must be
identical everywhere or cross-hospital comparison and the prior both break.

Order matters (it is the fixed index order of the label vector). Standard ATC-4
codes (strip any prefix):

```
B01A, N02B, A01A, A06A, A02B, A12C, C10A, B05C, A12A, C07A,
N02A, N06A, A04A, C09A, A02A, H03A, H04A, A10A, C08C, J01D,
C03C, B03B, A07A, N05B, R03A, N03A, C01E, C03A, C01D, C05A,
C01B, C09C, J01M, N05A, M04A, A12B, A07E, J01C, N01A, C02D,
M01A, D07A, A03F, D04A, N05C, D11A, N07B, C01C, R01A, C03D,
J05A, J01E, R05D, J07A, A10B
```

**Medication → ATC-4 mapping**: map OneFlorida drug codes to ATC 4th level the
same way MIMIC-IV does — NDC → RxNorm → ATC-4, using GAMENet's
`ndc2atc_level4.csv` + `ndc2rxnorm_mapping.txt` (in `data_process/.../GAMENet/`).
Take the first 4 characters of the ATC code (the 4th level).

## 3. Per-admission label columns

For every admission, add two columns:

- **`CUR_MED_ATC`** — a length-55 binary list, in the exact order above: entry
  `i` = 1 if the admission's medication set contains code `i`, else 0.
- **`CUR_MED_FILTER`** — `1` if the admission contains ≥1 of the 55 codes, else
  `NaN`. (Admissions whose label is all-zero are dropped downstream.)

## 4. Input construction (for reference — enforced in our model code)

Our `utils/dataset.py` already handles the input masking via the task registry;
you only need to produce the label columns above. For completeness, at train time:

- **Target (current) admission**: only its **diagnoses** are fed to the model.
  Its medication tokens (= the label) and its lab / procedure tokens are dropped.
- **Prior admissions**: full codes (diagnoses / medications / labs / procedures).

## 5. Output pkl

Same structure as MIMIC-IV's `mimic_med_rec.pkl`:

- A pickled list `[finetune_df, val_df, test_df]`.
- **Patient-level 40/30/30 split** over admissions eligible for the task
  (`CUR_MED_FILTER` not NaN), fixed random seed, no patient in two splits.
- Each row is one admission carrying `CUR_MED_ATC` + `CUR_MED_FILTER` (alongside
  the existing diagnosis / medication / lab / procedure code columns).
- File name: `<hospital>_med_rec.pkl` (or match the local convention), placed
  next to the other processed pkls for that hospital.

## 6. Please report back

- **The prevalence of each of the 55 codes in OneFlorida** (fraction of
  admissions containing it). If any code is **< 1%** in OneFlorida (too sparse to
  learn there), tell us — we will decide whether to swap it. Any change to the
  vocabulary applies to **all** hospitals (it stays global), so please report
  before finalizing.

## 7. Authoritative reference code (in this repo)

- **`data_process/MIMIC-IV/build_med_rec.py`** — the exact label + filter +
  split logic MIMIC-IV used (the ground truth to replicate).
- **`utils/task_registry.py`** — `MED_LABEL_ORDER` (these 55 codes) + the
  `med_rec` task entry (`label_col=CUR_MED_ATC`, `filter_col=CUR_MED_FILTER`,
  `target_visit_diag_only=True`).
- **`data_process/MIMIC-IV/MIMIC-IV.ipynb`** — the med_rec cells, if a
  notebook-style reference is preferred.
