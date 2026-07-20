"""
Task Registry
==============
Single source of truth for task-specific properties: type (binary vs
multilabel), output dimensionality, and which preprocessed pkl to load.

Used by every entry script + dataset + engine + experiment_agent so that
adding/changing a task requires only editing this file.
"""

from pathlib import Path

# Phenotype label list — must match LABEL_ORDER in MIMIC-IV.ipynb's helper cell.
PHENO_LABEL_ORDER = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and related",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Pneumonia",
    "Septicemia (except in labor)",
]
N_PHENO = len(PHENO_LABEL_ORDER)  # 18

# Drug-recommendation label list — ATC-4 codes present in >= 1% of MIMIC-IV
# admissions (55 of 140), frozen in document-frequency order. MUST match
# MED_LABEL_ORDER printed by build_med_vocab() in MIMIC-IV.ipynb.
#
# Why the >= 1% cut (not top-k, not all 140): standard med-rec benchmarks
# (GAMENet/SafeDrug) use the full ATC vocabulary minus ultra-rare codes. All 140
# codes average only ~2.4% prevalence (76 are < 0.5% — too sparse to learn); the
# >= 1% threshold keeps the 55 learnable ones (prevalence 1.0%–21.8%).
MED_LABEL_ORDER = [
    "MED_B01A",  # antithrombotic agents                 21.8%
    "MED_N02B",  # other analgesics / antipyretics       21.7%
    "MED_A01A",  # stomatological preparations           21.1%
    "MED_A06A",  # laxatives                             21.0%
    "MED_A02B",  # peptic ulcer / GORD drugs             17.7%
    "MED_A12C",  # other mineral supplements             16.4%
    "MED_C10A",  # lipid modifying agents                16.3%
    "MED_B05C",  # irrigating solutions                  12.2%
    "MED_A12A",  # calcium                               11.0%
    "MED_C07A",  # beta blocking agents                  10.1%
    "MED_N02A",  # opioids                                9.9%
    "MED_N06A",  # antidepressants                        9.3%
    "MED_A04A",  # antiemetics / antinauseants            7.2%
    "MED_C09A",  # ACE inhibitors, plain                  7.0%
    "MED_A02A",  # antacids                               6.3%
    "MED_H03A",  # thyroid preparations                   6.0%
    "MED_H04A",  # pancreatic hormones (glucagon)         5.9%
    "MED_A10A",  # insulins and analogues                 5.7%
    "MED_C08C",  # selective calcium channel blockers     5.3%
    "MED_J01D",  # other beta-lactam antibacterials       4.4%
    "MED_C03C",  # high-ceiling diuretics                 4.1%
    "MED_B03B",  # vitamin B12 / folic acid               3.8%
    "MED_A07A",  # intestinal antiinfectives              3.8%
    "MED_N05B",  # anxiolytics                            3.7%
    "MED_R03A",  # adrenergics, inhalants                 3.6%
    "MED_N03A",  # antiepileptics                         3.6%
    "MED_C01E",  # other cardiac preparations             3.1%
    "MED_C03A",  # thiazide diuretics                     2.9%
    "MED_C01D",  # vasodilators (cardiac)                 2.8%
    "MED_C05A",  # antihemorrhoidals, topical             2.8%
    "MED_C01B",  # antiarrhythmics class I/III            2.7%
    "MED_C09C",  # angiotensin-II receptor blockers       2.5%
    "MED_J01M",  # quinolone antibacterials               2.5%
    "MED_N05A",  # antipsychotics                         2.3%
    "MED_M04A",  # antigout preparations                  2.3%
    "MED_A12B",  # potassium                              2.2%
    "MED_A07E",  # intestinal antiinflammatory            1.9%
    "MED_J01C",  # penicillins                            1.9%
    "MED_N01A",  # general anesthetics                    1.8%
    "MED_C02D",  # arteriolar smooth muscle agents        1.8%
    "MED_M01A",  # NSAIDs                                 1.8%
    "MED_D07A",  # corticosteroids, dermatological        1.8%
    "MED_A03F",  # propulsives                            1.7%
    "MED_D04A",  # antipruritics                          1.7%
    "MED_N05C",  # hypnotics and sedatives                1.5%
    "MED_D11A",  # other dermatological preparations      1.5%
    "MED_N07B",  # drugs for addictive disorders          1.4%
    "MED_C01C",  # cardiac stimulants (non-glycoside)     1.4%
    "MED_R01A",  # nasal decongestants, topical           1.3%
    "MED_C03D",  # potassium-sparing agents               1.3%
    "MED_J05A",  # direct-acting antivirals               1.2%
    "MED_J01E",  # sulfonamides / trimethoprim            1.1%
    "MED_R05D",  # cough suppressants                     1.1%
    "MED_J07A",  # bacterial vaccines                     1.1%
    "MED_A10B",  # oral blood-glucose-lowering drugs      1.0%
]
N_MED = len(MED_LABEL_ORDER)  # 55


TASK_INFO = {
    # ---- Binary tasks: legacy, share mimic_downstream.pkl ----
    # `time_horizon_days`: prediction lookahead in days. 0 = in-stay outcome
    # (mortality / length-of-stay are about the current admission, not a
    # future event window). Used by mas_search.py task-similarity fallback
    # when the target task isn't an exact match in the source hospital.
    "death": {
        "type": "binary",
        "num_classes": 2,
        "data_pkl": "mimic_downstream.pkl",
        "label_col": None,        # encoded inline in dataset.py via row["DEATH"]
        "filter_col": None,
        "time_horizon_days": 0,
    },
    "stay": {
        "type": "binary",
        "num_classes": 2,
        "data_pkl": "mimic_downstream.pkl",
        "label_col": None,
        "filter_col": None,
        "time_horizon_days": 0,
    },
    "readmission": {
        "type": "binary",
        "num_classes": 2,
        "data_pkl": "mimic_downstream.pkl",
        "label_col": None,
        "filter_col": None,
        "time_horizon_days": 90,    # readmission within 3 months
    },
    # ---- Multilabel tasks: 18-class phenotype prediction ----
    "next_diag_6m_pheno": {
        "type": "multilabel",
        "num_classes": N_PHENO,
        "data_pkl": "mimic_nextdiag_6m.pkl",
        "label_col": "NEXT_DIAG_6M_PHENO",      # length-18 binary list per row
        "filter_col": "NEXT_DIAG_6M",           # rows with NaN here are dropped
        "time_horizon_days": 180,
    },
    "next_diag_12m_pheno": {
        "type": "multilabel",
        "num_classes": N_PHENO,
        "data_pkl": "mimic_nextdiag_12m.pkl",
        "label_col": "NEXT_DIAG_12M_PHENO",
        "filter_col": "NEXT_DIAG_12M",
        "time_horizon_days": 365,
    },
    # ---- Multilabel task: 55-class drug RECOMMENDATION (standard same-visit) ----
    # Recommend the CURRENT admission's medications given that admission's
    # DIAGNOSES + the patient's history (the GAMENet/SafeDrug formulation). NOT a
    # forecasting task — no time horizon. `target_visit_diag_only` tells
    # dataset.py to feed ONLY the target (last) visit's diagnoses and drop its
    # med (=label) / lab / procedure tokens; history visits stay full.
    "med_rec": {
        "type": "multilabel",
        "num_classes": N_MED,                 # 55
        "data_pkl": "mimic_med_rec.pkl",
        "label_col": "CUR_MED_ATC",           # length-55 binary list per row
        "filter_col": "CUR_MED_FILTER",       # NaN when the visit has 0 vocab drugs
        "time_horizon_days": 0,               # same-visit recommendation
        "target_visit_diag_only": True,       # input masking, handled in dataset.py
    },
}

ALL_TASKS = list(TASK_INFO.keys())
BINARY_TASKS = [t for t, info in TASK_INFO.items() if info["type"] == "binary"]
MULTILABEL_TASKS = [t for t, info in TASK_INFO.items() if info["type"] == "multilabel"]


def task_info(task: str) -> dict:
    """Return the registry entry for a task. Raises if unknown."""
    if task not in TASK_INFO:
        raise ValueError(
            f"Unknown task '{task}'. Available: {ALL_TASKS}"
        )
    return TASK_INFO[task]


def is_multilabel(task: str) -> bool:
    return task_info(task)["type"] == "multilabel"


def task_num_classes(task: str) -> int:
    return task_info(task)["num_classes"]


def task_time_horizon(task: str) -> int:
    """Prediction lookahead in days. 0 means in-stay outcome."""
    return task_info(task)["time_horizon_days"]


def task_data_pkl_path(hospital: str, task: str, data_root: str = "./data_process") -> Path:
    """Resolve the pkl path for a given hospital + task."""
    pkl_name = task_info(task)["data_pkl"]
    return Path(f"{data_root}/{hospital}/{hospital}-processed/{pkl_name}")
