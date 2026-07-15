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

# Drug-recommendation label list — the TOP-20 most frequent ATC-4 codes, frozen
# in document-frequency order. Must match MED_LABEL_ORDER printed by
# build_med_vocab() in MIMIC-IV.ipynb's drug-rec-multilabel cell.
#
# Why top-20 rather than all 140 ATC-4 codes: the full vocabulary averages only
# ~2.4% prevalence (76 of 140 codes are <0.5%) — too sparse to learn. Top-20
# lifts mean prevalence to ~11.8% (range 4.4%–21.8%, comparable to the 18
# phenotypes' ~17%) while still retaining ~72% of all prescription records and
# leaving only ~8.7% of visits with an all-zero label.
MED_LABEL_ORDER = [
    "MED_B01A",  # antithrombotic agents                21.8%
    "MED_N02B",  # other analgesics / antipyretics      21.7%
    "MED_A01A",  # stomatological preparations          21.1%
    "MED_A06A",  # laxatives                            21.0%
    "MED_A02B",  # peptic ulcer / GORD drugs            17.7%
    "MED_A12C",  # other mineral supplements            16.4%
    "MED_C10A",  # lipid modifying agents               16.3%
    "MED_B05C",  # irrigating solutions                 12.2%
    "MED_A12A",  # calcium                              11.0%
    "MED_C07A",  # beta blocking agents                 10.1%
    "MED_N02A",  # opioids                               9.9%
    "MED_N06A",  # antidepressants                       9.3%
    "MED_A04A",  # antiemetics / antinauseants           7.2%
    "MED_C09A",  # ACE inhibitors, plain                 7.0%
    "MED_A02A",  # antacids                              6.3%
    "MED_H03A",  # thyroid preparations                  6.0%
    "MED_H04A",  # glycogenolytic hormones               5.9%
    "MED_A10A",  # insulins and analogues                5.7%
    "MED_C08C",  # selective calcium channel blockers    5.3%
    "MED_J01D",  # other beta-lactam antibacterials      4.4%
]
N_MED = len(MED_LABEL_ORDER)  # 20


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
    # ---- Multilabel tasks: 20-class drug recommendation (top-20 ATC-4) ----
    # Same patient pool AND same train/val/test split as the corresponding
    # next_diag_*_pheno task (the notebook re-uses RANDOM_SEED+2 / +3 on an
    # identical pool), so drug vs phenotype is a like-for-like comparison on the
    # same patients — only the label space differs (20 ATC-4 codes vs 18
    # phenotypes). Higher output dimensionality is the reason this task is
    # expected to be more architecture-sensitive.
    "next_med_6m": {
        "type": "multilabel",
        "num_classes": N_MED,
        "data_pkl": "mimic_nextmed_6m.pkl",
        "label_col": "NEXT_MED_6M_ATC",       # length-20 binary list per row
        "filter_col": "NEXT_MED_6M",          # rows with NaN here are dropped
        "time_horizon_days": 180,
    },
    "next_med_12m": {
        "type": "multilabel",
        "num_classes": N_MED,
        "data_pkl": "mimic_nextmed_12m.pkl",
        "label_col": "NEXT_MED_12M_ATC",
        "filter_col": "NEXT_MED_12M",
        "time_horizon_days": 365,
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
