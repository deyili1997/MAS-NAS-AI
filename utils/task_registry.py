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
