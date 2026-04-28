"""Centralized data path resolution for dual-environment work
(local Mac vs HiPerGator server).

Use case: on HiPerGator, the repo lives in /home/lideyi/MAS-NAS (10 GB
home-quota limit) while bulky artifacts (processed data pkls, model
checkpoints) sit in /blue/mei.liu/lideyi/MAS-NAS/... (TB-scale group
quota). Repo root != big-output root, so the naive relative-path
convention resolves to the wrong location on HPC.

Resolution policy (per-function): if `_is_hpc()` returns True (i.e. the
HPC project root /blue/mei.liu/lideyi/MAS-NAS/ is reachable), large
artifacts route there; otherwise they fall back to the local repo tree.
Per-function env var overrides are honored when set.

Functions:
    get_processed_root(hospital): processed pkl dir for a hospital.
    get_checkpoint_dir(hospital):  pretrain .pt checkpoint dir.

Small outputs (CSVs / JSONs / PNGs / agent_io_log.txt) are NOT routed
through this module — they stay in the repo at results/<hospital>/...
where git can track them.
"""
from __future__ import annotations
import os
from pathlib import Path

# This file lives at <project_root>/utils/paths.py — go up two levels.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# HiPerGator project root (only exists when running on UF HPC; the
# directory's existence is our portable "am I on HPC?" probe).
_HPC_PROJECT_ROOT: Path = Path("/blue/mei.liu/lideyi/MAS-NAS")


def _is_hpc() -> bool:
    """True iff we're running on HiPerGator with the Mei Liu lab
    project space mounted. Used as the routing predicate for big
    outputs (processed data, checkpoints)."""
    return _HPC_PROJECT_ROOT.exists()


def get_processed_root(hospital: str) -> Path:
    """Resolve the absolute path of <hospital>-processed/ across local/HPC.

    On HPC: /blue/mei.liu/lideyi/MAS-NAS/data_process/<H>/<H>-processed/
    On Mac: <project_root>/data_process/<H>/<H>-processed/
    Override: $MIMIC_PROCESSED_DIR (only when hospital == "MIMIC-IV").

    The leaf dir is auto-mkdir'd if missing.
    """
    if hospital == "MIMIC-IV":
        env_override = os.environ.get("MIMIC_PROCESSED_DIR")
        if env_override:
            p = Path(env_override)
            p.mkdir(parents=True, exist_ok=True)
            return p

    base = _HPC_PROJECT_ROOT if _is_hpc() else _PROJECT_ROOT
    p = base / "data_process" / hospital / f"{hospital}-processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_checkpoint_dir(hospital: str) -> Path:
    """Resolve where the pretrain MLM checkpoint (.pt, ~100-300 MB) lives.

    On HPC: /blue/mei.liu/lideyi/MAS-NAS/results/<H>/checkpoint_mlm/
    On Mac: <project_root>/results/<H>/checkpoint_mlm/
    Override: $MIMIC_CHECKPOINT_DIR (only when hospital == "MIMIC-IV").

    Big files only — small outputs (CSVs / JSONs / PNGs) keep using the
    repo-local results/ dir so git can track them.

    The leaf dir is auto-mkdir'd if missing.
    """
    if hospital == "MIMIC-IV":
        env_override = os.environ.get("MIMIC_CHECKPOINT_DIR")
        if env_override:
            p = Path(env_override)
            p.mkdir(parents=True, exist_ok=True)
            return p

    base = _HPC_PROJECT_ROOT if _is_hpc() else _PROJECT_ROOT
    p = base / "results" / hospital / "checkpoint_mlm"
    p.mkdir(parents=True, exist_ok=True)
    return p
