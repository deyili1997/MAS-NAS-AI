"""Centralized data path resolution for dual-environment work
(local Mac vs HiPerGator server).

Use case: on HiPerGator, the repo lives in /home/lideyi/MAS-NAS (10 GB
home-quota limit) while processed data sits in /blue/mei.liu/lideyi/MAS-NAS/...
(TB-scale group quota). Repo root != data root, so the old relative-path
convention `Path(f"./data_process/{H}/{H}-processed")` resolves to the
wrong location on HPC.

Resolution order for processed root:
    1. $MIMIC_PROCESSED_DIR (if set, only honored when hospital == "MIMIC-IV")
    2. /blue/mei.liu/lideyi/MAS-NAS/data_process/<H>/<H>-processed (HiPerGator)
    3. <project_root>/data_process/<H>/<H>-processed (local Mac fallback)

Detection: a candidate is "live" iff its parent dir exists. The leaf is
auto-mkdir'd when missing (so the data-prep notebook can write into it
on first run).
"""
from __future__ import annotations
import os
from pathlib import Path

# This file lives at <project_root>/utils/paths.py — go up two levels.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def get_processed_root(hospital: str) -> Path:
    """Resolve the absolute path of <hospital>-processed/ across local/HPC.

    Args:
        hospital: e.g. "MIMIC-IV". Used to construct directory names and
                  pick hospital-specific HPC overrides.

    Returns:
        Absolute Path to the <hospital>-processed/ directory.
        The leaf directory is mkdir-ed if missing (parent must already exist).

    Raises:
        FileNotFoundError if no candidate's parent directory exists. This
        means the project layout doesn't have data_process/<hospital>/ at
        all — likely a wrong --hospital flag or a fresh clone with no data.
    """
    candidates: list[Path] = []

    # --- Hospital-specific HPC overrides ---
    if hospital == "MIMIC-IV":
        env_override = os.environ.get("MIMIC_PROCESSED_DIR")
        if env_override:
            candidates.append(Path(env_override))
        candidates.append(
            Path("/blue/mei.liu/lideyi/MAS-NAS/data_process/MIMIC-IV/MIMIC-IV-processed")
        )

    # --- Generic project-relative fallback (works on Mac and on any host
    #     where the repo and data tree live together) ---
    candidates.append(_PROJECT_ROOT / "data_process" / hospital / f"{hospital}-processed")

    for cand in candidates:
        if cand.parent.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand

    raise FileNotFoundError(
        f"Cannot resolve processed dir for hospital={hospital!r}; "
        f"tried (in order): {[str(c) for c in candidates]}"
    )
