#!/usr/bin/env python
"""Symlink the collaborator's OneFlorida drug-rec pkls into the paths the code expects.

The med_rec label was annotated by the collaborator and lands, per OneFlorida
site, under a shared read-only tree with hash-suffixed directory names:

    <SPLITS>/<site>_<hash>/<site>_<hash>_drug.pkl

but the model loads a task's pkl from a fixed location:

    get_processed_root(<site>) / "mimic_med_rec.pkl"
    = /blue/mei.liu/lideyi/MAS-NAS/data_process/<site>/<site>-processed/mimic_med_rec.pkl

Verified schema of the drug.pkl: a [train, val, test] list of DataFrames, each
carrying CUR_MED_ATC (length-55) + CUR_MED_FILTER and the ICD9/NDC/LAB/PRO input
columns — exactly what utils/dataset.py reads for task=med_rec. So no conversion
is needed; a symlink is the whole bridge. Nothing is copied and nothing under
results/ is touched.

SAFETY:
  * Dry-run by default — prints the plan, writes nothing. Pass --apply to link.
  * Refuses to overwrite a REAL file at the target (only creates a missing link
    or updates an existing symlink), so a hand-built mimic_med_rec.pkl is never
    clobbered.
  * Asserts exactly one <site>_*/ dir and one *_drug.pkl per site, so an
    ambiguous glob is a hard error rather than a silent wrong link.

MIMIC-IV is intentionally NOT handled here — its med_rec pkl comes from
build_med_rec.py, not the OneFlorida share.

Usage (server):
    cd /home/lideyi/MAS-NAS
    python data_process/link_onefl_medrec.py            # dry-run: show the plan
    python data_process/link_onefl_medrec.py --apply    # create the symlinks
    python data_process/link_onefl_medrec.py --sites source_1 source_4 source_14 source_16 source_15
"""
import argparse
import glob
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from utils.paths import get_processed_root  # noqa: E402

DEFAULT_SPLITS = "/blue/mei.liu/data_shared/OneFlorida_processed/MA_NAS/onefl_processed/splits"
# OneFlorida sites in the study (prior pool + internal target). source_6/9/12
# were dropped during cohort selection; MIMIC-IV uses build_med_rec.py instead.
DEFAULT_SITES = ["source_1", "source_3", "source_4", "source_10",
                 "source_11", "source_14", "source_16", "source_15"]
LINK_NAME = "mimic_med_rec.pkl"


def resolve_drug_pkl(splits: str, site: str) -> Path:
    """Find the single <site>_<hash>/<site>_<hash>_drug.pkl, or raise."""
    dirs = sorted(glob.glob(os.path.join(splits, f"{site}_*")))
    # Guard against source_1 matching source_10/11/... — require an underscore+hex hash.
    dirs = [d for d in dirs if os.path.basename(d).rsplit("_", 1)[0] == site]
    if len(dirs) != 1:
        raise SystemExit(f"[{site}] expected exactly 1 split dir, found {len(dirs)}: {dirs}")
    pkls = sorted(glob.glob(os.path.join(dirs[0], "*_drug.pkl")))
    if len(pkls) != 1:
        raise SystemExit(f"[{site}] expected exactly 1 *_drug.pkl in {dirs[0]}, found {len(pkls)}: {pkls}")
    return Path(pkls[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default=DEFAULT_SPLITS)
    ap.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    ap.add_argument("--apply", action="store_true",
                    help="Actually create/update the symlinks (default: dry-run).")
    args = ap.parse_args()

    print(f"{'APPLY' if args.apply else 'DRY-RUN'}  splits={args.splits}\n")
    planned = skipped = 0
    for site in args.sites:
        src = resolve_drug_pkl(args.splits, site)
        dst = get_processed_root(site) / LINK_NAME   # leaf dir auto-created by get_processed_root

        if dst.exists() and not dst.is_symlink():
            print(f"  SKIP  {site}: target is a REAL file, not touching → {dst}")
            skipped += 1
            continue
        if dst.is_symlink() and os.readlink(dst) == str(src):
            print(f"  OK    {site}: already linked → {src.name}")
            continue

        action = "relink" if dst.is_symlink() else "link"
        print(f"  {action.upper():6s}{site}: {dst}")
        print(f"        -> {src}")
        planned += 1
        if args.apply:
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

    print(f"\n{'created/updated' if args.apply else 'would create/update'}: {planned}   skipped(real file): {skipped}")
    if not args.apply and planned:
        print("Re-run with --apply to create the symlinks.")


if __name__ == "__main__":
    main()
