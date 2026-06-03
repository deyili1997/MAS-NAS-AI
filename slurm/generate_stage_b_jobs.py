"""Generate the Stage B job index TSV consumed by submit_stage_b.sbatch.

The array sbatch reads line $SLURM_ARRAY_TASK_ID of `stage_b_jobs.tsv` to
decode (seed, hospital, task, method) for that job. By keeping the index
table separate from the sbatch body we:

  (a) avoid hard-coding 270 cases into the sbatch
  (b) make it trivial to regenerate (e.g. drop a method, add a hospital)
  (c) make it trivial to re-submit only failed indices later — just look at
      the TSV to find which (seed, hospital, task, method) failed and pass
      those indices to `sbatch --array=<indices> submit_stage_b.sbatch`.

Default factorial design = 3 seeds × 3 hospitals × 5 tasks × 9 methods = 405.

Output:
    slurm/stage_b_jobs.tsv  (405 lines, 4 tab-separated columns, no header)
        col 1: seed         e.g. 123
        col 2: hospital     e.g. MIMIC-IV
        col 3: task         e.g. death
        col 4: method       e.g. mas_loto

Order: stable lexicographic (seed → hospital → task → method) so that
SLURM_ARRAY_TASK_ID is reproducible across regenerations.

Usage:
    python slurm/generate_stage_b_jobs.py
    # → writes slurm/stage_b_jobs.tsv

    python slurm/generate_stage_b_jobs.py --seeds 123 456 789 \\
        --hospitals source_15 MIMIC-IV \\
        --output slurm/stage_b_jobs.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Defaults match the agreed Stage B factorial design (see paper Methods §8).
# Edit these lists then re-run the script to regenerate the TSV.
DEFAULT_SEEDS = [111, 123, 456, 789, 999]
DEFAULT_HOSPITALS = ["source_15", "MIMIC-IV"]
DEFAULT_TASKS = [
    "death",
    "stay",
    "readmission",
    "next_diag_6m_pheno",
    "next_diag_12m_pheno",
]
# 9 methods: 4 MAS-NAS variants + 5 baselines. Order matches the case
# statement in run_baseline.sbatch / submit_stage_b.sbatch.
DEFAULT_METHODS = [
    "baseline0",
    "baseline1",
    "baseline2",
    "baseline3",
    "baseline4",
    "mas",
    "mas_layer1_only",
    "mas_loto",
    "mas_cold",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"List of seeds (default: {DEFAULT_SEEDS})")
    ap.add_argument("--hospitals", type=str, nargs="+", default=DEFAULT_HOSPITALS,
                    help=f"Target hospitals (default: {DEFAULT_HOSPITALS})")
    ap.add_argument("--tasks", type=str, nargs="+", default=DEFAULT_TASKS,
                    help=f"Downstream tasks (default: {DEFAULT_TASKS})")
    ap.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS,
                    help=f"Search methods (default: {DEFAULT_METHODS})")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "stage_b_jobs.tsv",
                    help="Output TSV path (default: slurm/stage_b_jobs.tsv)")
    args = ap.parse_args()

    n_total = len(args.seeds) * len(args.hospitals) * len(args.tasks) * len(args.methods)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for seed in args.seeds:
            for hospital in args.hospitals:
                for task in args.tasks:
                    for method in args.methods:
                        f.write(f"{seed}\t{hospital}\t{task}\t{method}\n")

    print(f"→ Wrote {n_total} jobs to {args.output}")
    print(f"  Factorial: {len(args.seeds)} seeds × {len(args.hospitals)} hospitals "
          f"× {len(args.tasks)} tasks × {len(args.methods)} methods = {n_total}")
    print(f"\nSubmit with:")
    print(f"  sbatch --array=1-{n_total}%10 slurm/submit_stage_b.sbatch")


if __name__ == "__main__":
    main()
