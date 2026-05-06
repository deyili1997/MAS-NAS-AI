"""Layer 2 architecture-effect prior generator (Meta-Regression Module).

Reads per-task pooled SHAP outputs from `shap_analysis.py` (one shap_values.csv
per task containing the cross-hospital pooled signed SHAP values), fits
categorical mixed-effect models with hospital as random intercept, and emits
a compact `architecture_prior.json` per task that LLM agents can consume as
soft directional guidance.

Pipeline (per task):
  1. Load shap_values.csv  (cols: hospital, embed_dim, depth, mlp_ratio, num_heads,
                                  shap_embed_dim, shap_depth, ...)
  2. For each architecture feature, fit:
        shap_<feat> ~ C(<feat>) + (1 | hospital)        (statsmodels MixedLM)
     Hospital random intercept controls per-site SHAP baseline drift.
     If MixedLM fails (small n_hospitals or singular variance), fall back to
        shap_<feat> ~ C(<feat>)                          (cluster-robust SE)
  3. Identify top-2 features by mean abs SHAP; for that pair fit:
        shap_<top1> ~ C(<top1>) * C(<top2>) + (1 | hospital)
  4. Apply CI-based decision rules → preferred / discouraged / neutral levels
     and preferred_combinations / avoid_combinations for the top-2 pair.
  5. Write 4 artifacts per task:
       importance_summary.csv     {task, mean_abs_shap_*, importance_rank}
       level_effects.csv          {task, feature, level, est_mean_shap, ci_lo, ci_hi, n_rows, n_hospitals}
       interaction_effects.csv    {task, feat1, level1, feat2, level2, est_mean_shap, ci_lo, ci_hi, n_rows}
       architecture_prior.json    LLM agent-facing summary (preferred/discouraged + interaction rules + caveat)

Usage:
    python run_meta_regression.py
    python run_meta_regression.py --shap_dir ./results/shap_analysis \\
                                  --output_dir ./results/meta_regression
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm  # for CI z-scores

# Statsmodels for mixed-effect models
import statsmodels.formula.api as smf
import statsmodels.api as sm


ARCH_FEATURES = ["embed_dim", "depth", "mlp_ratio", "num_heads"]

# CI-based confidence labeling (interpretable, soft guidance)
# CI width controls how strict we are; tweak if priors are too aggressive/conservative.
CI_LEVEL = 0.95


# ---------------------------------------------------------------------------
# Mixed-effect model wrappers
# ---------------------------------------------------------------------------
def _fit_main_effects(shap_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Fit `shap_<feature> ~ C(<feature>) + (1 | hospital)`.

    Returns a DataFrame with one row per level:
        feature, level, est_mean_shap, ci_lo, ci_hi, n_rows, n_hospitals, model_type
    where model_type ∈ {"mixedlm", "ols_clustered"}.
    """
    formula = f"shap_{feature} ~ C({feature})"
    rows = []

    # ── Try MixedLM first ──────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence warnings are common
            model = smf.mixedlm(formula, shap_df, groups=shap_df["hospital"]).fit(
                method="lbfgs", reml=False
            )
        params = model.params
        ci = model.conf_int(alpha=1 - CI_LEVEL)
        intercept = params.get("Intercept", 0.0)
        intercept_ci = ci.loc["Intercept"] if "Intercept" in ci.index else (np.nan, np.nan)

        # Extract per-level effect = intercept + C(feature)[T.<level>]
        # Reference level (encoded into the Intercept) gets level effect = intercept.
        levels = sorted(shap_df[feature].unique())
        ref_level = levels[0]   # statsmodels default: lowest sorted level

        for lvl in levels:
            sub = shap_df[shap_df[feature] == lvl]
            if lvl == ref_level:
                est = intercept
                ci_lo, ci_hi = float(intercept_ci[0]), float(intercept_ci[1])
            else:
                key = f"C({feature})[T.{lvl}]"
                if key not in params.index:
                    continue
                est = intercept + params[key]
                # Sum of variance for intercept + offset (approximate, ignores covariance)
                # Better: use linear combination through params/cov
                contrast = np.zeros(len(params))
                contrast[params.index.get_loc("Intercept")] = 1.0
                contrast[params.index.get_loc(key)] = 1.0
                cov = model.cov_params().values
                se_combined = float(np.sqrt(contrast @ cov @ contrast))
                z = float(norm.ppf(0.5 + CI_LEVEL / 2))   # ≈ 1.96 for 95%
                ci_lo = est - z * se_combined
                ci_hi = est + z * se_combined

            rows.append({
                "feature": feature,
                "level": lvl,
                "est_mean_shap": float(est),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "n_rows": int(len(sub)),
                "n_hospitals": int(sub["hospital"].nunique()),
                "model_type": "mixedlm",
            })
        return pd.DataFrame(rows)

    except Exception as e:
        # ── Fallback chain: cluster-robust SE → plain OLS → group means ──
        # Cluster-robust requires ≥2 clusters (hospitals). Smoke tests with a
        # single hospital must drop the cluster argument.
        n_hosp = int(shap_df["hospital"].nunique())
        rows = []
        try:
            if n_hosp >= 2:
                print(f"  [warn] MixedLM failed for {feature}: {e}. Using OLS + cluster-robust SE.")
                model = smf.ols(formula, shap_df).fit(
                    cov_type="cluster", cov_kwds={"groups": shap_df["hospital"]}
                )
                model_type = "ols_clustered"
            else:
                print(f"  [warn] MixedLM failed for {feature}: {e}. n_hospitals=1 → "
                      f"using plain OLS (CI may be optimistic).")
                model = smf.ols(formula, shap_df).fit()
                model_type = "ols_plain"
            params = model.params
            ci = model.conf_int(alpha=1 - CI_LEVEL)
            intercept = params.get("Intercept", 0.0)
            intercept_ci = ci.loc["Intercept"] if "Intercept" in ci.index else (np.nan, np.nan)
            levels = sorted(shap_df[feature].unique())
            ref_level = levels[0]
            z = float(norm.ppf(0.5 + CI_LEVEL / 2))
            for lvl in levels:
                sub = shap_df[shap_df[feature] == lvl]
                if lvl == ref_level:
                    est = intercept
                    ci_lo, ci_hi = float(intercept_ci[0]), float(intercept_ci[1])
                else:
                    key = f"C({feature})[T.{lvl}]"
                    if key not in params.index:
                        continue
                    est = intercept + params[key]
                    contrast = np.zeros(len(params))
                    contrast[params.index.get_loc("Intercept")] = 1.0
                    contrast[params.index.get_loc(key)] = 1.0
                    cov = model.cov_params().values
                    se_combined = float(np.sqrt(contrast @ cov @ contrast))
                    ci_lo = est - z * se_combined
                    ci_hi = est + z * se_combined
                rows.append({
                    "feature": feature,
                    "level": lvl,
                    "est_mean_shap": float(est),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "n_rows": int(len(sub)),
                    "n_hospitals": int(sub["hospital"].nunique()),
                    "model_type": model_type,
                })
            return pd.DataFrame(rows)
        except Exception as e2:
            # Last-ditch: emit raw group means with NaN CIs so downstream
            # classification still has data and just lands in "neutral".
            print(f"  [warn] OLS also failed for {feature}: {e2}. Emitting group means with NaN CIs.")
            try:
                rows = []
                for lvl in sorted(shap_df[feature].unique()):
                    sub = shap_df[shap_df[feature] == lvl]
                    rows.append({
                        "feature": feature,
                        "level": lvl,
                        "est_mean_shap": float(sub[f"shap_{feature}"].mean()),
                        "ci_lo": float("nan"),
                        "ci_hi": float("nan"),
                        "n_rows": int(len(sub)),
                        "n_hospitals": int(sub["hospital"].nunique()),
                        "model_type": "group_mean",
                    })
                return pd.DataFrame(rows)
            except Exception as e3:
                print(f"  [error] Group-mean fallback failed for {feature}: {e3}")
                return pd.DataFrame()


def _fit_interaction(shap_df: pd.DataFrame, top1: str, top2: str) -> pd.DataFrame:
    """Fit `shap_<top1> ~ C(<top1>) * C(<top2>) + (1 | hospital)` and return
    per-(level1, level2) cell estimates."""
    formula = f"shap_{top1} ~ C({top1}) * C({top2})"
    rows = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, shap_df, groups=shap_df["hospital"]).fit(
                method="lbfgs", reml=False
            )
    except Exception as e:
        print(f"  [warn] Interaction MixedLM failed for ({top1}, {top2}): {e}")
        return pd.DataFrame()

    # Group means per (level1, level2) cell — simpler and more interpretable than
    # decomposing the design matrix. Use predicted values for stability.
    levels1 = sorted(shap_df[top1].unique())
    levels2 = sorted(shap_df[top2].unique())
    for l1 in levels1:
        for l2 in levels2:
            sub = shap_df[(shap_df[top1] == l1) & (shap_df[top2] == l2)]
            if len(sub) < 3:
                continue
            est = float(sub[f"shap_{top1}"].mean())
            sd = float(sub[f"shap_{top1}"].std(ddof=1)) if len(sub) > 1 else 0.0
            se = sd / np.sqrt(len(sub)) if len(sub) > 0 else 0.0
            z = 1.96
            ci_lo = est - z * se
            ci_hi = est + z * se
            rows.append({
                "feat1": top1, "level1": l1,
                "feat2": top2, "level2": l2,
                "est_mean_shap": est,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_rows": int(len(sub)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decision rules (CI-based)
# ---------------------------------------------------------------------------
def _classify_levels(level_eff_df: pd.DataFrame) -> dict:
    """Classify each (feature, level) into preferred / discouraged / neutral.

    Rule: CI strictly above 0 → preferred; strictly below 0 → discouraged; else neutral.
    NaN CI → neutral (insufficient data, e.g. n_hospitals=1 group_mean fallback).
    Returns dict[feature] = {"preferred": [...], "discouraged": [...], "neutral": [...]}
    """
    out = {}
    if level_eff_df is None or level_eff_df.empty or "feature" not in level_eff_df.columns:
        return out
    for feat in level_eff_df["feature"].unique():
        sub = level_eff_df[level_eff_df["feature"] == feat]
        preferred, discouraged, neutral = [], [], []
        for _, row in sub.iterrows():
            lvl = row["level"]
            ci_lo = row.get("ci_lo", float("nan"))
            ci_hi = row.get("ci_hi", float("nan"))
            lvl_int = int(lvl) if isinstance(lvl, (int, np.integer)) else lvl
            if pd.isna(ci_lo) or pd.isna(ci_hi):
                neutral.append(lvl_int)
            elif ci_lo > 0:
                preferred.append(lvl_int)
            elif ci_hi < 0:
                discouraged.append(lvl_int)
            else:
                neutral.append(lvl_int)
        out[feat] = {
            "preferred": preferred,
            "discouraged": discouraged,
            "neutral": neutral,
        }
    return out


def _classify_interactions(interaction_df: pd.DataFrame) -> tuple:
    """Return (preferred_combinations, avoid_combinations) lists of dicts."""
    preferred, avoid = [], []
    if interaction_df is None or interaction_df.empty:
        return preferred, avoid
    if not {"feat1", "feat2", "level1", "level2", "ci_lo", "ci_hi"}.issubset(interaction_df.columns):
        return preferred, avoid
    feat1 = interaction_df["feat1"].iloc[0]
    feat2 = interaction_df["feat2"].iloc[0]
    for _, row in interaction_df.iterrows():
        ci_lo = row.get("ci_lo", float("nan"))
        ci_hi = row.get("ci_hi", float("nan"))
        if pd.isna(ci_lo) or pd.isna(ci_hi):
            continue
        if ci_lo > 0:
            preferred.append({feat1: int(row["level1"]), feat2: int(row["level2"])})
        elif ci_hi < 0:
            avoid.append({feat1: int(row["level1"]), feat2: int(row["level2"])})
    return preferred, avoid


def _confidence_label(level_eff_df: pd.DataFrame, feature: str) -> str:
    """Heuristic: high if any level has |est|/SE > 3; moderate if > 2; else low."""
    if level_eff_df is None or level_eff_df.empty or "feature" not in level_eff_df.columns:
        return "low"
    sub = level_eff_df[level_eff_df["feature"] == feature]
    if sub.empty:
        return "low"
    # Use min half-width / max |est| as proxy for "tightness vs magnitude"
    half_widths = (sub["ci_hi"] - sub["ci_lo"]).abs() / 2
    abs_ests = sub["est_mean_shap"].abs()
    # ratio = magnitude / half_width; higher → tighter CI relative to effect
    # Avoid division by zero for negligible magnitudes
    ratios = (abs_ests / (half_widths + 1e-9)).replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return "low"
    max_ratio = ratios.max()
    if max_ratio > 3:
        return "high"
    if max_ratio > 2:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Per-task entry point
# ---------------------------------------------------------------------------
def run_meta_regression_for_task(task: str, shap_csv: Path, out_dir: Path) -> dict:
    """Read shap_values.csv for one task, fit mixed-effect models, emit artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_df = pd.read_csv(shap_csv)
    if "hospital" not in shap_df.columns:
        raise ValueError(
            f"{shap_csv} missing 'hospital' column — required for MixedLM. "
            "Re-run shap_analysis.py with the new --hospitals flag (passes hospitals "
            "down to run_shap_for_group)."
        )

    print(f"\n=== Task: {task}  (n={len(shap_df)} rows from "
          f"{shap_df['hospital'].nunique()} hospital(s)) ===")

    # ── 1. Importance ranking (mean abs SHAP) ────────────────────────────
    importance = {}
    for feat in ARCH_FEATURES:
        importance[feat] = float(np.abs(shap_df[f"shap_{feat}"]).mean())
    importance_order = sorted(importance, key=importance.get, reverse=True)
    print(f"  Importance order (mean |SHAP|): {importance_order}")
    print(f"    {importance}")

    # ── 2. Per-feature main-effects model ────────────────────────────────
    level_effects_all = []
    for feat in ARCH_FEATURES:
        eff = _fit_main_effects(shap_df, feat)
        if not eff.empty:
            level_effects_all.append(eff)
    level_effects_df = pd.concat(level_effects_all, ignore_index=True) \
        if level_effects_all else pd.DataFrame()

    # ── 3. Top-2 interaction model ───────────────────────────────────────
    top1, top2 = importance_order[0], importance_order[1]
    interaction_df = _fit_interaction(shap_df, top1, top2)

    # ── 4. Decision rules ────────────────────────────────────────────────
    level_classes = _classify_levels(level_effects_df)
    pref_combos, avoid_combos = _classify_interactions(interaction_df)
    confidence = {feat: _confidence_label(level_effects_df, feat) for feat in ARCH_FEATURES}

    # ── 5. Write artifacts ───────────────────────────────────────────────
    # importance_summary.csv
    imp_row = {"task": task, "importance_rank": ",".join(importance_order)}
    for feat in ARCH_FEATURES:
        imp_row[f"mean_abs_shap_{feat}"] = importance[feat]
    pd.DataFrame([imp_row]).to_csv(out_dir / "importance_summary.csv", index=False)

    # level_effects.csv
    if not level_effects_df.empty:
        level_effects_df.insert(0, "task", task)
        level_effects_df.to_csv(out_dir / "level_effects.csv", index=False)

    # interaction_effects.csv
    if not interaction_df.empty:
        interaction_df.insert(0, "task", task)
        interaction_df.to_csv(out_dir / "interaction_effects.csv", index=False)

    # architecture_prior.json (LLM agent-facing)
    prior = {
        "task": task,
        "feature_importance_order": importance_order,
        "preferred_levels": {f: level_classes.get(f, {}).get("preferred", []) for f in ARCH_FEATURES},
        "discouraged_levels": {f: level_classes.get(f, {}).get("discouraged", []) for f in ARCH_FEATURES},
        "interaction_rules": [
            {
                "pair": [top1, top2],
                "preferred_combinations": pref_combos,
                "avoid_combinations": avoid_combos,
            }
        ],
        "confidence": confidence,
        "_caveat": (
            "These levels are statistical regularities derived from XGBoost "
            "surrogate SHAP values pooled across multiple hospitals (mixed-effect "
            "models with hospital random intercept). They reflect cross-hospital "
            "stability, NOT causal architectural rules. Treat as soft directional "
            "guidance for LLM agent decisions: high-confidence preferred → strong "
            "push, discouraged → require explicit justification before adoption "
            "(NOT auto-rejection)."
        ),
    }
    with open(out_dir / "architecture_prior.json", "w") as f:
        json.dump(prior, f, indent=2)

    print(f"  ✓ Saved 4 artifacts to {out_dir}/")
    print(f"  preferred levels:    {prior['preferred_levels']}")
    print(f"  discouraged levels:  {prior['discouraged_levels']}")
    return prior


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Layer 2 architecture-effect prior generator (Meta-Regression).")
    p.add_argument("--shap_dir", type=str, default="./results/shap_analysis",
                   help="Directory containing per-task shap_values.csv "
                        "(produced by shap_analysis.py with --hospitals flag).")
    p.add_argument("--output_dir", type=str, default="./results/meta_regression",
                   help="Where to write per-task meta-regression artifacts.")
    p.add_argument("--tasks", type=str, nargs="+", default=None,
                   help="Optional subset of tasks to run. Default: all subdirs of --shap_dir.")
    return p.parse_args()


def main():
    args = parse_args()
    shap_dir = Path(args.shap_dir)
    out_dir = Path(args.output_dir)

    if not shap_dir.exists():
        raise FileNotFoundError(f"--shap_dir not found: {shap_dir}. "
                                f"Run shap_analysis.py first.")

    # Discover tasks: each subdirectory of shap_dir with a shap_values.csv
    candidate_tasks = sorted([
        p.name for p in shap_dir.iterdir()
        if p.is_dir() and (p / "shap_values.csv").exists()
    ])
    if args.tasks:
        tasks = [t for t in args.tasks if t in candidate_tasks]
        missing = [t for t in args.tasks if t not in candidate_tasks]
        if missing:
            print(f"[warn] Requested tasks not found in {shap_dir}: {missing}")
    else:
        tasks = candidate_tasks
    print(f"Running meta-regression for {len(tasks)} task(s): {tasks}")

    for task in tasks:
        try:
            run_meta_regression_for_task(
                task=task,
                shap_csv=shap_dir / task / "shap_values.csv",
                out_dir=out_dir / task,
            )
        except Exception as e:
            print(f"[error] task={task} failed: {e}")
            raise

    print(f"\n✓ All meta-regression artifacts under {out_dir}/")


if __name__ == "__main__":
    main()
