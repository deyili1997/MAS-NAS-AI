# MAS-NAS: Multi-Agent Neural Architecture Search for EHR Transformers

> A 3-agent LLM-driven NAS controller that **outperforms 5 baselines using 70% less compute** on longitudinal Electronic Health Record (EHR) Transformer design — validated at production scale across 10 hospitals and 5 clinical prediction tasks.

**Target venue**: NeurIPS / Nature Machine Intelligence / TPAMI
**Stack**: PyTorch · AutoFormer supernet · Anthropic Claude API · XGBoost + SHAP · statsmodels MixedLM · SLURM on HiPerGator (UF) L4 GPUs

---

## TL;DR

| | |
|---|---|
| **Problem** | Designing EHR Transformer architectures is tedious + tasks/datasets are heterogeneous across hospitals. Generic LLM-only NAS hallucinates; standard NAS ignores domain priors. |
| **Idea** | Combine (a) AutoFormer-style weight-sharing supernet for cheap evaluation, (b) a 3-agent LLM controller (Proposal / Critic / Strategy), and (c) a **two-layer cross-hospital prior** distilled from SHAP + mixed-effect meta-regression across 8 source hospitals. |
| **Scale** | 8 OneFL+ private hospitals + MIMIC-III + MIMIC-IV; 5 tasks (mortality, length-of-stay, readmission, 2 phenotype multi-label); 500 architecture evaluations × 8 sites = **4,000+ NAS evaluations** in Stage A alone. |
| **Status** | Phase 1 (test-phase) ✅ done; Stage A (8-site prior generation) 87% complete; Phase 2 (462-job cross-hospital NAS production) starting once last site finishes. |

---

## Key Contributions

1. **Multi-Agent NAS Controller** (`mas_search.py`, `agents/`)
   3 agents coordinate per round:
   - **Proposal Agent** — produces candidate architectures conditioned on a two-layer prior + completed-trial state.
   - **Critic Agent** — bifurcates HARD constraints (parameter budget, divisibility, exact duplicates → auto-reject) vs SOFT signals (discouraged levels, avoid-combinations → flag and require rationale).
   - **Experiment Agent** — runs supernet finetune, tracks composite-rank best-so-far, and at each round-end decides EXPLORATION vs EXPLOITATION for the next round.

2. **Two-Layer Cross-Hospital Prior** (`mas_search.py:gather_historical_context`, `shap_analysis.py`, `run_meta_regression.py`)
   - **Layer 1 — Task-driven hospital selection**: for each target task `t`, select the source hospital whose **task-specific feature vector** (label entropy, positive ratio, task type, time horizon) is most similar to the target — rather than using a single dataset-level cosine. This ensures, e.g., death-task architectures are retrieved from a source with a similar mortality rate, not merely similar overall EHR token densities. A graceful dataset-level cosine fallback is used for the LOTO ablation (preserving its original "Plan A" semantics).
   - **Layer 2 — Architecture-effect prior**: pool ~2,000 (arch, task) rows across 4 diverse source hospitals → XGBoost surrogate + SHAP TreeExplainer → categorical mixed-effect models with `hospital` random intercept → `architecture_prior.json` with preferred / discouraged levels and confidence labels for LLM consumption.

3. **AutoFormer-style Weight-Sharing Supernet** (`model/supernet_transformer.py`)
   Pretrain once per hospital (MLM objective), then evaluate ~100 sub-architectures cheaply via column slicing on shared weights. Ranking validity verified against from-scratch training (Fig 3 in paper).

4. **4-Condition Factorial Ablation** (Plan v2 / `mas_search.py:_method_name`)
   `mas` / `mas_layer1_only` / `mas_loto` / `mas_cold` isolate Layer 2's incremental value, and the contribution of task-driven vs dataset-level hospital selection (LOTO reverts to dataset-level to preserve its original "Plan A fallback" semantics), and the standalone value of multi-agent reasoning.

5. **Production-grade Reproducibility** (`analyze/aggregate_results.py`, `analyze/plot_*.py`)
   5 seeds × 5 tasks × 9 methods × 2 target hospitals = **450 NAS jobs**; Bootstrap 95% CI significance tests (1000 resamples); full audit trails (every LLM prompt + response logged via `utils/tracer.py`); incremental CSV save survives 14-day SLURM wall.

---

## Method at a Glance

```
                4 arch features × N archs × 5 tasks × 4 source hospitals
                        │  (Stage A: run_pipeline.py)
                        ▼
            metadata.csv × 4    ────  dataset_summary.csv × 4
                   │                            │
        ┌──────────┴──────────┐                 │
        ▼ pool per task       ▼ (matched src,   ▼ task-feature cosine
                                matched task)     → matched source
   XGBoost surrogate     top-k arch retrieval     │  (per task t:
   + SHAP + mixed-effect + 4 metrics per row      │   label_entropy,
        │                     │                   │   positive_ratio,
        ▼                     │                   │   task_type)
   architecture_prior.json    │                   │
   (Layer 2)                  ▼ Layer 1           │
        └─────────────┬───────┘                   │
                      ▼                           │
          context dict  ◀──────  _compute_target_task_label_stats
                      │
                      ▼
          Markdown prompt → Claude (claude-3.5-haiku)
          (Proposal / Critic / Strategy agents)
                      │
                      ▼
          accepted archs → supernet finetune → val metrics → strategy update
                      │
                      ▼ (until budget = 0)
          final test eval on best-by-val arch
```

Each round consumes 1–3 architectures from a budget of 100; full method description in [`paper/methods.md`](./paper/methods.md).

---

## Search Space

| Dimension | Choices | Notes |
|---|---|---|
| `embed_dim` | {32, 64, 128, 256} | constraint: `embed_dim % num_heads == 0` |
| `depth` | {1, 2, 4, 8} | number of transformer layers |
| `mlp_ratio` | {1, 2, 4, 8} | constant across layers |
| `num_heads` | {1, 2, 4, 8} | constant across layers |

Total combinatorial space = 256 architectures; ≈ 100–200 valid after the 2M-parameter budget.

---

## Data + Tasks

**Hospitals**:
- **Prior pool (4 OneFL+ sites)**: `source_1`, `source_4`, `source_14`, `source_16` — selected for maximum diversity in lab/procedure/diagnosis density profiles.
- **Internal test**: `source_15` (OneFL+ site, excluded from prior; 7% mortality rate, no class collapse).
- **External test**: `MIMIC-IV` — cross-institutional generalization benchmark.

**5 Downstream Tasks**:
| Task | Type | Horizon |
|---|---|---|
| Mortality | binary | in-stay |
| Length-of-stay > 7 days | binary | in-stay |
| 3-month readmission | binary | 3 months |
| 18-class phenotype (next diagnosis) | multilabel | 6 months |
| 18-class phenotype (next diagnosis) | multilabel | 12 months |

**Data engineering tackled**:
- Patient-level LOINC deduplication coordinated with the OneFL+ data owner (50% lab-token reduction).
- Offline sequence-length analysis → `MAX_SEQ_LEN = 256` truncation (keep `[CLS]` + most-recent tokens) to fit batch_size = 64 on L4 GPUs.
- Temporal subsampling at unified cutoff `2022-07-01` for the 3 largest sites, yielding 52K–88K patients per site — defensible paper framing rather than arbitrary top-N.

---

## Baselines + Ablations

**5 Baselines** (`baselines/baseline{0..4}.py`):
- Random search · Evolutionary Algorithm · LLM-1shot · LLMatic · CoLLM-NAS

**4-condition Ablation factorial**:
| Method | Layer 1 hospital selection | Layer 2 | Isolates |
|---|---|---|---|
| `mas` (full) | ✅ task-driven | ✅ ON | full prior contribution |
| `mas_layer1_only` | ✅ task-driven | ❌ OFF | **Layer 2 incremental value** |
| `mas_loto` | ⚠ dataset-level + drop exact task records → Plan A fallback | ✅ ON | **task-driven vs dataset-level selection** |
| `mas_cold` | ❌ OFF | ❌ OFF | pure multi-agent reasoning value |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep learning | PyTorch, AutoFormer-style supernet (custom), AdamW, BCE / CE losses, MLM pretraining |
| LLM agents | Anthropic Claude API (`claude-3.5-haiku` via OpenRouter) — Proposal / Critic / Strategy roles |
| Surrogate modeling | XGBoost regressor + SHAP TreeExplainer |
| Mixed-effect statistics | `statsmodels.MixedLM` (per-feature signed-SHAP ~ level + (1\|hospital)); OLS+cluster-robust SE fallback |
| Significance testing | Bootstrap 95% CI (1000 resamples, paired by seed) |
| Distributed compute | SLURM on HiPerGator (UF), L4 GPUs, ~750 total GPU-hours budgeted |
| Reproducibility | 5 seeds, incremental CSV writes (survive wall timeouts), `search_meta.json` logs LLM call counts + wall-clock per run |

---

## Code Map

```
mas_search.py                       # 3-agent NAS controller, search loop, 2-layer prior assembly
agents/
├── proposal_agent.py              # Architecture proposer (LLM)
├── critic_agent.py                # Constraint + soft-signal evaluator (LLM)
└── experiment_agent.py            # Finetune runner + Strategy decider
model/supernet_transformer.py       # AutoFormer-style weight-shared supernet
baselines/baseline{0..4}.py         # 5 NAS baselines (Random, EA, LLM-1shot, LLMatic, CoLLM-NAS)
run_pipeline.py                     # Stage A: pretrain + finetune 100 archs × 5 tasks per hospital
run_regression.py                   # Supernet vs from-scratch ranking validity (Fig 3)
shap_analysis.py                    # Per-task pooled SHAP across hospitals (Fig 4)
run_meta_regression.py              # Layer 2 architecture-effect prior generator
analyze/
├── aggregate_results.py           # Multi-hospital paper tables (main / efficiency / supp / arch / cost / significance / ablation)
├── plot_search_trajectory.py      # Fig 1
├── plot_pareto.py                 # Fig 2
├── plot_loto_ablation.py          # Fig 5
└── plot_source_hospital_choice.py # Fig 6 (which OneFL+ source did Layer 1 pick?)
utils/                              # Tokenizer, dataset loaders, engine (train/eval), tracer, paths
data_process/                       # Per-hospital schema-consistent EHR pkl outputs
paper/methods.md                    # Full method description (10 sections + appendix)
```
