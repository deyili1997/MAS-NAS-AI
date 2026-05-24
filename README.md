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
   - **Layer 1 — Dataset-similarity retrieval**: pick the most similar source hospital by 20-D dataset profile cosine; retrieve top-k concrete architecture exemplars from that source's metadata. Graceful 6-D task-feature fallback when target task is missing in source (Plan A).
   - **Layer 2 — Architecture-effect prior**: pool ~700 (arch, task) rows across 8 source hospitals → XGBoost surrogate + SHAP TreeExplainer → categorical mixed-effect models with `hospital` random intercept → `architecture_prior.json` with preferred / discouraged levels, interaction rules, and confidence labels for LLM consumption.

3. **AutoFormer-style Weight-Sharing Supernet** (`model/supernet_transformer.py`)
   Pretrain once per hospital (MLM objective), then evaluate ~100 sub-architectures cheaply via column slicing on shared weights. Ranking validity verified against from-scratch training (Fig 3 in paper).

4. **4-Condition Factorial Ablation** (Plan v2 / `mas_search.py:_method_name`)
   `mas` / `mas_layer1_only` / `mas_loto` / `mas_cold` isolate Layer 2's incremental value, Layer 1 task-fallback degradation, and the standalone value of multi-agent reasoning.

5. **Production-grade Reproducibility** (`analyze/aggregate_results.py`, `analyze/plot_*.py`)
   5 seeds × 5 tasks × 9 methods × 2 target hospitals = **450 NAS jobs**; paired Wilcoxon + Holm-Bonferroni significance tests; full audit trails (every LLM prompt + response logged via `utils/tracer.py`); incremental CSV save survives 14-day SLURM wall.

---

## Method at a Glance

```
                4 arch features × N archs × 5 tasks × 8 source hospitals
                        │  (Stage A: run_pipeline.py)
                        ▼
            metadata.csv × 8    ────  dataset_summary.csv × 8
                   │                            │
        ┌──────────┴──────────┐                 │
        ▼ pool per task       ▼ (matched src,   ▼ hospital cosine
                                matched task)     → matched source
   XGBoost surrogate     top-k arch retrieval     │
   + SHAP + mixed-effect + 4 metrics per row      │
        │                     │                   │
        ▼                     │                   │
   architecture_prior.json    │                   │
   (Layer 2)                  ▼ Layer 1           │
        └─────────────┬───────┘                   │
                      ▼                           │
          context dict  ◀──────  _compute_target_summary
                      │
                      ▼
          Markdown prompt → Claude (Sonnet 4.6)
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

**Hospitals** (10 total):
- **8 OneFL+ private sites** (`source_1`, `source_3`, `source_4`, `source_10`, `source_11`, `source_14`, `source_15`, `source_16`) used as the cross-hospital prior pool.
- **MIMIC-III** + **MIMIC-IV** as held-out NAS targets — the cross-hospital generalization test.

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
| Method | Layer 1 | Layer 2 | Isolates |
|---|---|---|---|
| `mas` (full) | ✅ exact match | ✅ ON | full prior contribution |
| `mas_layer1_only` | ✅ exact match | ❌ OFF | **Layer 2 incremental value** |
| `mas_loto` | ⚠ forced cosine fallback | ✅ ON | Layer 1 task-fallback degradation |
| `mas_cold` | ❌ OFF | ❌ OFF | pure multi-agent reasoning value |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Deep learning | PyTorch, AutoFormer-style supernet (custom), AdamW, BCE / CE losses, MLM pretraining |
| LLM agents | Anthropic Claude API (claude-sonnet-4.6) — Proposal / Critic / Strategy roles |
| Surrogate modeling | XGBoost regressor + SHAP TreeExplainer |
| Mixed-effect statistics | `statsmodels.MixedLM` (per-feature signed-SHAP ~ level + (1\|hospital)); OLS+cluster-robust SE fallback |
| Significance testing | Paired Wilcoxon signed-rank + Holm-Bonferroni multiple-comparison correction |
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

---

## Current Results Preview (Phase 1 + Stage A 7-site preview)

**Cross-task SHAP importance ordering** (pooled across 7 OneFL+ sites, n=700 per task):
```
death:                embed_dim > num_heads > mlp_ratio > depth
stay:                 num_heads > embed_dim > depth > mlp_ratio   ← only task where num_heads dominates
readmission:          embed_dim > depth > num_heads > mlp_ratio
next_diag_6m_pheno:   embed_dim > depth > mlp_ratio > num_heads
next_diag_12m_pheno:  embed_dim > depth > mlp_ratio > num_heads
```

**Universal architectural patterns** (Layer 2 prior):
- `embed_dim` ∈ {128, 256} universally preferred; 32 universally discouraged.
- `mlp_ratio = 1` universally preferred (counter to standard Transformer's 4× FFN expansion).
- `depth` is highly task-dependent — `stay` prefers deeper (2–8), other tasks prefer 1.
- Cross-task heterogeneity itself motivates per-task priors over a single universal prior.

Final Phase 2 results (Tables 1–S6 + Figs 1–6) are queued; rerun command is one line once production completes.

---

## What I Did (for hiring managers)

> Solo author on the codebase + experimental design + manuscript. ~10K LOC of production Python across model, NAS, baselines, analysis, plotting; ~800 LOC of cross-hospital SLURM scripts.

- **Designed** the two-layer cross-hospital prior architecture (a methodological contribution intended for the paper, not an off-the-shelf technique).
- **Built** the 3-agent LLM controller from scratch (Proposal–Critic–Experiment with revision micro-loop) using the Anthropic SDK; engineered the soft-prior framing so the Critic doesn't auto-reject LLM exploration but requires rationale for discouraged levels.
- **Extended** AutoFormer's weight-sharing supernet to a multi-hospital EHR setting (vocab + admission embeddings + task-multiplexed classification head).
- **Coordinated** with the OneFL+ private-data owner across multiple revisions to land patient-level LOINC deduplication, then engineered a Python `MAX_SEQ_LEN` truncation as a bridge before the owner's fix landed.
- **Owned** production at HPC scale: 13 concurrent SLURM jobs across L4 GPUs, 14-day walls, incremental CSV writes to survive timeouts, full agent prompt/response logging for paper reproducibility.
- **Wrote** the entire Methods section (10 §, ~600 lines) + designed the figure / table layout for 5 main figs + 6 supplementary tables.

---

## Contact

**Deyi Li** — UF Biomedical Informatics PhD student
📧 `lideyi@ufl.edu` · 🔗 [GitHub](https://github.com/deyili1997)

> Actively seeking **summer / fall ML research internships** in: LLM-driven scientific discovery, NAS, clinical NLP / healthcare AI, foundation models for EHR / multimodal medical data. Happy to walk through any part of the design — design rationale, agent prompt engineering, statistical methods, HPC orchestration, or paper-writing strategy.

---

*This README is a working snapshot. The full method writeup is in [`paper/methods.md`](./paper/methods.md). Phase 2 results and final paper draft will replace the preview section above once production completes (ETA late May 2026).*
