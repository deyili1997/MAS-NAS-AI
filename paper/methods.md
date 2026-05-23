# Methods

This section describes our Multi-Agent Neural Architecture Search (MAS-NAS) framework for
EHR transformer architecture discovery, the two-layer cross-hospital prior, baseline
methods, ablation conditions, and the full experimental setup.

---

## 1. Problem Formulation

We aim to discover task-specific Transformer architectures for downstream EHR prediction
under a constrained parameter budget (≤ 2M parameters). Let $\mathcal{A}$ be a discrete
architecture search space (defined in §3) and $\mathcal{T} = \{t_1, \ldots, t_5\}$ a set
of clinical prediction tasks. For each target hospital $H_{\text{tgt}}$ and task $t$, the
search produces $a^\star_{t,H_{\text{tgt}}} \in \mathcal{A}$ maximizing held-out validation
AUPRC, subject to (i) $\text{params}(a) \le 2{,}000{,}000$ and (ii) $\text{embed\_dim} \bmod
\text{num\_heads} = 0$. To improve sample efficiency, the search leverages cross-hospital
prior knowledge mined from a pool of 8 source hospitals
$\mathcal{S} = \{H_1, \ldots, H_8\}$ (none of which overlap with $H_{\text{tgt}}$).

---

## 2. EHR Data Representation

### 2.1 Tokenization

Each patient's electronic health record is converted into a flat token sequence
preserving multi-encounter chronology. Four token modalities are used:

| Modality | Source | Token format |
|---|---|---|
| Diagnosis | ICD-9 codes | `DIAG_<code>` |
| Medication | NDC → ATC4 | `MED_<atc4>` |
| Lab test | LOINC + quantile bin | `LAB_LOINC_<code>-<bin>` (5 quantiles + text modifier) |
| Procedure | ICD-9 procedure | `PRO_<code>` |

For each patient, tokens from all admissions are concatenated chronologically (oldest →
newest) into a single sequence prefixed with `[CLS]`, with each token additionally
carrying an `admission_index` $i \in \{1, \ldots, K\}$ identifying its admission of
origin ($K$ = max admissions per patient, capped at 8). Special tokens `[PAD]`, `[CLS]`,
and `[MASK]` are reserved.

### 2.2 Lab token deduplication (OneFL+ only)

For OneFL+ source hospitals, raw LOINC lab measurements were retained across all
admissions and quantile bins, producing per-patient token counts up to 7× higher than
MIMIC-IV (which uses `groupby(subject_id, itemid).head(1)` patient-level
deduplication during preprocessing). To restore cross-hospital comparability, we applied
the equivalent patient-level deduplication to all 8 OneFL+ sites: per patient, retain
only the first occurrence of each LOINC code (stripping the value bin suffix as the
identity key). After this step, lab tokens per patient dropped from 50–128 to 25–50
across sites — closer to MIMIC-IV's 17.6 baseline.

### 2.3 Sequence-length cap and oldest-token truncation

To bound memory at fixed batch size and prevent outlier patients (with many admissions
and heavy lab tracking) from inflating the attention matrix, we enforce a hard
per-patient sequence-length cap $L_{\max} = 256$ (chosen offline to cover 99.01% of
patients in the pooled training set un-truncated). For patients whose flattened sequence
exceeds $L_{\max}$, the truncation rule keeps `[CLS]` plus the most-recent
$L_{\max} - 1$ tokens, dropping oldest admissions first. This rule is clinically
motivated (recent encounters are more predictive of near-term outcomes) and applied
uniformly across all hospitals.

### 2.4 Temporal subsampling of high-volume sites

For three OneFL+ sites whose patient cohort exceeded 180K (source_1: 199K, source_4:
187K, source_14: 232K), we restricted to patients whose most recent hospital admission
occurred on or after **2022-07-01**. This unified cutoff yielded 52,831 / 55,735 / 88,527
patients respectively. The remaining five OneFL+ sites (source_3, source_10, source_11,
source_15, source_16) had ≤ 84K patients and were not subsampled. The cutoff date was
chosen to balance the computational budget while preserving recent clinical practice
patterns; no other inclusion/exclusion was applied beyond the data owner's standard
preprocessing pipeline.

---

## 3. Search Space

The architecture search space is a 4-dimensional Cartesian product:

$$
\mathcal{A} = \{32, 64, 128, 256\} \times \{1, 2, 4, 8\} \times \{1, 2, 4, 8\} \times \{1, 2, 4, 8\}
$$

corresponding to (`embed_dim`, `depth`, `num_heads`, `mlp_ratio`), yielding $4^4 = 256$
candidate architectures. The $\text{embed\_dim} \bmod \text{num\_heads} = 0$ constraint
is satisfied by all $256$ combinations because each $\text{embed\_dim}$ is divisible by
each $\text{num\_heads}$. The $\text{params} \le 2{,}000{,}000$ constraint is enforced by
a hard reject in the Critic agent (§5.2) and is configurable at evaluation time.

---

## 4. AutoFormer-style Supernet

Following AutoFormer [Chen et al., ICCV 2021], we train a single
**supernet** spanning the search space, then evaluate any sub-architecture by direct
weight inheritance (no per-architecture retraining).

### 4.1 Supernet pretraining (per hospital)

A single supernet is masked-language-model (MLM) pretrained per hospital. Each batch
samples a random sub-architecture from $\mathcal{A}$ (uniform distribution) and applies
gradients only to the sampled weights. Pretraining objective: 15% token masking
(BERT-style — 80% `[MASK]`, 10% random token of same type, 10% unchanged) with
cross-entropy loss on masked positions.

**Hyperparameters**: `pretrain_epochs=100`, learning rate $1 \times 10^{-4}$ with linear
warmup over the first 10% of epochs then cosine decay to $1 \times 10^{-6}$, AdamW
(weight decay $0.01$), gradient clipping at norm $1.0$, batch size 64, dropout 0.1
(attention, residual, drop-path). Early stopping with patience 5 epochs of no validation
MLM loss improvement.

### 4.2 Sub-architecture evaluation (finetune)

To evaluate one sub-architecture $a \in \mathcal{A}$ on task $t$: load the supernet
checkpoint, extract weights corresponding to $a$'s configuration, and finetune for at
most 30 epochs with task-specific classification head (binary BCE for `death`,
`stay > 7d`, `readmission_3m`; multilabel BCE for 18-class phenotype prediction at 6M
and 12M horizons). Optimizer and dropout follow §4.1; learning rate $2 \times 10^{-4}$
with cosine decay. Early stopping patience 5. The reported metrics are the **average
test AUPRC over the top-3 best-validation-AUPRC epochs** (`top_k_epochs=3`) — a
checkpoint-averaging variant that reduces sensitivity to noisy single-epoch fluctuations
without additional training cost.

---

## 5. MAS-NAS: Multi-Agent Search Loop

The MAS-NAS controller is a coordinated LLM ensemble (default model:
`claude-sonnet-4-6`) that proposes, critiques, and tracks architecture trials over a
budget of $B = 100$ architecture evaluations.

### 5.1 Three agent roles

**Proposal Agent** (`agents/proposal_agent.py`). Given the search history, two-layer
prior context (§6), and the current strategy decision (explore/exploit), proposes a
small batch of candidate architectures (typically 1–3 per round). Receives the
architecture choice space, the strategy directive, completed trials' metrics, and the
top-k historical architectures retrieved from similar source hospitals as context.

**Critic Agent** (`agents/critic_agent.py`). Each proposal is reviewed against two
constraint categories:

| Category | Examples | Disposition |
|---|---|---|
| HARD constraints | params > 2M; `embed_dim % num_heads ≠ 0`; exact duplicate of a completed trial or a retrieved historical top-k arch | Auto-reject |
| SOFT constraints | Use of a "discouraged" level per Layer 2 prior; failure to explore the highest-importance feature dimension | Flag and require explicit rationale; reject only if rationale is absent or non-substantive |

Soft-constraint rejection is asymmetric — a discouraged level may still be accepted if
the Proposal Agent justifies why the target dataset might deviate from the
cross-hospital pattern (e.g., under-represented region, exploration of complementary
coverage).

**Experiment Agent** (`agents/experiment_agent.py`, dual role: strategy + evaluation).
After each accepted proposal is finetuned, the Experiment Agent updates the search state
with the new (architecture, metrics) tuple. Before the next round, it issues a
**strategy decision** — `EXPLORE` (sample under-represented architecture regions) or
`EXPLOIT` (refine near the best-so-far) — based on the empirical distribution of
completed experiments. The Layer 2 prior is treated as a *secondary* signal: it informs
the decision only when the empirical distribution is ambiguous.

### 5.2 Search loop

```
Initialize search_state = {completed: [], best: None}
context = gather_historical_context(target_hospital, task)  # two-layer prior
for round in 1..ceil(B / proposal_batch_size):
    strategy = ExperimentAgent.decide_strategy(context, search_state)
    proposals = ProposalAgent.propose(context, search_state, strategy)
    accepted = CriticAgent.review(proposals, completed_set, max_params)
    for a in accepted:
        metrics = finetune(supernet, a, task)
        search_state.completed.append((a, metrics))
        search_state.best = argmax_val_auprc(search_state.completed)
        save_to_disk(search_state)
return search_state.best
```

The loop terminates when `budget_remaining ≤ 0`. The final selected architecture is then
**finetuned once more with held-out test evaluation** to report the test AUPRC reported
in the main table.

---

## 6. Two-Layer Cross-Hospital Prior

The prior context $\mathcal{C}(H_{\text{tgt}}, t)$ supplied to the agents is the union
of two layers, each addressing a distinct question.

### 6.0 Overview

The raw material for both prior layers is the per-source-hospital metadata table
produced in Stage A — for each of the 8 OneFL+ source hospitals, the supernet is
pretrained once and $K = 100$ sub-architectures are finetuned on each of the 5
downstream tasks, producing one row per (architecture, task) tuple with the 4
architecture features, parameter count, FLOPs, and the four downstream metrics.
Pooled across hospitals, this yields roughly $8 \times 100 \times 5 = 4000$ rows
that drive both prior layers.

```
            4 arch features × N archs × 5 tasks × 8 source hospitals
                            │  (Stage A: run_pipeline.py)
                            ▼
                metadata.csv × 8   ────  dataset_summary.csv × 8
                       │                          │
            ┌──────────┴──────────┐               │
            ▼ pool per task       ▼ (matched src, ▼ hospital cosine
                                    matched task)   → matched source
    XGBoost surrogate +    top-k arch retrieval     │
    SHAP TreeExplainer     + 4 metrics per row      │
            │                     │                 │
            ▼                     │                 │
    mixed-effect model            │                 │
    on signed SHAP                │                 │
            │                     │                 │
            ▼                     │                 │
    architecture_prior.json       │                 │
    (Layer 2)                     ▼ Layer 1         │
            └─────────────┬───────┘                 │
                          ▼                         │
              context dict  ◀──────  _compute_target_summary
                          │
                          ▼
              Markdown prompt → Claude
              (Proposal / Critic / Strategy agents)
```

Conceptually the two layers run in parallel. Layer 1 is *target-specific*: given the
target (hospital, task), it returns concrete architecture exemplars from the most
similar source hospital. Layer 2 is *target-agnostic*: for each task it pools rows
across all 8 source hospitals and distills cross-hospital structural regularities
into a JSON-encoded set of preferred / discouraged levels and interaction rules.
Both outputs are merged into a single Markdown prompt that the Proposal Agent
consumes (described in §6.3).

### 6.1 Layer 1 — Dataset-similarity retrieval

**Goal**: provide concrete architecture examples (with metrics) from source hospitals
whose patient-population statistics are most similar to the target.

**Hospital similarity**: each hospital $H$ is summarized by a 20-dimensional dataset
profile vector (pretrain/finetune sample counts; unique diag/med/lab/pro vocabulary
sizes; average admissions per patient; average codes per patient per modality). The
similarity between target $H_{\text{tgt}}$ and each candidate $H_s$ is cosine similarity
of their summary vectors. The most-similar source hospital
$\hat{H}_s = \arg\max_{H_s \in \mathcal{S}} \cos(H_{\text{tgt}}, H_s)$ is selected as
the donor.

**Task fallback (Plan A)**: if the target task $t$ exists in $\hat{H}_s$'s metadata,
exact match is used; otherwise, a 6-dimensional task feature vector — `is_binary`,
`is_multilabel`, `num_classes_log_norm`, `label_entropy`, `positive_ratio`,
`time_horizon_norm` — is computed for the target task and compared via cosine similarity
to each task in $\hat{H}_s$. The closest source task is used in place of the missing
target task.

**Top-k architecture retrieval**: from $(\hat{H}_s, \hat{t})$'s metadata, the top
$k = 3$ architectures ranked by average rank across $\{\text{accuracy}, F_1, \text{AUROC}, \text{AUPRC}\}$
are retrieved and rendered as concrete (config, metrics) tuples in the Proposal Agent
prompt. The setting $k = 3$ is the production default
(`run_pipeline.sbatch`: `--top_k 3`); larger $k$ broadens exemplars but also inflates
the prompt and increases the chance of the Proposal Agent over-imitating a single
source hospital.

### 6.2 Layer 2 — Cross-hospital meta-regression prior

**Goal**: provide *universal directional priors* about which architecture levels help or
hurt across all hospitals, distilled from per-hospital SHAP signals via mixed-effect
models. This is the "Architecture Effect Prior".

**Stage A — per-task SHAP, pooled across source hospitals**. For each task
$t \in \mathcal{T}$, all $(a, \text{metrics})$ rows from the 8 source hospitals'
metadata are pooled. A composite target is constructed by ranking the pooled rows
on each of the four metrics in descending order (rank 1 = best) and averaging the
four ranks: $\text{avg\_rank}(a) = \frac{1}{4}\sum_{m \in \{\text{accuracy}, F_1, \text{AUROC}, \text{AUPRC}\}} \text{rank}_m(a)$.
The regression target is then $y = -\text{avg\_rank}$ (higher = better).
A single XGBoost regressor is fit to predict $y$ from the 4 architecture features
(treated as categorical). SHAP TreeExplainer is then applied to obtain signed
per-(row, feature) SHAP values. This yields a long-format $\text{shap\_values.csv}$
with the hospital identifier preserved per row, enabling Stage B below.

**Stage B — mixed-effect modeling on signed SHAP**. For each (task, feature) pair, a
categorical mixed-effects model is fit:

$$
\text{shap}_{\text{feat}} \sim C(\text{feat\_level}) + (1 \mid \text{hospital})
$$

The hospital random intercept controls per-site baseline drift in SHAP magnitude. From
the fitted coefficients and their 95% confidence intervals (per
`statsmodels.MixedLM`), each (feature, level) is classified into:

| Class | Criterion |
|---|---|
| Preferred | CI lower bound > 0 (significantly positive SHAP across hospitals) |
| Discouraged | CI upper bound < 0 (significantly negative SHAP) |
| Neutral | CI straddles 0 (no cross-hospital signal) |

For the top-2 features by mean abs SHAP, a pairwise interaction model is additionally
fit; preferred and avoid combinations are reported with the same CI rule. A
per-(feature) confidence label — `high` / `moderate` / `low` — is computed from the
ratio of effect magnitude to CI half-width.

If `statsmodels.MixedLM` fails to converge for a particular (task, feature) pair
(rare with 8 hospitals × $K = 100$ archs, but possible for degenerate level
distributions), the script falls back to ordinary least squares with cluster-robust
standard errors clustered by hospital — no random intercept, but cross-hospital
sampling noise is still propagated into the reported CI.

The output, `architecture_prior.json`, contains six structured fields per task:
`feature_importance_order`, `preferred_levels`, `discouraged_levels`,
`interaction_rules`, `confidence`, and a mandatory `_caveat` field that explicitly
labels the prior as **soft directional guidance derived from a TreeExplainer surrogate,
not causal**.

**Agent consumption**. The Layer 2 prior is rendered as a structured section in the
Proposal Agent and Critic Agent prompts (see Soft-prior framing — §5.1). Importantly,
the Critic does NOT auto-reject discouraged levels; it flags them and requires a
rationale field in the proposal.

### 6.3 Agent prompt assembly

The two prior layers feed into a single `context` dictionary that is shared across
all three agents and the search loop. It is constructed once at the start of each
search by `gather_historical_context` (`mas_search.py:348`) and contains seven
fields:

| Field | Source | Content |
|---|---|---|
| `target_summary` | `_compute_target_summary` | 20-D dataset profile of $H_{\text{tgt}}$ |
| `similar_hospital` | Layer 1 | $\hat{H}_s$, the most similar source hospital |
| `similarity_score` | Layer 1 | cosine similarity of summary vectors |
| `matched_task` | Layer 1 | $\hat{t}$, the exact or task-feature-matched source task |
| `matched_task_similarity` | Layer 1 | `None` if exact, else cosine similarity in 6-D task feature space |
| `top_k_archs` | Layer 1 | list of $k = 3$ (config, metrics) tuples from $(\hat{H}_s, \hat{t})$ |
| `meta_regression_prior` | Layer 2 | full content of `architecture_prior.json`; `{}` if Layer 2 disabled |

The Proposal Agent's `_build_prompt` function (`agents/proposal_agent.py:60-212`)
serialises these fields into a single Markdown document whose top-level section
structure is:

```markdown
## Search Space                         ← static, derived from CHOICES
## Parameter Budget                     ← static, max_params = 2M
## Target Dataset                       ← rendered from target_summary
## Historical Best Architectures        ← Layer 1: top_k_archs as JSON lines
## Architecture Prior — SOFT statistical guidance, NOT prescriptive rules
                                        ← Layer 2: preferred / discouraged /
                                          interactions / confidence / _caveat
## Already Tried (N architectures)      ← search_state.completed_experiments
## Budget Remaining: M architectures
## Search Strategy (set by Strategy Agent)   ← explore / exploit + rationale
## Output Format                        ← JSON schema enforced by parser
```

The Markdown is rendered with **soft-prior framing language** (cross-hospital
*directional priors, not hard constraints*) and an explicit instruction that any
proposal using a discouraged level must justify the deviation in its `rationale`
field. When Layer 2 is disabled (`mas_layer1_only`, `mas_cold`, or when
`architecture_prior.json` is absent), the `## Architecture Prior` section is omitted
from the prompt entirely rather than rendered as empty.

The Critic Agent prompt reuses the exact same `context` dict and adds a
critique-specific instruction block (HARD vs SOFT constraint dispositions, already
documented in §5.1, Table). The Experiment Agent — in its Strategy role — reads only
`search_state.completed_experiments` and the `meta_regression_prior` (as a
*secondary* signal), since exploration vs exploitation should be driven primarily by
the empirical distribution of completed trials, not by the prior.

This single-prompt design means the only prior-related state at inference time is
the small JSON `context` dictionary; there are no per-round prompt mutations beyond
the dynamically updated `Already Tried` and `Search Strategy` sections, which
simplifies reproducibility and ablation logging.

---

## 7. Baseline Methods

To benchmark MAS-NAS, we compare against five NAS baselines spanning random search,
evolutionary, and LLM-driven approaches. All baselines operate on the same search space
$\mathcal{A}$, supernet weights, and budget $B = 100$ as MAS-NAS, and report test AUPRC
under the same final-arch refinement protocol.

| Baseline | Method | Reference |
|---|---|---|
| `baseline0` | Uniform random sampling without replacement | — |
| `baseline1` | Evolutionary algorithm — population 8, select 4, mutation 4, crossover 4, mutation prob 0.1 | NAS-Bench-201 conventions |
| `baseline2` | LLM single-shot — Claude generates all 100 architectures in one prompt | — |
| `baseline3` | LLMatic — LLM iterative search with 16 niches, 0.3 random-init fraction, 0.85 mutation prob | Nasir et al., GECCO 2024 |
| `baseline4` | CoLLM-NAS — LLM Navigator/Generator pair with iterative refinement, temperature 0.7 | Adapted from Pham et al. |

`baseline0` and `baseline1` are pure black-box and do not call an LLM. `baseline2`,
`baseline3`, `baseline4` use the same LLM backbone (`claude-sonnet-4-6`) as MAS-NAS,
ensuring like-for-like compute comparison.

---

## 8. Ablation Conditions

To isolate the contribution of each component of the two-layer prior, we evaluate four
MAS-NAS variants in a 2×2 factorial layout over Layer 1 (retrieval) × Layer 2
(meta-regression):

| Mode | Layer 1 | Layer 2 | Tests |
|---|---|---|---|
| `mas` | Exact-match retrieval ON | ON | Full method (production) |
| `mas_layer1_only` | Exact-match retrieval ON | **OFF** | Incremental value of Layer 2 over Layer 1 alone |
| `mas_loto` (leave-one-task-out) | Exact task removed, 6-D feature-cosine fallback ON | ON | Layer 1 graceful degradation when exact task missing |
| `mas_cold` | OFF (no prior at all) | OFF | Multi-agent reasoning's intrinsic value, prior-free |

`mas_cold` is implemented via the `--no_history` flag, which short-circuits
`gather_historical_context()` and sets both `top_k_archs = []` and
`meta_regression_prior = {}`. `mas_layer1_only` uses `--no_meta_regression` which
preserves Layer 1 retrieval but skips loading `architecture_prior.json`. `mas_loto`
uses `--exclude_exact_task_from_history`, which drops the exact (source, target_task)
rows from the metadata before top-k retrieval, forcing the 6-D feature-cosine fallback.

For the ablation panel (Fig. 5), each mode is evaluated on all 5 tasks × 2 test
hospitals (MIMIC-III, MIMIC-IV) × 5 seeds {123, 456, 789, 1000, 1234}. Paired one-sided
Wilcoxon signed-rank tests (alternative: MAS variant > best LLM baseline) with
Holm-Bonferroni correction across the 5 tasks establish statistical significance.

---

## 9. Experimental Setup

### 9.1 Datasets

We use 10 hospitals in total:

| Role | Hospitals | n_patients | Selection criterion |
|---|---|---|---|
| Source pool (prior) | 8 OneFL+ sites: source_1, source_3, source_4, source_10, source_11, source_14, source_15, source_16 | 28K–88K | Patient-level LOINC dedup applied; source_1/4/14 additionally restricted to admissions ≥ 2022-07-01 to balance compute |
| Test (target) | MIMIC-IV | 42K | — |
| Test (target) | MIMIC-III | 24K | — |

OneFL+ sites are *de-identified secondary-care institutions* from the OneFlorida+
Clinical Research Consortium. Three sites in the original pool were excluded:
`source_6` (empty pickle, < 1MB), `source_9` (897 rows, statistically underpowered), and
`source_12` (death rate ≈ 0.09%, classifier collapse to majority class).

### 9.2 Tasks

Five binary and multilabel clinical prediction tasks (defined in
`utils/task_registry.py`):

| Task | Type | Definition |
|---|---|---|
| `death` | Binary | In-hospital mortality during the index admission |
| `stay` | Binary | Length of stay > 7 days |
| `readmission` | Binary | Re-admission within 3 months (90 days) |
| `next_diag_6m_pheno` | Multilabel (18 classes) | Any of 18 CCS phenotype categories diagnosed within 6 months post-discharge |
| `next_diag_12m_pheno` | Multilabel (18 classes) | Same, 12-month horizon |

The 18 CCS phenotype classes are defined in
`utils/task_registry.py:PHENO_LABEL_ORDER` and span common chronic conditions and acute
events.

Each task has a per-hospital train/val/test split (70/15/15 by patient ID) provided by
the upstream preprocessing pipeline.

### 9.3 Hyperparameters (summary)

| Parameter | Pretrain | Finetune | Note |
|---|---|---|---|
| Epochs (max) | 100 | 30 | Early stop if no val improvement for 5 epochs |
| Batch size | 64 | 64 | bs=64 chosen to fit L4 GPU memory at $L_{\max}=256$ |
| Learning rate | $1 \times 10^{-4}$ | $2 \times 10^{-4}$ | Linear warmup 10% → cosine decay → $1 \times 10^{-6}$ |
| Optimizer | AdamW | AdamW | weight_decay 0.01, grad clip norm 1.0 |
| Dropout (attn / residual / drop-path) | 0.1 / 0.1 / 0.1 | same | — |
| Mask rate (MLM) | 0.15 | 0.15 (label noise reg.) | BERT-style 80/10/10 |
| Sequence length cap | 256 | 256 | Truncate oldest tokens |
| Max architecture params | 2,000,000 | 2,000,000 | Critic hard reject if exceeded |
| Top-k epoch averaging | — | 3 | Reduces single-epoch noise |
| MAS-NAS budget per (target, task) | — | 100 | Architecture evaluations |
| MAS-NAS LLM backbone | — | claude-sonnet-4-6 | Smoke runs used claude-haiku-4-5 for cost |
| Random seeds | — | 5 (123, 456, 789, 1000, 1234) | All main results |
| Supernet ckpt | — | Shared across all archs/tasks per hospital | Stored once at `<hospital>/checkpoint_mlm/mlm_model.pt` |

### 9.4 Hardware & Implementation

All training runs use a single NVIDIA L4 GPU (24 GB VRAM) on the University of Florida
HiPerGator cluster (`hpg-turin` partition, max 14-day wall time). Implementation is in
PyTorch 2.5 with mixed precision disabled (FP32) for reproducibility. Code is open-source
at `https://github.com/<...>/MAS-NAS-AI` and uses the
`Autoformer` conda environment (Python 3.10).

Memory management: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to mitigate
fragmentation on long-running pretraining jobs. Intermediate results
(`metadata.csv`, `autoformer_results.csv`, `traditional_results.csv`) are flushed
incrementally after every (task, architecture) finetune so wall-time timeouts do not
lose data.

### 9.5 Cost & runtime

Per-hospital production run (`run_pipeline.py`, single L4 GPU):

| Phase | Cost |
|---|---|
| Supernet MLM pretraining (100 epochs) | 1–3 hours, depending on patient cohort size |
| Finetune (100 archs × 5 tasks × ≤30 epochs) | 30–70 hours |
| Total per hospital | 30–75 hours |
| **8 OneFL+ sites in parallel** | **~3 days wall clock (slowest site sets the pace)** |

For LLM-based baselines and MAS-NAS, average API consumption is ~15 LLM calls per
(target, task) at budget 100, totaling ~$10–15 in API cost per full ablation panel at
Sonnet pricing.

---

## 10. Evaluation Protocol

### 10.1 Primary metric

Test AUPRC on the held-out test split, computed once per (method, task, target_hospital,
seed) tuple. AUPRC is preferred over AUROC for the moderately imbalanced binary tasks
(positive rates 1–6%) and is the standard metric for the multilabel phenotype tasks.

### 10.2 Reporting

| Output | Form | Source |
|---|---|---|
| Table 1 (Main results) | Mean ± std test AUPRC over 5 seeds, per (method, task, hospital), with paired Wilcoxon significance markers vs. random baseline | `analyze/aggregate_results.py:build_main_table` |
| Table 2 (Compute efficiency) | MAS-NAS at budget 30 vs. all baselines at budget 100: best AUPRC, wall-clock, LLM call count | `analyze/aggregate_results.py:build_efficiency_table` |
| Figure 1 (Search trajectory) | Best val AUPRC vs. architecture evaluations 1..100, 6 method curves with ± std shade | `analyze/plot_search_trajectory.py` |
| Figure 2 (Pareto front) | Validation AUPRC vs. architecture parameter count, per task | `analyze/plot_pareto.py` |
| Figure 3 (Supernet validity) | Scatter of AutoFormer-style supernet rankings vs. independent-pretrain ("Traditional") rankings for k=150 random architectures, per task; Pearson r reported | `run_regression.py` + `analyze/plot_regression.py` |
| Figure 4 (Layer 2 SHAP interpretability) | Per-task SHAP summary, bar, and pairwise interaction plots from pooled cross-hospital meta-regression | `shap_analysis.py` |
| Figure 5 (Layer ablation) | Bar chart per task: MAS-exact / MAS-Layer1-only / MAS-LOTO / MAS-cold / best LLM baseline; significance bracket = paired Wilcoxon MAS-LOTO vs. best baseline | `analyze/plot_loto_ablation.py` |
| Figure 6 (Source-hospital selection, supplementary) | Heatmap of source-hospital choice frequencies across (target, task) combinations, demonstrating per-task source selection rather than a single dominant prior | `analyze/plot_source_hospital_choice.py` |

### 10.3 Statistical testing

For Table 1 and Figure 5: paired one-sided Wilcoxon signed-rank tests comparing each
method's per-seed test AUPRC distribution vs. the baseline being compared (random for
Table 1; best LLM baseline for Figure 5 LOTO claim). P-values are adjusted within each
(task, hospital) family using Holm-Bonferroni correction over the 5 candidate baselines.
Significance markers: $* p < 0.05$, $** p < 0.01$, $*** p < 0.001$ after correction.

### 10.4 Reproducibility

All 5 random seeds are fixed via `set_random_seed(seed, deterministic=True)`. CUDNN
benchmark mode is **disabled** for the main results (enabled only for the search
trajectory plot smoke checks). The full per-seed metadata.csv files (architecture
configurations + all 4 metrics per epoch) and agent I/O logs are released alongside the
paper.

---

## Appendix: Code-level reference

The following files in our open-source implementation correspond directly to the
methods above:

| Methods section | File | Function / line |
|---|---|---|
| §2.1 Tokenization | `utils/dataset.py` | `PreTrainEHRDataset._transform_pretrain_data` (line 44) |
| §2.2 LOINC dedup (OneFL+) | OneFL+ preprocessing pipeline (data owner) | `lab_pd.drop_duplicates(['SUBJECT_ID', 'LOINC_KEY'])` |
| §2.3 Sequence-length cap | `utils/dataset.py` | `MAX_SEQ_LEN = 256`, `_truncate_oldest()` (line 35) |
| §2.4 Temporal subsample | `tools/subsample_recent_patients.py` | `--cutoff_date 2022-07-01` |
| §3 Search space | `run_pipeline.py` | `CHOICES` dict (line 33) |
| §4.1 Supernet pretrain | `run_pipeline.py` | `pretrain()` (line ~220) |
| §4.2 Sub-arch finetune | `agents/experiment_agent.py` | `_finetune_one_arch()` |
| §5.1 Agents | `agents/{proposal,critic,experiment}_agent.py` | — |
| §5.2 Search loop | `mas_search.py:main()` | line ~640 |
| §6.1 Layer 1 retrieval | `mas_search.py` | `gather_historical_context()`, `_find_most_similar_hospital()`, `_get_top_k_archs()` |
| §6.2 Layer 2 meta-regression | `shap_analysis.py` + `run_meta_regression.py` | — |
| §7 Baselines | `baselines/baseline{0..4}.py` | — |
| §8 Ablations | `mas_search.py` | `--no_history`, `--no_meta_regression`, `--exclude_exact_task_from_history` |
| §10.2 Tables/figures | `analyze/` | aggregate_results, plot_search_trajectory, plot_pareto, plot_loto_ablation, plot_regression |
