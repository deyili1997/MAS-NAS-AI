# Methods: All Baselines and MAS-NAS

## Search Space

All methods share the same scalar search space over four hyperparameters:

| Hyperparameter | Choices |
|----------------|---------|
| `embed_dim`    | {32, 64, 128, 256} |
| `depth`        | {1, 2, 4, 8} |
| `mlp_ratio`    | {1, 2, 4, 8} |
| `num_heads`    | {1, 2, 4, 8} |

Total discrete configurations: 4⁴ = 256, subject to `embed_dim % num_heads == 0`
and `num_params ≤ 4,000,000`. All methods operate under a budget of **B = 30**
architecture evaluations per (hospital, task, seed) run.

---

## Shared Infrastructure

Every method reuses the same evaluation pipeline:

- **Supernet**: A single EHR Transformer pretrained via masked-language modelling
  (MLM) per hospital. Weights are shared across all architectures (weight-sharing NAS /
  one-shot NAS). This amortises the pretraining cost to a one-time O(1) overhead.
- **Evaluation**: Each candidate architecture is extracted from the supernet and
  fine-tuned on the downstream task for up to 30 epochs with early stopping
  (patience = 5). Top-3 epoch averaging is applied to obtain stable validation metrics.
  Four metrics are recorded: Accuracy, F1, AUROC, AUPRC.
- **Selection**: The architecture with the best **composite rank** (average of per-metric
  validation ranks, lower = better) is selected and re-evaluated on the held-out test set.
- **LLM backbone** (for LLM-based methods): `anthropic/claude-3.5-haiku` via OpenRouter.

---

## B0 — Random Search (Baseline 0)

**Reference**: Classic random NAS baseline (Bergstra & Bengio, 2012).

**Algorithm**: For each budget step, uniformly sample one configuration from the valid
search space (reject duplicates and constraint violations) and evaluate it.

**Purpose**: Establishes a lower bound. Any method claiming "search intelligence"
must exceed random sampling under the same compute.

**Key parameters**: None beyond the shared budget and search space.

---

## B1 — Evolutionary Algorithm / EA (Baseline 1)

**Reference**: Regularised Evolution (Real et al., 2019).

**Algorithm**:
1. Initialise a population of `P = 8` random architectures.
2. Each iteration: select the top-`S = 4` architectures (by composite rank),
   apply uniform mutation (randomly perturb one hyperparameter) and crossover
   to generate `M = 4` mutants and `C = 4` offspring.
3. Evaluate new candidates; maintain the population by removing the worst members.
   Mutation probability per hyperparameter: `p_mut = 0.3`.

**Purpose**: Tests whether classical evolutionary search outperforms random or LLM-driven methods.

**Key parameters**: `--population_num 8 --select_num 4 --mutation_num 4 --crossover_num 4 --m_prob 0.3`

---

## B2 — GENIUS (Baseline 2)

**Reference**: GENIUS (Zheng et al., 2023, *"Can GPT-4 Perform Neural Architecture
Search?"*, arXiv:2304.10970). A single-LLM iterative NAS framework that queries a
large language model to propose one architecture per round, conditioned on the
performance of previously evaluated candidates.

**Algorithm**:
1. Build a prompt containing the full search history (all evaluated architectures
   and their validation metrics).
2. Ask the LLM to propose exactly **one** new architecture.
3. Evaluate, append to history, repeat until budget exhausted.

No multi-agent debate, no cross-hospital context, no architectural prior.
The LLM sees only its own search trajectory.

**Purpose**: Isolates the value of LLM-driven proposal vs. random. Tests whether a
single LLM without context can guide effective NAS — the closest prior-work
counterpart to MAS-NAS that lacks both multi-agent coordination and the two-layer
cross-hospital prior.

**Key parameters**: `--model anthropic/claude-3.5-haiku`

---

## B3 — LLMatic / Quality-Diversity + LLM (Baseline 3)

**Reference**: LLMatic (Nasir et al., 2023, arXiv:2306.01102).

**Algorithm**: Combines MAP-Elites quality-diversity (QD) archiving with LLM-driven
mutation and crossover.

1. **Archive**: A 2D grid of `N = 16` niches partitioned by behavioural descriptors
   `(log_params, log_FLOPs)`, normalised to [0,1]².
2. **Initialisation**: Randomly fill a fraction (`random_init_frac = 0.3`) of
   niches with random architectures.
3. **QD loop** (until budget exhausted):
   - Select parent(s) from high-curiosity niches (niches where LLM proposals
     previously improved the archive).
   - **Mutation LLM call**: prompt the LLM with the parent config and a
     natural-language mutation directive ("increase depth", "reduce embed_dim", etc.).
     Mutation probability: `p_mut = 0.85`; temperature jitter: `±0.1`.
   - **Crossover LLM call**: occasionally combine two parents.
   - Evaluate candidate; update archive if it improves a niche.
   - Update curiosity scores: `+1` for archive improvements, `−0.5` otherwise.
4. Top-`n = 3` niche elites are used for final selection.

**Purpose**: Tests LLMatic-style curiosity-driven QD search without historical
cross-hospital context.

**Key parameters**: `--n_niches 16 --random_init_frac 0.3 --mutation_prob 0.85 --temp_jitter 0.1 --top_n_select 3 --temperature 0.7`

---

## B4 — CoLLM-NAS (Baseline 4)

**Reference**: Wang et al., "CoLLM-NAS: Exploring High-Performing Architectures via
Collaborative LLMs" (2024).

**Algorithm**: Decomposes NAS into three components:

1. **Navigator LLM** (stateful): Receives the full search history
   `H = [(strategy, [eval_results]), ...]` and outputs a free-text *search strategy*
   describing which region of the space to target next. Starts in exploration mode
   and progressively shifts to exploitation. `temperature = 0.7`.

2. **Generator LLM** (stateless): Receives only the current strategy (not H) and
   the search-space rules. Generates `C = 3` concrete candidate architectures per
   iteration as JSON. Stateless design prevents "prior-knowledge contamination" of
   the Navigator's strategy.

3. **Coordinator**: Enforces legality, deduplication, and resource constraints.
   Evaluates each unique legal candidate via the shared supernet pipeline.

**Purpose**: Tests the Navigator–Generator separation paradigm without cross-hospital
context (no architectural prior, no history from other hospitals).

**Key parameters**: `--candidates_per_iter 3 --navigator_temperature 0.7 --generator_temperature 0.7`

---

## MAS-NAS (Proposed Method)

**Reference**: This work.

MAS-NAS augments any LLM-driven NAS with a **two-layer cross-hospital prior** and a
**three-agent propose–critique–revise loop**.

### Two-Layer Prior (Historical Context)

**Layer 1 — Task-Driven Hospital Selection**:
Given target hospital H and task t, we select the most similar source hospital s* by
computing cosine similarity between task-specific feature vectors:

```
task_feat(H, t) = [is_binary, is_multilabel, log(num_classes)/log(20),
                   label_entropy, positive_ratio, time_horizon/365]
```

where `label_entropy` and `positive_ratio` are computed from H's downstream data.
The source hospital s* with the most similar task feature vector is selected:

```
s* = argmax_s  cosine(task_feat(H,t), task_feat(s,t))
```

(Dataset-level cosine similarity is used as a fallback for the LOTO ablation.)

The top-k architectures from s*/t (ranked by composite validation metric) are
retrieved and included in the agent prompt as concrete examples.

**Layer 2 — Meta-Regression Prior**:
A mixed-effects meta-regression (MixedLM) is fitted on SHAP values pooled across
all source hospitals. For each hyperparameter level, it estimates a mean SHAP effect
on the composite rank. Levels with significantly positive effect are labelled
`preferred`; significantly negative are `discouraged`. These soft directional priors
are injected into the agent prompts as architectural guidance.

### Three-Agent Search Loop

The search proceeds in **rounds** until the evaluation budget B is exhausted.
Each round invokes four agent calls in sequence (A1 → A2 → A1-revise → A3a → A3b),
with an inner Propose–Critique–Revise loop of up to R = 3 passes.

#### Round Structure

```
WHILE budget_remaining > 0:
    Round r:
      A1  →  Propose K candidates
      FOR pass p in 1..R:
          A2  →  Critique proposals → accepted_p, rejected_p
          IF all accepted OR all rejections are exact duplicates: BREAK
          A1  →  Revise rejected_p (non-duplicate only)
      END FOR
      Deduplicate across revision passes (keep first occurrence)
      A3a →  Evaluate all accepted candidates (supernet finetune)
      A3b →  Decide strategy for round r+1
```

#### Agent 1 — Architecture Proposal

**Role**: Generates a batch of novel architecture candidates informed by the
two-layer prior and the current search state.

**Input context** (injected into LLM prompt):
- **Search space**: All valid choices for `embed_dim`, `depth`, `mlp_ratio`, `num_heads`
  and the `embed_dim % num_heads == 0` constraint.
- **Infeasibility pre-computation**: A pre-computed list of `(embed_dim, depth)` pairs
  that always exceed the parameter budget at minimum `mlp_ratio = 1`. These are
  shown to the LLM as hard constraints to avoid wasted proposals.
- **Layer 1 — Historical top-k architectures**: The top-k architectures retrieved
  from the task-similar source hospital (with their validation metrics), served as
  concrete examples of what works in a similar clinical environment.
- **Layer 2 — Meta-regression prior**: A soft directional prior listing
  `preferred_levels` (positive cross-hospital SHAP effect) and `discouraged_levels`
  (negative effect) for each hyperparameter, ranked by `feature_importance_order`.
  Confidence labels (high/medium/low) are included.
- **Search history**: All architectures evaluated so far in this run, with their
  validation AUPRC, AUROC, F1, and Accuracy. The agent is instructed not to
  propose exact duplicates of already-evaluated configurations.
- **Current strategy**: `"exploration"` (diversity required, do not cluster near
  the current best) or `"exploitation"` (focus on refining the best region).

**Output**: A JSON array of architecture proposals, each specifying
`{embed_dim, depth, mlp_ratio, num_heads}` with a brief rationale.

**Revision mode**: When called for revision, Agent 1 additionally receives the
Critic's rejection reasons and risk tags for each rejected proposal, and must
produce revised configurations that address the stated issues.

---

#### Agent 2 — Proposal Critic

**Role**: Reviews each proposal from Agent 1 and issues an ACCEPT or REJECT
decision with structured feedback.

**Input context** (injected into LLM prompt):
- Same search space, constraints, and Layer 1/2 context as Agent 1.
- All proposals from Agent 1 in this revision pass.
- Current strategy (influences acceptance threshold: exploitation mode applies
  stricter diversity requirements; exploration mode penalises clustering near
  already-evaluated regions).
- Already-accepted proposals from earlier passes of this round (for cross-proposal
  deduplication).

**Review criteria** (applied in order):
1. **Constraint check**: `embed_dim % num_heads == 0`; `params ≤ max_params`.
   Violations are hard-rejected with tag `"constraint_violation"`.
2. **Duplicate detection**: Exact match against all previously evaluated
   architectures and already-accepted proposals in this round. Hard-rejected with
   tag `"duplicate"`.
3. **Prior alignment**: The Critic checks alignment of each proposal with the
   Layer 2 preferred/discouraged levels and flags departures from the prior
   (soft feedback, does not auto-reject).
4. **Diversity assessment**: During exploration, clustered proposals (multiple
   candidates with identical `depth` or `embed_dim`) are flagged as `"low_diversity"`.

**Output**: Two lists — `accepted` (proceed to evaluation) and `rejected`
(sent back to Agent 1 for revision). Each rejected proposal carries structured
`risk_tags` and a free-text critique.

**Duplicate-rejection bypass**: Exact-duplicate rejections are filtered before
passing to Agent 1 for revision. Since LLMs cannot meaningfully revise an exact
duplicate (they tend to re-propose the same configuration), the revision round
is skipped for such cases to avoid wasting LLM calls.

---

#### Agent 3a — Experiment Manager

**Role**: Evaluates each accepted architecture via the shared supernet pipeline
and records results.

For each accepted architecture:
1. Extract the subnet from the supernet checkpoint.
2. Fine-tune on the downstream task for up to 30 epochs (early stopping patience = 5).
3. Record val Accuracy, F1, AUROC, AUPRC. Compute top-3 epoch average for stability.
4. Append the result to `search_state["completed_experiments"]`.
5. Update the global best architecture tracker (by composite validation rank).

This agent is **purely algorithmic** — no LLM is involved. GPU compute is the
bottleneck here, not LLM inference.

---

#### Agent 3b — Strategy Agent

**Role**: After each round's experiments, decides whether the next round should
pursue **exploration** (diversity) or **exploitation** (refinement).

**Input context**:
- Full trajectory of all evaluated architectures, sorted by composite rank.
- Current best architecture and its validation metrics.
- Layer 2 prior (secondary signal: if the search has exhausted preferred-level
  combinations, a switch to exploitation is more likely warranted).
- Budget remaining.

**Decision logic** (LLM-driven):
- `"exploration"`: Chosen when the trajectory shows high variance, when preferred
  regions have not yet been thoroughly sampled, or when budget is ample.
- `"exploitation"`: Chosen when a promising neighbourhood has emerged, when
  remaining budget is limited, or when the Pareto frontier around the best
  architecture is still unexplored.

**Output**: `{"strategy": "exploration" | "exploitation", "rationale": "..."}`.

The first round always uses `"exploration"` (no history available yet).

---

#### Termination and Failure Handling

- The loop terminates when `budget_remaining == 0`.
- If Agent 1 produces no valid proposals (after constraint filtering), or if no
  proposals survive the Critic review, this counts as a **consecutive failure**.
  After `max_consecutive_failures = 3` consecutive empty rounds, the search
  terminates early to avoid wasting compute.
- All LLM API errors are caught gracefully; already-completed results are preserved
  and the failure counter is incremented.

---

#### Complete Round Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   MAS-NAS Search Round r                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  INPUTS (shared across all agents in round r):           │
│    • Two-layer prior (Layer 1 top-k + Layer 2 SHAP)      │
│    • search_state.completed_experiments                  │
│    • strategy from round r-1 (A3b output)                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ A1: Propose  →  [p₁, p₂, ..., pₖ]               │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────┐    │
│  │ Pass 1..R:                                       │    │
│  │   A2: Critique → accepted_p, rejected_p          │    │
│  │   IF all accepted OR duplicates only: BREAK      │    │
│  │   A1: Revise(rejected_p \ duplicates)            │    │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│  [Deduplicate across revision passes]                    │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ A3a: Evaluate accepted architectures             │    │
│  │      (supernet finetune, GPU-bound)              │    │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ A3b: Strategy decision → "exploration" or        │    │
│  │       "exploitation" for round r+1               │    │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| B | 30 | Total evaluation budget |
| R | 3 | Max revision passes per round |
| max_consecutive_failures | 3 | Early-stop threshold |
| First-round strategy | exploration | No history available |
| LLM backbone | claude-3.5-haiku | Via OpenRouter |

### MAS-NAS Ablations (Fig. 5)

| Variant | Layer 1 | Layer 2 | Hospital Selection |
|---------|---------|---------|-------------------|
| **MAS-NAS** | ✅ | ✅ | Task-driven |
| MAS-L1only | ✅ | ❌ | Task-driven |
| MAS-LOTO | ✅ | ✅ | Dataset-level → drop exact task records → Plan A fallback |
| MAS-cold | ❌ | ❌ | None |

**MAS-LOTO** tests robustness of Layer 1 Plan A (task-feature cosine fallback) when
exact task records are absent from the most similar source hospital. It reverts to
dataset-level hospital selection, then drops the target task's records, forcing the
fallback path.

### Prior Configuration

| Item | Value |
|------|-------|
| Prior source hospitals | source_1, source_4, source_14, source_16 |
| Internal test hospital | source_15 |
| External test hospital | MIMIC-IV |
| Architecture prior excluded from | source_15 (target) |
| Layer 1 excluded from candidates | source_15 |
| LLM backbone | `anthropic/claude-3.5-haiku` |
| Budget B | 30 evaluations |
| Seeds | 111, 123, 456, 789, 999 |

---

## Summary Table

| Method | LLM | Multi-agent | Cross-hospital prior | Search paradigm |
|--------|-----|-------------|---------------------|-----------------|
| B0 Random | ❌ | ❌ | ❌ | Uniform sampling |
| B1 EA | ❌ | ❌ | ❌ | Evolutionary |
| B2 GENIUS | ✅ | ❌ | ❌ | Single-agent iterative |
| B3 LLMatic | ✅ | ❌ | ❌ | QD + LLM mutation/crossover |
| B4 CoLLM-NAS | ✅ | ✅ (Nav+Gen) | ❌ | Navigator–Generator |
| **MAS-NAS** | ✅ | ✅ (3-agent) | ✅ (L1+L2) | Propose–Critique–Revise |
