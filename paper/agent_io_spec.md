# ATHENA Agent I/O Specification

This document specifies the exact inputs and outputs of each agent in the
ATHENA three-agent NAS controller. All four agents share a common `context`
dict assembled once per search run by `gather_historical_context()`, plus a
`search_state` dict updated incrementally after every evaluation.

---

## Shared Data Structures

### `context` (assembled once, read-only for agents)

```python
context = {
    # Target hospital profile
    "target_summary": {
        "hospital": str,               # e.g. "source_15"
        "Pretrain_num_samples": int,
        "Pretrain_avg_lab_per_patient": float,
        ...                            # 20-D dataset summary features
    },

    # Layer 1 — task-driven hospital selection result
    "similar_hospital":       str,     # e.g. "source_16"
    "similarity_score":       float,   # task-feature cosine similarity
    "matched_task":           str,     # task used for retrieval
    "matched_task_similarity": float | None,  # None if exact match

    # Layer 1 — retrieved architecture exemplars
    "top_k_archs": [
        {
            "embed_dim": int, "depth": int, "mlp_ratio": int, "num_heads": int,
            "num_params": int, "flops": int,
            "val_accuracy": float, "val_f1": float,
            "val_auroc": float, "val_auprc": float
        },
        ...  # typically top-3 by composite validation rank
    ],

    # Layer 2 — meta-regression prior
    "meta_regression_prior": {
        "feature_importance_order": ["embed_dim", "depth", "mlp_ratio", "num_heads"],
        "preferred_levels": {
            "embed_dim":  [128, 256],   # stable positive cross-hospital SHAP
            "depth":      [4, 8],
            "mlp_ratio":  [4],
            "num_heads":  [4, 8]
        },
        "discouraged_levels": {
            "embed_dim":  [32],
            "depth":      [1],
            "mlp_ratio":  [],
            "num_heads":  [1]
        },
        "confidence": {
            "embed_dim": "high", "depth": "medium",
            "mlp_ratio": "low",  "num_heads": "medium"
        },
        "interaction_rules": [
            {
                "pair": ["embed_dim", "depth"],
                "preferred_combinations": [[256, 4], [128, 8]],
                "avoid_combinations":     [[32, 1]]
            }
        ],
        "_caveat": "SHAP reflects correlation not causation ..."
    }
}
```

### `search_state` (updated after every evaluation)

```python
search_state = {
    "completed_experiments": [
        {
            "embed_dim": int, "depth": int, "mlp_ratio": int, "num_heads": int,
            "num_params": int, "flops": int,
            # Validation metrics only (test set never touched during search)
            "val_accuracy": float, "val_f1": float,
            "val_auroc": float,    "val_auprc": float
        },
        ...
    ],
    "budget_remaining": int,          # decrements by 1 per evaluation
    "best_proposal":   dict | None,   # arch with lowest composite val rank
    "best_model_sd":   dict | None,   # PyTorch state_dict of best model
    "best_config":     dict | None,   # internal per-layer config of best arch
    "eval_times_sec":  list[float],   # wall-clock seconds per evaluation
}
```

---

## Agent 1 — Proposal Agent

**File**: `agents/proposal_agent.py`  
**Calls per round**: 1 (propose) + up to R=3 (revise)  
**LLM backbone**: `anthropic/claude-3.5-haiku`

### Input

| Field | Type | Description |
|-------|------|-------------|
| `context` | dict | Full context including Layer 1 top-k and Layer 2 prior |
| `search_state` | dict | All previously evaluated architectures + budget remaining |
| `strategy` | dict | `{"strategy": "exploration"\|"exploitation", "rationale": str}` |
| `max_params` | int | Hard parameter budget (e.g. 4,000,000) |
| `vocab_size` | int | Hospital-specific vocabulary size (needed for param estimation) |
| `num_classes` | int | 2 for binary tasks, 18 for phenotype multilabel |
| `max_flops` | int\|None | Optional FLOPs budget |

**Additional input in Revise mode**:

| Field | Type | Description |
|-------|------|-------------|
| `rejected_with_critiques` | list[dict] | Each entry: `{proposal, critique, risk_tags}` from Agent 2 |

### Prompt content (injected sections)

1. **Role**: NAS expert for EHR Transformer models
2. **Search space**: `CHOICES = {embed_dim:[32,64,128,256], depth:[1,2,4,8], mlp_ratio:[1,2,4,8], num_heads:[1,2,4,8]}`; constraint `embed_dim % num_heads == 0`
3. **Infeasibility table**: pre-computed `(embed_dim, depth)` pairs that always exceed `max_params` even at `mlp_ratio=1`
4. **Target dataset**: hospital name + 20-D summary stats
5. **Layer 1 top-k**: historical best architectures from task-similar source hospital
6. **Layer 2 prior**: preferred/discouraged levels, confidence, feature importance ranking, interaction rules
7. **Search history**: all `completed_experiments` with val metrics
8. **Budget**: `budget_remaining`
9. **Strategy**: `"exploration"` (propose diverse architectures) or `"exploitation"` (refine around best)
10. *(Revise only)*: rejected proposals with critique text and `risk_tags`

### Output

```json
[
  {
    "embed_dim": 128,
    "depth": 4,
    "mlp_ratio": 4,
    "num_heads": 4,
    "rationale": "Mid-size model targeting preferred embed_dim=128; depth=4 balances capacity and convergence speed per the architecture prior."
  },
  {
    "embed_dim": 256,
    "depth": 2,
    "mlp_ratio": 2,
    "num_heads": 8,
    "rationale": "Exploration: wide but shallow to cover an under-explored region."
  }
]
```

**Format**: JSON array. Number of proposals: ≥1, ≤`budget_remaining`.  
Each entry must include all four hyperparameters plus a free-text `rationale`.

---

## Agent 2 — Critic Agent

**File**: `agents/critic_agent.py`  
**Calls per round**: up to R=3 (one per revision pass)  
**LLM backbone**: `anthropic/claude-3.5-haiku`

### Input

| Field | Type | Description |
|-------|------|-------------|
| `context` | dict | Same context as Agent 1 (Layer 1 + Layer 2) |
| `search_state` | dict | All previously evaluated architectures |
| `proposals` | list[dict] | Architectures from Agent 1 to review |
| `strategy` | dict | Current strategy (affects diversity threshold) |
| `max_params` | int | Hard parameter budget |
| `already_accepted` | list[dict] | Proposals accepted in earlier passes this round |
| `vocab_size`, `num_classes`, `max_flops`, `flops_seq_len` | | Constraint checking |

### Prompt content (injected sections)

1. **Role**: NAS critic for EHR Transformer models
2. **Search space + constraints**: same as Agent 1
3. **Layer 1 top-k**: historical best architectures (reference)
4. **Layer 2 prior**: preferred/discouraged levels
5. **Proposals to review**: full list from Agent 1
6. **Strategy**: `"exploration"` → stricter diversity requirement; `"exploitation"` → allow clustering near best
7. **Already accepted**: proposals accepted in earlier passes (cross-pass deduplication)

### Review criteria (applied in order)

1. **Constraint violation** (hard reject): `embed_dim % num_heads ≠ 0` or `params > max_params` → tag `"constraint_violation"`
2. **Duplicate** (hard reject): exact match in `completed_experiments` or `already_accepted` → tag `"duplicate"`
3. **Prior misalignment** (soft flag): proposes discouraged level without rationale → tag `"prior_misalignment"`
4. **Low diversity** (soft flag, exploration mode only): multiple proposals with identical `depth` or `embed_dim` → tag `"low_diversity"`

### Output

```
accepted: [
  {"embed_dim": 128, "depth": 4, "mlp_ratio": 4, "num_heads": 4, "rationale": "..."},
  ...
]

rejected: [
  {
    "proposal": {"embed_dim": 32, "depth": 1, ...},
    "critique": "embed_dim=32 is strongly discouraged (negative SHAP, high confidence). No compensating rationale provided.",
    "risk_tags": ["prior_misalignment"]
  },
  {
    "proposal": {"embed_dim": 128, "depth": 4, "mlp_ratio": 4, "num_heads": 4},
    "critique": "Exact duplicate of experiment #3 (val_auprc=0.512).",
    "risk_tags": ["duplicate"]
  }
]
```

> **Note**: `"duplicate"`-tagged rejections are filtered out before being passed
> to Agent 1 for revision. LLMs cannot meaningfully fix an exact duplicate.

---

## Agent 3a — Experiment Manager

**File**: `agents/experiment_agent.py` (`run_trials`)  
**No LLM call** — purely algorithmic.

### Input

| Field | Type | Description |
|-------|------|-------------|
| `reviewed_proposals` | list[dict] | Architectures accepted by Agent 2 |
| `search_state` | dict | Current search state (updated in-place) |
| `ckpt` | dict | Supernet checkpoint: weights + vocab_size + max_adm_num + arch config |
| `train_loader`, `val_loader` | DataLoader | Hospital/task-specific data |
| `args` | Namespace | Fine-tuning hyperparams: lr, weight_decay, finetune_epochs, finetune_patience, top_k_epochs |

### Subnet extraction (weight-sharing)

Each accepted architecture is extracted as a **left-prefix slice** of the supernet:

```python
config = {
    "embed_dim": [proposal["embed_dim"]] * depth,
    "layer_num": depth,                             # first N layers only
    "mlp_ratio": [proposal["mlp_ratio"]] * depth,
    "num_heads": [proposal["num_heads"]] * depth,
}
model.set_sample_config(config)  # activates left-prefix slice

# Weight slicing (LinearSuper):
# sample_weight = W[:sample_out_dim, :sample_in_dim]  ← top-left submatrix
# Layers beyond depth are inactive; dimensions beyond embed_dim receive no gradients.
```

### Fine-tuning procedure

- Epochs: up to `finetune_epochs = 30`, early stopping with `patience = 5`
- Criterion: `CrossEntropyLoss` (binary) or `BCEWithLogitsLoss` (multilabel)
- Top-k averaging: average metrics of the best `top_k_epochs = 3` validation epochs

### Output (updates `search_state` in-place)

```python
# Appended to search_state["completed_experiments"]:
{
    "embed_dim": int, "depth": int, "mlp_ratio": int, "num_heads": int,
    "num_params": int,   # exact parameter count
    "flops": int,        # estimated FLOPs at flops_seq_len=512
    # Validation metrics (top-k epoch averaged):
    "val_accuracy": float,
    "val_f1":       float,   # macro-averaged for multilabel
    "val_auroc":    float,
    "val_auprc":    float
}

# Updated best (composite rank = mean of per-metric ranks, lower = better):
search_state["best_proposal"]  # arch with min avg_rank across 4 val metrics
search_state["best_model_sd"]  # corresponding PyTorch state_dict
```

---

## Agent 3b — Strategy Agent

**File**: `agents/experiment_agent.py` (`decide_strategy`)  
**Calls per round**: 1 (after A3a, if budget_remaining > 0)  
**LLM backbone**: `anthropic/claude-3.5-haiku`

### Input

| Field | Type | Description |
|-------|------|-------------|
| `context` | dict | Context including Layer 2 prior (secondary signal) |
| `search_state` | dict | All completed experiments + best proposal + budget remaining |

### Prompt content (injected sections)

1. **Role**: search strategy agent for NAS
2. **Completed experiments**: all evaluated architectures with val metrics (sorted by composite rank)
3. **Best architecture so far**: current best by composite rank
4. **Budget remaining**: how many evaluations are left
5. **Layer 2 prior**: secondary signal — if preferred regions are exhausted, lean exploitation
6. **Decision guidance**:
   - `"exploration"` when: trajectory variance is high; preferred regions not yet covered; budget is ample
   - `"exploitation"` when: a stable best neighbourhood has emerged; budget is limited; Pareto frontier around best is under-explored

### Output

```json
{"strategy": "exploration", "rationale": "High variance in val_auprc (0.31–0.52); preferred embed_dim region not yet sampled at depth=4."}
```

or

```json
{"strategy": "exploitation", "rationale": "Best arch (embed_dim=128, depth=4) stable across last 5 evals. Budget=6 remaining — focus on neighbour refinement."}
```

| Field | Type | Values |
|-------|------|--------|
| `strategy` | str | `"exploration"` or `"exploitation"` |
| `rationale` | str | Brief free-text explanation |

> **First round**: always `{"strategy": "exploration", "rationale": "initial round"}` — no search history available.  
> **Parse failure fallback**: `{"strategy": "exploration", "rationale": "parse failure fallback"}`.

---

## Round Flow Summary

```
context (Layer 1 + Layer 2, assembled once)
search_state (grows with each evaluation)
strategy (A3b output from previous round)
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  A1 Propose ──► [{arch₁, rationale}, {arch₂, ...}, ...]        │
│       │                                                         │
│  FOR pass p in 1..3:                                            │
│    A2 Critique ──► accepted_p, rejected_p                      │
│    IF all_accepted OR only_duplicates: BREAK                    │
│    A1 Revise(rejected_p \ duplicates) ──► new proposals        │
│  END FOR                                                        │
│       │                                                         │
│  [Deduplicate within round]                                     │
│       │                                                         │
│  A3a Evaluate ──► val metrics → search_state (GPU-bound)       │
│       │                                                         │
│  A3b Strategy ──► {"strategy", "rationale"} for next round     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Termination Conditions

| Condition | Trigger | Action |
|-----------|---------|--------|
| Budget exhausted | `budget_remaining == 0` | Normal termination |
| Consecutive failures | 3 rounds produce no accepted proposals | Early termination |
| LLM API error | Network/auth failure | Increment failure counter; preserve results |

After termination, the architecture in `search_state["best_proposal"]` is evaluated on the **held-out test set** (single evaluation, no leakage).
