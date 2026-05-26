"""
Agent 2: Proposal Critic (LLM-based)
======================================
Uses Claude API to critique architecture proposals:
- Validates parameter constraints
- Checks for redundancy with already-tried architectures
- Evaluates alignment with SHAP insights
- Adjusts review criteria based on current search strategy (exploration/exploitation)
- Returns accepted proposals + rejected proposals with critique reasons
  (rejected proposals are sent back to Agent 1 for revision)
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.search_space import CHOICES
from run_pipeline import count_subnet_params, count_subnet_flops
from utils.tracer import get_tracer
from utils.llm_counter import increment as _llm_increment


def _np_default(obj):
    """numpy-aware default= for json.dumps. See agents/experiment_agent.py."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_prompt(context, search_state, proposals, max_params, strategy=None,
                  max_flops=None):
    """Build the critique prompt for Claude."""
    parts = []

    parts.append(
        "You are an expert architecture critic for Transformer-based NAS on EHR data. "
        "Your job is to review proposed architectures and either ACCEPT or REJECT them.\n"
        "You do NOT revise proposals yourself — rejected proposals will be sent back "
        "to the proposal agent for revision.\n"
    )

    # Search space & constraints
    parts.append(f"\n## Search Space\n{json.dumps(CHOICES, default=_np_default)}\n")
    parts.append(
        "All four values (embed_dim, depth, mlp_ratio, num_heads) are scalars.\n"
        "CONSTRAINT: embed_dim must be divisible by num_heads.\n"
        f"Maximum parameters: {max_params:,}\n"
    )
    if max_flops is not None:
        parts.append(f"Maximum FLOPs (reference flops_seq_len): {max_flops:,}\n")

    # Historical context
    top_k = context.get("top_k_archs", [])
    if top_k:
        parts.append(f"\n## Historical Best Architectures (from similar hospital)\n")
        for i, arch in enumerate(top_k):
            parts.append(f"  Top-{i+1}: {json.dumps(arch, default=_np_default)}\n")

    # Layer 2: Architecture Prior (meta-regression) — soft directional guidance
    arch_prior = context.get("meta_regression_prior") or {}
    if arch_prior:
        parts.append("\n## Architecture Prior (SOFT directional guidance from meta-regression)\n")
        order = arch_prior.get("feature_importance_order", [])
        if order:
            parts.append(f"Feature importance: {' > '.join(order)}\n")
        preferred = arch_prior.get("preferred_levels", {}) or {}
        discouraged = arch_prior.get("discouraged_levels", {}) or {}
        confidence = arch_prior.get("confidence", {}) or {}
        if any(preferred.values()):
            parts.append("Preferred levels (positive cross-hospital SHAP):\n")
            for feat in order or list(preferred.keys()):
                lvls = preferred.get(feat, [])
                if lvls:
                    parts.append(f"  {feat}: {lvls}    [confidence: {confidence.get(feat, 'unknown')}]\n")
        if any(discouraged.values()):
            parts.append("Discouraged levels (negative cross-hospital SHAP):\n")
            for feat in order or list(discouraged.keys()):
                lvls = discouraged.get(feat, [])
                if lvls:
                    parts.append(f"  {feat}: {lvls}    [confidence: {confidence.get(feat, 'unknown')}]\n")

    # Already tried
    completed = search_state.get("completed_experiments", [])
    if completed:
        parts.append(f"\n## Already Tried ({len(completed)} architectures)\n")
        for exp in completed:
            parts.append(f"  {json.dumps(exp, default=_np_default)}\n")

    # Proposals to review
    parts.append(f"\n## Proposals to Review\n")
    for i, prop in enumerate(proposals):
        parts.append(f"  Proposal {i}: {json.dumps(prop, default=_np_default)}\n")

    # Strategy
    if strategy:
        strat = strategy.get("strategy", "exploration")
        rationale = strategy.get("rationale", "")
        parts.append(f"\n## Current Search Strategy: {strat}\n")
        if rationale:
            parts.append(f"Rationale: {rationale}\n")
        parts.append(
            "\nAdjust your review criteria based on the strategy:\n"
            "- During \"exploration\": be more tolerant of proposals that differ from "
            "historical best architectures. Prioritize diversity and coverage of the "
            "search space. Reject only for hard constraint violations or exact duplicates.\n"
            "- During \"exploitation\": be more tolerant of proposals similar to the best "
            "performers. Reject proposals that deviate too far from the proven good region.\n"
        )

    # Instructions — bifurcate HARD constraints vs SOFT signals
    parts.append(
        "\n## Your Task\n"
        "Review each proposal. For each one, decide: ACCEPT or REJECT.\n"
        "Do NOT provide revised configs — just explain what is wrong so the "
        "proposal agent can fix it.\n\n"
        "**Two categories of constraints**:\n\n"
        "### 1. HARD constraints — auto-reject if violated\n"
        "  - max_params budget exceeded (>2M unless overridden)\n"
        "  - embed_dim NOT divisible by num_heads\n"
        "  - exact duplicate of an already-tried architecture\n"
        "  - exact duplicate of a Top-k historical architecture (proposal duplicates known evidence)\n"
        "\n"
        "### 2. SOFT signals — flag and require justification, but do NOT auto-reject\n"
        "  - Discouraged level used (e.g. embed_dim=32 listed as discouraged)\n"
        "  - Avoid_combination interaction used (from Architecture Prior)\n"
        "  - Doesn't explore high-importance features (e.g. only varies low-importance num_heads)\n"
        "\n"
        "**For SOFT signals**:\n"
        "  - Check the proposal's `rationale` field for explicit justification.\n"
        "  - If rationale is convincing (e.g. \"exploring under-represented region\", "
        "\"complementary to Top-k coverage\", \"specific to data property X\") → ACCEPT.\n"
        "  - If rationale is absent or weak (e.g. \"just trying it\") → REJECT with critique:\n"
        "    \"Proposal uses discouraged <level> without justification. Please explain why "
        "this might work despite cross-hospital evidence of negative SHAP contribution.\"\n"
        "  - Goal: encourage informed deviation, discourage blind violation.\n"
        "\n"
        "## Output Format\n"
        "Return a JSON array. Each element:\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "proposal_idx": 0,\n'
        '    "decision": "accept",\n'
        '    "critique": "Looks good, explores high-importance embed_dim dimension",\n'
        '    "risk_tags": []\n'
        "  },\n"
        "  {\n"
        '    "proposal_idx": 1,\n'
        '    "decision": "reject",\n'
        '    "critique": "embed_dim=64 with num_heads=8 is valid but too similar to already-tried arch #3 (HARD: too_similar).",\n'
        '    "risk_tags": ["too_similar"]\n'
        "  },\n"
        "  {\n"
        '    "proposal_idx": 2,\n'
        '    "decision": "reject",\n'
        '    "critique": "embed_dim=32 is in discouraged_levels (SOFT) but rationale only says \"exploring small\". Please justify why this might work despite negative cross-hospital SHAP.",\n'
        '    "risk_tags": ["uses_discouraged_level"]\n'
        "  }\n"
        "]\n"
        "```\n"
        'Valid risk_tags:\n'
        '  HARD: "too_large", "too_similar", "constraint_violation"\n'
        '  SOFT: "uses_discouraged_level", "uses_avoid_combination", "ignores_high_importance_features"\n\n'
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )

    return "".join(parts)


def _parse_critiques(response_text):
    """Parse Claude's critique response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = text.find("[")
        if start != -1:
            result = decoder.raw_decode(text, start)[0]
        else:
            raise
    if isinstance(result, dict):
        result = [result]
    return result


def _validate_config(config):
    """Validate a config dict."""
    embed_dim = config.get("embed_dim")
    depth = config.get("depth")
    mlp_ratio = config.get("mlp_ratio")
    num_heads = config.get("num_heads")

    if embed_dim not in CHOICES["embed_dim"]:
        return False
    if depth not in CHOICES["depth"]:
        return False
    if mlp_ratio not in CHOICES["mlp_ratio"]:
        return False
    if num_heads not in CHOICES["num_heads"]:
        return False
    if embed_dim % num_heads != 0:
        return False
    return True


def critique(context, search_state, proposals, max_params, client,
             vocab_size=None, max_adm=8, model="google/gemini-2.5-flash-lite", strategy=None,
             max_flops=None, flops_seq_len=512, num_classes=2,
             already_accepted=None):
    """
    Use Claude to critique architecture proposals.

    Args:
        num_classes: head output dimensionality used for parameter counting
                     (2 for binary tasks, 18 for multilabel phenotypes).
        already_accepted: list of config dicts already accepted in the
                          current round (earlier passes).  Used to reject
                          revised proposals that converge to an already-
                          accepted architecture within the same round.

    Returns:
        (accepted, rejected_with_critiques)
        - accepted: list of config dicts ready for Agent 4
        - rejected_with_critiques: list of {"proposal": config, "critique": str, "risk_tags": [...]}
    """
    print("\n[Agent 2: Proposal Critic]")

    if not proposals:
        print("  No proposals to critique.")
        return [], []

    # Programmatic dedup safety net: the LLM critic is *prompted* to reject
    # duplicates of already-tried architectures, but Gemini Flash Lite (and any
    # LLM under load) can miss matches. Build a tuple-set of completed configs
    # once up-front; every accept path below will check membership before
    # admitting a proposal. Tuple order matches the 4-dim search-space schema.
    # Also includes architectures already accepted in the current round's
    # earlier passes to prevent within-round duplication.
    completed_tuples: set[tuple[int, int, int, int]] = {
        (c["embed_dim"], c["depth"], c["mlp_ratio"], c["num_heads"])
        for c in search_state.get("completed_experiments", [])
        if all(k in c for k in ("embed_dim", "depth", "mlp_ratio", "num_heads"))
    }
    for c in (already_accepted or []):
        if all(k in c for k in ("embed_dim", "depth", "mlp_ratio", "num_heads")):
            completed_tuples.add(
                (c["embed_dim"], c["depth"], c["mlp_ratio"], c["num_heads"]))

    def _is_completed_dupe(cfg: dict) -> bool:
        return (cfg["embed_dim"], cfg["depth"],
                cfg["mlp_ratio"], cfg["num_heads"]) in completed_tuples

    prompt = _build_prompt(context, search_state, proposals, max_params, strategy=strategy,
                           max_flops=max_flops)

    max_parse_retries = 1
    critiques = None
    for parse_attempt in range(max_parse_retries + 1):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                _llm_increment()
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  API error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        response_text = response.content[0].text
        print(f"  Raw LLM response length: {len(response_text)} chars")

        tracer = get_tracer()
        if tracer:
            tracer.log_subsection("LLM Call")
            tracer.log_prompt(prompt)
            tracer.log_response(response_text)

        try:
            critiques = _parse_critiques(response_text)
            break
        except json.JSONDecodeError as e:
            if parse_attempt < max_parse_retries:
                print(f"  Parse failed (attempt {parse_attempt+1}), retrying LLM call...")
                continue
            print(f"  ERROR: Failed to parse critiques after {max_parse_retries+1} attempts: {e}")
            print(f"  Response: {response_text[:500]}")

    if critiques is None:
        print("  Falling back: accepting all proposals without critique")
        accepted = []
        rejected_with_critiques = []
        for prop in proposals:
            config = {"embed_dim": prop["embed_dim"], "depth": prop["depth"],
                      "mlp_ratio": prop["mlp_ratio"], "num_heads": prop["num_heads"]}
            if _is_completed_dupe(config):
                print(f"  Proposal {prop.get('proposal_idx', '?')} [REJECTED]: "
                      f"duplicate of already-tried arch (parse-fail fallback)")
                rejected_with_critiques.append({
                    "proposal": config,
                    "critique": "Exact duplicate of a completed architecture "
                                "(LLM critic JSON parse failed; filtered by "
                                "post-LLM dedup safety net).",
                    "risk_tags": ["duplicate", "critic_parse_fail"],
                })
                continue
            accepted.append(config)
        return accepted, rejected_with_critiques

    # Process critiques
    accepted = []
    rejected_with_critiques = []

    for crit in critiques:
        idx = crit.get("proposal_idx", -1)
        decision = crit.get("decision", "accept")
        critique_text = crit.get("critique", "")
        risk_tags = crit.get("risk_tags", [])

        if not (0 <= idx < len(proposals)):
            print(f"  Critique for invalid index {idx}, skipping")
            continue

        proposal = proposals[idx]
        config = {
            "embed_dim": proposal["embed_dim"],
            "depth": proposal["depth"],
            "mlp_ratio": proposal["mlp_ratio"],
            "num_heads": proposal["num_heads"],
        }

        if decision == "reject":
            print(f"  Proposal {idx} [REJECTED]: {critique_text[:80]}  tags={risk_tags}")
            rejected_with_critiques.append({
                "proposal": config,
                "critique": critique_text,
                "risk_tags": risk_tags,
            })
            continue

        # Accept — do final validation
        if not _validate_config(config):
            print(f"  Proposal {idx} [REJECTED] because [INVALID] config (but [ACCEPTED] by LLM)")
            rejected_with_critiques.append({
                "proposal": config,
                "critique": "Invalid config: constraint violation detected by validator",
                "risk_tags": ["constraint_violation"],
            })
            continue

        # Programmatic dedup safety net (HIGH-1): the LLM critic is prompted
        # to reject duplicates but can miss them — especially on Gemini Flash
        # Lite. Reject any LLM-accepted config that matches an
        # already-completed experiment to avoid spending budget on a re-eval.
        if _is_completed_dupe(config):
            print(f"  Proposal {idx} [REJECTED] because duplicate of completed arch "
                  f"(but [ACCEPTED] by LLM)")
            rejected_with_critiques.append({
                "proposal": config,
                "critique": "Exact duplicate of a completed architecture "
                            "(LLM critic missed it; filtered by post-LLM "
                            "dedup safety net).",
                "risk_tags": ["duplicate"],
            })
            continue

        # Check param count if vocab_size is known
        if vocab_size is not None:
            internal_config = {
                "embed_dim": [config["embed_dim"]] * config["depth"],
                "layer_num": config["depth"],
                "mlp_ratio": [config["mlp_ratio"]] * config["depth"],
                "num_heads": [config["num_heads"]] * config["depth"],
            }
            n_params = count_subnet_params(internal_config, vocab_size,
                                            num_classes=num_classes, max_adm=max_adm)
            if n_params > max_params:
                print(f"  Proposal {idx} [REJECTED] because {n_params:,} > {max_params:,} (but [ACCEPTED] by LLM)")
                rejected_with_critiques.append({
                    "proposal": config,
                    "critique": f"Exceeds parameter budget: {n_params:,} > {max_params:,}",
                    "risk_tags": ["too_large"],
                })
                continue

            if max_flops is not None:
                n_flops = count_subnet_flops(internal_config, flops_seq_len)
                if n_flops > max_flops:
                    print(f"  Proposal {idx} [REJECTED] because FLOPs {n_flops:,} > {max_flops:,} (but [ACCEPTED] by LLM)")
                    rejected_with_critiques.append({
                        "proposal": config,
                        "critique": f"Exceeds FLOPs budget: {n_flops:,} > {max_flops:,}",
                        "risk_tags": ["too_large"],
                    })
                    continue

        print(f"  Proposal {idx} [ACCEPTED]: {critique_text[:80]}")
        accepted.append(config)

    print(f"  {len(accepted)} [ACCEPTED], {len(rejected_with_critiques)} [REJECTED]")
    return accepted, rejected_with_critiques
