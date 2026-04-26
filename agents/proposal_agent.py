"""
Agent 1: Architecture Proposal (LLM-based)
============================================
Uses Claude API to propose transformer architectures based on:
- Search strategy (exploration or exploitation, set by Agent 3)
- Historical context (dataset similarity, top-k archs, SHAP importance)
- Current search state (tried archs + results)
- Computational constraint (max_params)
"""

import json
import time

from utils.tracer import get_tracer

CHOICES = {
    "mlp_ratio": [1, 2, 4, 8],
    "num_heads": [1, 2, 4, 8],
    "embed_dim": [16, 32, 64, 128],
    "depth": [1, 2, 4, 8],
}


def _build_prompt(context, search_state, max_params, strategy=None, max_flops=None):
    """Build the proposal prompt for Claude."""
    parts = []

    parts.append(
        "You are an expert neural architecture search agent for Transformer models "
        "applied to longitudinal Electronic Health Record (EHR) data. Your job is to "
        "propose architectures that will perform well on the target task.\n"
    )

    # Search space
    parts.append("## Search Space\n")
    parts.append(f"Available choices: {json.dumps(CHOICES)}\n")
    parts.append(
        "Each architecture is defined by four scalar hyperparameters:\n"
        f"- embed_dim: one value from {CHOICES['embed_dim']}\n"
        f"- depth: number of transformer layers, from {CHOICES['depth']}\n"
        f"- mlp_ratio: one value from {CHOICES['mlp_ratio']}, constant across all layers\n"
        f"- num_heads: one value from {CHOICES['num_heads']}, constant across all layers\n\n"
        "CONSTRAINT: embed_dim must be divisible by num_heads (all currently "
        "listed combos satisfy this — every embed_dim is a multiple of every "
        "num_heads, so any pairing is valid).\n"
    )

    # Parameter budget
    parts.append(f"\n## Parameter Budget\n")
    parts.append(f"Maximum allowed parameters: {max_params:,}\n")
    if max_flops is not None:
        parts.append(f"Maximum allowed FLOPs: {max_flops:,}\n")

    # Target dataset info
    target = context.get("target_summary", {})
    if target:
        parts.append(f"\n## Target Dataset\n")
        parts.append(f"Hospital: {target.get('hospital', 'unknown')}\n")
        for k, v in target.items():
            if k != "hospital":
                parts.append(f"  {k}: {v}\n")

    # Historical top-k architectures
    top_k = context.get("top_k_archs", [])
    if top_k:
        similar = context.get("similar_hospital", "unknown")
        matched_task = context.get("matched_task", "unknown")
        sim_score = context.get("similarity_score", 0)
        parts.append(
            f"\n## Historical Best Architectures\n"
            f"From most similar hospital: {similar} (similarity={sim_score:.4f}), "
            f"task: {matched_task}\n\n"
        )
        for i, arch in enumerate(top_k):
            parts.append(f"  Top-{i+1}: {json.dumps(arch)}\n")
    else:
        parts.append(
            "\n## Historical Data\n"
            "No historical architecture data available. This is a cold start — "
            "propose a diverse set of architectures for exploration.\n"
        )

    # SHAP importance
    shap = context.get("shap_importance", {})
    if shap:
        parts.append(f"\n## SHAP Feature Importance (from historical data)\n")
        parts.append(
            "These show which architecture features matter most for performance:\n"
        )
        for feat, val in sorted(shap.items(), key=lambda x: -x[1]):
            parts.append(f"  {feat}: {val:.4f}\n")
        parts.append(
            "\nUse this to guide your search: vary the most important features more.\n"
        )

    # Already tried architectures (only val metrics — no test leakage)
    completed = search_state.get("completed_experiments", [])
    if completed:
        parts.append(f"\n## Already Tried ({len(completed)} architectures)\n")
        parts.append(
            "Performance shown below is on the VALIDATION set.\n\n"
        )
        for exp in completed:
            parts.append(f"  {json.dumps(exp)}\n")
        parts.append(
            "\nAvoid proposing architectures that are identical or very similar "
            "to already-tried ones. Learn from what worked and what didn't.\n"
        )

    # Budget
    budget_remaining = search_state.get("budget_remaining", 0)
    parts.append(f"\n## Budget Remaining: {budget_remaining} architectures\n")
    parts.append(
        "\nDecide how many architectures to propose (at least 1, at most budget_remaining).\n"
    )

    # Search strategy
    if strategy:
        strat = strategy.get("strategy", "exploration")
        rationale = strategy.get("rationale", "")
        parts.append(
            f"\n## Search Strategy (set by the Strategy Agent)\n"
            f"Current strategy: **{strat}**\n"
        )
        if rationale:
            parts.append(f"Rationale: {rationale}\n")
        parts.append(
            "\n- When strategy is \"exploration\": propose diverse architectures covering "
            "different regions of the search space. Vary the most important SHAP features.\n"
            "- When strategy is \"exploitation\": propose architectures similar to the best "
            "performers, with small targeted variations to refine performance.\n"
        )

    # Output format
    parts.append(
        "\n## Output Format\n"
        "Return a JSON array of architecture proposals. Each proposal must have:\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "embed_dim": 128,\n'
        '    "depth": 4,\n'
        '    "mlp_ratio": 4,\n'
        '    "num_heads": 4,\n'
        '    "rationale": "Brief explanation of why this architecture"\n'
        "  }\n"
        "]\n"
        "```\n"
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )

    return "".join(parts)


def _parse_proposals(response_text):
    """Parse Claude's response into a list of proposal dicts."""
    text = response_text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        proposals = json.loads(text)
    except json.JSONDecodeError:
        # LLM may have added preamble text — extract the JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            proposals = json.loads(text[start:end + 1])
        else:
            raise

    if not isinstance(proposals, list):
        proposals = [proposals]
    return proposals


def _validate_proposal(proposal):
    """Basic validation of a single proposal."""
    embed_dim = proposal.get("embed_dim")
    depth = proposal.get("depth")
    mlp_ratio = proposal.get("mlp_ratio")
    num_heads = proposal.get("num_heads")

    if embed_dim not in CHOICES["embed_dim"]:
        return False, f"embed_dim {embed_dim} not in {CHOICES['embed_dim']}"
    if depth not in CHOICES["depth"]:
        return False, f"depth {depth} not in {CHOICES['depth']}"
    if mlp_ratio not in CHOICES["mlp_ratio"]:
        return False, f"mlp_ratio {mlp_ratio} not in {CHOICES['mlp_ratio']}"
    if num_heads not in CHOICES["num_heads"]:
        return False, f"num_heads {num_heads} not in {CHOICES['num_heads']}"
    if embed_dim % num_heads != 0:
        return False, f"embed_dim {embed_dim} not divisible by num_heads={num_heads}"

    return True, "ok"


def _call_llm(prompt, client, model, max_retries=5):
    """Call Claude and parse the response into a list of proposals."""
    for attempt in range(max_retries):
        try:
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

    # Trace LLM prompt/response
    tracer = get_tracer()
    if tracer:
        tracer.log_subsection("LLM Call")
        tracer.log_prompt(prompt)
        tracer.log_response(response_text)

    try:
        proposals = _parse_proposals(response_text)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse proposals: {e}")
        print(f"  Response: {response_text[:500]}")
        return []

    # Validate each proposal
    valid_proposals = []
    for i, prop in enumerate(proposals):
        ok, msg = _validate_proposal(prop)
        if ok:
            valid_proposals.append(prop)
            print(f"  Proposal {i+1}: embed_dim={prop['embed_dim']}, depth={prop['depth']}, "
                  f"mlp_ratio={prop['mlp_ratio']}, num_heads={prop['num_heads']}")
            if prop.get("rationale"):
                print(f"    Rationale: {prop['rationale']}")
        else:
            print(f"  Proposal {i+1} INVALID: {msg}")

    print(f"  {len(valid_proposals)}/{len(proposals)} proposals valid")
    return valid_proposals


def propose(context, search_state, max_params, client, model="claude-sonnet-4-6",
            strategy=None, max_flops=None):
    """
    Use Claude to propose new architectures.

    Returns:
        list of proposal dicts, each with embed_dim, depth, mlp_ratio, num_heads, rationale
    """
    strat_name = strategy.get("strategy", "exploration") if strategy else "exploration"
    print(f"\n[Agent 1: Architecture Proposal] (strategy={strat_name})")
    prompt = _build_prompt(context, search_state, max_params, strategy=strategy,
                           max_flops=max_flops)
    return _call_llm(prompt, client, model)


def _build_revision_prompt(context, search_state, rejected_with_critiques, max_params,
                           strategy=None, max_flops=None):
    """Build a revision prompt based on critic feedback."""
    parts = []

    parts.append(
        "You are an expert neural architecture search agent for Transformer models "
        "applied to longitudinal Electronic Health Record (EHR) data.\n\n"
        "Your previous proposals were REJECTED by the critic. "
        "You must revise them based on the feedback below.\n"
    )

    # Search space
    parts.append(f"\n## Search Space\n")
    parts.append(f"Available choices: {json.dumps(CHOICES)}\n")
    parts.append(
        "All four values (embed_dim, depth, mlp_ratio, num_heads) are scalars.\n"
        "CONSTRAINT: embed_dim must be divisible by num_heads.\n"
        f"Maximum allowed parameters: {max_params:,}\n"
    )
    if max_flops is not None:
        parts.append(f"Maximum allowed FLOPs: {max_flops:,}\n")

    # SHAP importance
    shap = context.get("shap_importance", {})
    if shap:
        parts.append(f"\n## SHAP Feature Importance\n")
        for feat, val in sorted(shap.items(), key=lambda x: -x[1]):
            parts.append(f"  {feat}: {val:.4f}\n")

    # Already tried
    completed = search_state.get("completed_experiments", [])
    if completed:
        parts.append(f"\n## Already Tried ({len(completed)} architectures)\n")
        for exp in completed:
            parts.append(f"  {json.dumps(exp)}\n")

    # Rejected proposals + critiques
    parts.append(f"\n## Rejected Proposals & Critique Reasons\n")
    for i, item in enumerate(rejected_with_critiques):
        parts.append(f"\n### Rejected Proposal {i+1}\n")
        parts.append(f"  Config: {json.dumps(item['proposal'])}\n")
        parts.append(f"  Critique: {item['critique']}\n")
        parts.append(f"  Risk tags: {item['risk_tags']}\n")

    # Strategy
    if strategy:
        strat = strategy.get("strategy", "exploration")
        rationale = strategy.get("rationale", "")
        parts.append(f"\n## Current Search Strategy: {strat}\n")
        if rationale:
            parts.append(f"Rationale: {rationale}\n")
        parts.append(
            "Keep the revisions aligned with this strategy.\n"
        )

    # Instructions
    parts.append(
        "\n## Your Task\n"
        "For each rejected proposal, provide a REVISED architecture that addresses "
        "the critique. Make sure the revision:\n"
        "- Fixes any constraint violations\n"
        "- Is sufficiently different from already-tried architectures\n"
        "- Respects the parameter budget\n"
        "- Addresses the specific feedback from the critic\n\n"
        "## Output Format\n"
        "Return a JSON array of revised architecture proposals:\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "embed_dim": 128,\n'
        '    "depth": 4,\n'
        '    "mlp_ratio": 4,\n'
        '    "num_heads": 4,\n'
        '    "rationale": "Revised to address: ..."\n'
        "  }\n"
        "]\n"
        "```\n"
        "Return ONLY valid JSON, no markdown code fences, no extra text.\n"
    )

    return "".join(parts)


def revise(context, search_state, rejected_with_critiques, max_params, client,
           model="claude-sonnet-4-6", strategy=None, max_flops=None):
    """
    Use Claude to revise rejected proposals based on critic feedback.

    Args:
        rejected_with_critiques: list of {"proposal": config, "critique": str, "risk_tags": [...]}

    Returns:
        list of revised proposal dicts
    """
    print("\n[Agent 1: Architecture Revision]")
    print(f"  Revising {len(rejected_with_critiques)} rejected proposals...")

    prompt = _build_revision_prompt(context, search_state, rejected_with_critiques, max_params,
                                    strategy=strategy, max_flops=max_flops)
    return _call_llm(prompt, client, model)
