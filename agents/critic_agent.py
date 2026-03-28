"""
Agent 3: Proposal Critic (LLM-based)
======================================
Uses Claude API to critique architecture proposals:
- Validates parameter constraints
- Checks for redundancy with already-tried architectures
- Evaluates alignment with SHAP insights
- Can revise proposals or accept them as-is
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_pipeline import count_subnet_params

CHOICES = {
    "mlp_ratio": [2, 4, 8],
    "num_heads": [2, 4, 8],
    "embed_dim": [64, 128, 256],
    "depth": [2, 4, 8],
}


def _build_prompt(context, search_state, proposals, max_params):
    """Build the critique prompt for Claude."""
    parts = []

    parts.append(
        "You are an expert architecture critic for Transformer-based NAS on EHR data. "
        "Your job is to review proposed architectures and either accept or revise them.\n"
    )

    # Search space & constraints
    parts.append(f"\n## Search Space\n{json.dumps(CHOICES)}\n")
    parts.append(
        "CONSTRAINT: embed_dim must be divisible by each num_heads value.\n"
        f"Maximum parameters: {max_params:,}\n"
    )

    # Historical context
    top_k = context.get("top_k_archs", [])
    if top_k:
        parts.append(f"\n## Historical Best Architectures (from similar hospital)\n")
        for i, arch in enumerate(top_k):
            parts.append(f"  Top-{i+1}: {json.dumps(arch)}\n")

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

    # Proposals to review
    parts.append(f"\n## Proposals to Review\n")
    for i, prop in enumerate(proposals):
        parts.append(f"  Proposal {i}: {json.dumps(prop)}\n")

    # Instructions
    parts.append(
        "\n## Your Task\n"
        "Review each proposal. For each one, decide: accept or revise.\n\n"
        "Check for:\n"
        "1. Does the architecture respect embed_dim % num_heads == 0 for all layers?\n"
        "2. Is it too similar to an already-tried architecture? (same embed_dim, depth, "
        "and >75% overlap in mlp_ratio/num_heads values)\n"
        "3. Does it leverage SHAP insights? (if depth has high importance, "
        "is the proposal exploring depth variation?)\n"
        "4. Is the proposal diverse enough relative to other proposals in this batch?\n\n"
        "If revising, provide a corrected config that satisfies all constraints.\n\n"
        "## Output Format\n"
        "Return a JSON array. Each element:\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "proposal_idx": 0,\n'
        '    "decision": "accept",\n'
        '    "revised_config": null,\n'
        '    "critique": "Looks good, explores high-SHAP depth dimension",\n'
        '    "risk_tags": []\n'
        "  },\n"
        "  {\n"
        '    "proposal_idx": 1,\n'
        '    "decision": "revise",\n'
        '    "revised_config": {"embed_dim": 128, "depth": 4, "mlp_ratio": [4,4,8,4], "num_heads": [4,4,4,4]},\n'
        '    "critique": "Original had embed_dim=64 with num_heads=8, changed to valid combo",\n'
        '    "risk_tags": ["constraint_violation"]\n'
        "  }\n"
        "]\n"
        "```\n"
        'Valid risk_tags: "too_large", "too_similar", "constraint_violation", '
        '"unexplored_region", "ignores_shap"\n\n'
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
    return json.loads(text)


def _validate_config(config):
    """Validate a config dict."""
    embed_dim = config.get("embed_dim")
    depth = config.get("depth")
    mlp_ratio = config.get("mlp_ratio", [])
    num_heads = config.get("num_heads", [])

    if embed_dim not in CHOICES["embed_dim"]:
        return False
    if depth not in CHOICES["depth"]:
        return False
    if len(mlp_ratio) != depth or len(num_heads) != depth:
        return False
    for mr in mlp_ratio:
        if mr not in CHOICES["mlp_ratio"]:
            return False
    for nh in num_heads:
        if nh not in CHOICES["num_heads"]:
            return False
        if embed_dim % nh != 0:
            return False
    return True


def critique(context, search_state, proposals, max_params, client,
             vocab_size=None, max_adm=8, model="claude-sonnet-4-6"):
    """
    Use Claude to critique architecture proposals.

    Args:
        context: dict from context_agent.run()
        search_state: dict with completed_experiments and budget_remaining
        proposals: list of proposal dicts from proposal_agent
        max_params: max parameter count
        client: anthropic.Anthropic client
        vocab_size: vocabulary size for param counting
        max_adm: max admissions for param counting
        model: Claude model ID

    Returns:
        list of accepted/revised proposal dicts (only configs that passed critique)
    """
    print("\n[Agent 3: Proposal Critic]")

    if not proposals:
        print("  No proposals to critique.")
        return []

    prompt = _build_prompt(context, search_state, proposals, max_params)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = response.content[0].text
    print(f"  Raw LLM response length: {len(response_text)} chars")

    try:
        critiques = _parse_critiques(response_text)
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse critiques: {e}")
        print(f"  Response: {response_text[:500]}")
        # Fall back: accept all proposals as-is
        print("  Falling back: accepting all proposals without critique")
        return proposals

    # Process critiques
    reviewed = []
    for crit in critiques:
        idx = crit.get("proposal_idx", -1)
        decision = crit.get("decision", "accept")
        revised = crit.get("revised_config")
        critique_text = crit.get("critique", "")
        risk_tags = crit.get("risk_tags", [])

        if decision == "revise" and revised:
            config = revised
            source = "revised"
        elif 0 <= idx < len(proposals):
            config = proposals[idx]
            source = "accepted"
        else:
            print(f"  Critique for invalid index {idx}, skipping")
            continue

        # Final validation
        if not _validate_config(config):
            print(f"  Proposal {idx} ({source}): INVALID config, skipping")
            continue

        # Check param count if vocab_size is known
        if vocab_size is not None:
            internal_config = {
                "embed_dim": [config["embed_dim"]] * config["depth"],
                "layer_num": config["depth"],
                "mlp_ratio": config["mlp_ratio"],
                "num_heads": config["num_heads"],
            }
            n_params = count_subnet_params(internal_config, vocab_size, max_adm=max_adm)
            if n_params > max_params:
                print(f"  Proposal {idx} ({source}): {n_params:,} params > {max_params:,}, skipping")
                continue

        print(f"  Proposal {idx} ({source}): {critique_text[:80]}  tags={risk_tags}")

        # Strip rationale, keep only config keys
        final = {
            "embed_dim": config["embed_dim"],
            "depth": config["depth"],
            "mlp_ratio": config["mlp_ratio"],
            "num_heads": config["num_heads"],
        }
        reviewed.append(final)

    print(f"  {len(reviewed)}/{len(proposals)} proposals passed critique")
    return reviewed
