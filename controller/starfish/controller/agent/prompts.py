"""
Prompt templates for the four agent hook points.

Each function returns ``(system_prompt, user_message)`` ready for
:func:`engine.query_llm`.  All prompts instruct the LLM to return
strict JSON so the caller can parse the response mechanically.
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Post-training hook (called per-site after training())
# ---------------------------------------------------------------------------

POST_TRAINING_SYSTEM = """\
You are an expert federated learning analyst. You are inspecting the \
training output from a single site after one round of local training.

Return a JSON object:
{
  "summary": "1-2 sentence summary of this site's training results",
  "flag": null | "description of any concern (anomalous coefficients, high VIF, etc.)"
}

Rules:
- Be concise. The summary will be appended to task logs.
- Only set "flag" when something is clearly problematic.
- Always return valid JSON only, no extra text.
"""


def build_post_training_message(
    task_type: str, round_num: int, total_round: int,
    mid_artifacts: dict,
) -> str:
    """Build user message for the post-training hook."""
    return json.dumps({
        "task_type": task_type,
        "round": round_num,
        "total_round": total_round,
        "mid_artifacts": mid_artifacts,
    }, indent=2)


# ---------------------------------------------------------------------------
# Pre-aggregation hook (coordinator, before do_aggregate())
# ---------------------------------------------------------------------------

PRE_AGGREGATION_SYSTEM = """\
You are an expert federated learning analyst inspecting mid-artifacts \
from all participating sites before the coordinator aggregates them.

Return a JSON object:
{
  "action": "proceed" | "reweight" | "exclude_sites",
  "reason": "brief explanation",
  "flagged_sites": [],
  "summary": "1-2 sentence overview of cross-site comparison"
}

Rules:
- "proceed": all sites look normal
- "reweight": large sample-size imbalance (>10x ratio)
- "exclude_sites": clearly anomalous results — list indices in flagged_sites
- Be conservative; some variation is normal in FL.
- Always return valid JSON only, no extra text.
"""


def build_pre_aggregation_message(
    task_type: str, round_num: int, total_round: int,
    all_mid_artifacts: list[dict],
) -> str:
    """Build user message for the pre-aggregation hook."""
    return json.dumps({
        "task_type": task_type,
        "round": round_num,
        "total_round": total_round,
        "site_artifacts": all_mid_artifacts,
    }, indent=2)


# ---------------------------------------------------------------------------
# Post-aggregation hook (coordinator, after do_aggregate())
# ---------------------------------------------------------------------------

POST_AGGREGATION_SYSTEM = """\
You are an expert federated learning analyst evaluating whether a \
model has converged after an aggregation round. You receive the \
aggregated result plus history from previous rounds.

Return a JSON object:
{
  "converged": true | false,
  "reason": "brief explanation of convergence assessment",
  "convergence_score": <float 0.0-1.0>,
  "summary": "1-3 sentence round summary suitable for task logs"
}

Rules:
- On round 1, always set converged=false (not enough history).
- Look for coefficient stability, loss plateau, stable diagnostics.
- Be conservative: prefer continuing over premature stopping.
- Always return valid JSON only, no extra text.
"""


def build_post_aggregation_message(
    task_type: str, round_num: int, total_round: int,
    aggregated_result: dict, round_history: list[dict] | None = None,
) -> str:
    """Build user message for the post-aggregation hook."""
    return json.dumps({
        "task_type": task_type,
        "round": round_num,
        "total_round": total_round,
        "aggregated_result": aggregated_result,
        "previous_rounds": round_history or [],
    }, indent=2)


# ---------------------------------------------------------------------------
# On-failure hook (any site, after an exception)
# ---------------------------------------------------------------------------

ON_FAILURE_SYSTEM = """\
You are an expert federated learning failure analyst. A task has \
failed during execution. Diagnose the root cause from the logs and \
context provided.

Return a JSON object:
{
  "root_cause": "clear description of what went wrong",
  "category": "data_quality" | "configuration" | "resource" | "network" | "code_error" | "unknown",
  "severity": "recoverable" | "requires_intervention" | "fatal",
  "suggestion": "specific actionable recommendation"
}

Rules:
- Be specific: mention column names, config fields, etc. when possible.
- "recoverable" = retry or simple fix; "requires_intervention" = human needed.
- Always return valid JSON only, no extra text.
"""


def build_on_failure_message(
    task_type: str, task_config: dict, round_num: int,
    role: str, error_msg: str, logs: list[str],
) -> str:
    """Build user message for the on-failure hook."""
    return json.dumps({
        "task_type": task_type,
        "task_config": task_config,
        "round": round_num,
        "role": role,
        "error": error_msg,
        "logs": logs[-50:] if len(logs) > 50 else logs,
    }, indent=2)
