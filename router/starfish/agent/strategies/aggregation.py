"""
Adaptive aggregation advisor.

Analyzes mid-artifacts from all sites before aggregation and provides
advice on how to handle the aggregation (proceed, reweight, or exclude sites).
"""
from __future__ import annotations

import logging

from starfish.agent.engine import query_llm, is_agent_enabled, AGGREGATION_MODEL
from starfish.agent.prompts.aggregation import (
    AGGREGATION_SYSTEM_PROMPT,
    build_aggregation_message,
)

logger = logging.getLogger(__name__)

DEFAULT_ADVICE = {
    "action": "proceed",
    "reason": "Default behavior (agent not enabled or unavailable)",
    "flagged_sites": [],
    "details": {
        "sample_balance": "balanced",
        "coefficient_agreement": "consistent",
        "outlier_sites": [],
    },
}


def get_aggregation_advice(
    agent_config: dict,
    task_type: str,
    current_round: int,
    total_round: int,
    site_artifacts: list[dict],
) -> dict:
    """
    Get aggregation advice from the LLM agent.

    Parameters
    ----------
    agent_config : dict
        Project-level agent configuration.
    task_type : str
        The ML task type (e.g., "LogisticRegression").
    current_round : int
        Current training round.
    total_round : int
        Total planned rounds.
    site_artifacts : list[dict]
        Artifacts from each site with keys: site_name, site_id, artifacts.

    Returns
    -------
    dict
        Aggregation advice. Returns DEFAULT_ADVICE on failure.
    """
    if not is_agent_enabled(agent_config, "aggregation"):
        return DEFAULT_ADVICE

    message = build_aggregation_message(
        task_type, current_round, total_round, site_artifacts
    )

    result = query_llm(
        system_prompt=AGGREGATION_SYSTEM_PROMPT,
        user_message=message,
        model=AGGREGATION_MODEL,
        agent_config=agent_config,
    )

    if result is None:
        logger.info("Aggregation advisor returned no result, using defaults")
        return DEFAULT_ADVICE

    # Validate required fields
    if "action" not in result or "reason" not in result:
        logger.warning("Aggregation advice missing required fields: %s", result)
        return DEFAULT_ADVICE

    return result
