"""
Smart scheduling advisor.

Evaluates convergence after each aggregation round and recommends
whether to continue training or stop early.
"""
from __future__ import annotations

import logging

from starfish.agent.engine import query_llm, is_agent_enabled, DEFAULT_MODEL
from starfish.agent.prompts.scheduling import (
    SCHEDULING_SYSTEM_PROMPT,
    build_scheduling_message,
)

logger = logging.getLogger(__name__)

DEFAULT_SCHEDULING = {
    "continue": True,
    "reason": "Default behavior (agent not enabled or unavailable)",
    "suggested_rounds_remaining": None,
    "convergence_score": 0.0,
}


def get_scheduling_advice(
    agent_config: dict,
    task_type: str,
    current_round: int,
    total_round: int,
    current_metrics: dict,
    previous_metrics: list[dict] | None = None,
) -> dict:
    """
    Get scheduling advice from the LLM agent.

    Parameters
    ----------
    agent_config : dict
        Project-level agent configuration.
    task_type : str
        The ML task type.
    current_round : int
        Current training round.
    total_round : int
        Total planned rounds.
    current_metrics : dict
        Metrics from the current aggregation round.
    previous_metrics : list[dict] | None
        Metrics from previous rounds.

    Returns
    -------
    dict
        Scheduling advice. Returns DEFAULT_SCHEDULING on failure.
    """
    if not is_agent_enabled(agent_config, "scheduling"):
        return DEFAULT_SCHEDULING

    # First round: always continue
    if current_round <= 1:
        return {
            "continue": True,
            "reason": "First round — not enough data to assess convergence",
            "suggested_rounds_remaining": total_round - current_round,
            "convergence_score": 0.0,
        }

    message = build_scheduling_message(
        task_type, current_round, total_round,
        current_metrics, previous_metrics,
    )

    result = query_llm(
        system_prompt=SCHEDULING_SYSTEM_PROMPT,
        user_message=message,
        model=DEFAULT_MODEL,
        agent_config=agent_config,
    )

    if result is None:
        logger.info("Scheduling advisor returned no result, using defaults")
        return DEFAULT_SCHEDULING

    if "continue" not in result:
        logger.warning("Scheduling advice missing 'continue' field: %s", result)
        return DEFAULT_SCHEDULING

    return result
