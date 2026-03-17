"""
Failure triage advisor.

Diagnoses why a run failed and provides actionable recommendations.
"""
from __future__ import annotations

import logging

from starfish.agent.engine import query_llm, is_agent_enabled, DEFAULT_MODEL
from starfish.agent.prompts.triage import (
    TRIAGE_SYSTEM_PROMPT,
    build_triage_message,
)

logger = logging.getLogger(__name__)

DEFAULT_DIAGNOSIS = {
    "root_cause": "Unknown — agent not enabled or unavailable",
    "category": "unknown",
    "severity": "requires_intervention",
    "suggestion": "Check run logs manually for error details",
    "auto_action": None,
}


def get_failure_diagnosis(
    agent_config: dict,
    task_type: str,
    task_config: dict,
    run_status: str,
    current_round: int,
    role: str,
    logs: list,
    site_name: str | None = None,
) -> dict:
    """
    Get failure diagnosis from the LLM agent.

    Parameters
    ----------
    agent_config : dict
        Project-level agent configuration.
    task_type : str
        The ML task type.
    task_config : dict
        Task configuration.
    run_status : str
        Current run status.
    current_round : int
        Round when failure occurred.
    role : str
        Run role ("coordinator" or "participant").
    logs : list
        Available log entries.
    site_name : str | None
        Name of the site that failed.

    Returns
    -------
    dict
        Failure diagnosis. Returns DEFAULT_DIAGNOSIS on failure.
    """
    if not is_agent_enabled(agent_config, "triage"):
        return DEFAULT_DIAGNOSIS

    message = build_triage_message(
        task_type, task_config, run_status,
        current_round, role, logs, site_name,
    )

    result = query_llm(
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        user_message=message,
        model=DEFAULT_MODEL,
        agent_config=agent_config,
    )

    if result is None:
        logger.info("Triage advisor returned no result, using defaults")
        return DEFAULT_DIAGNOSIS

    if "root_cause" not in result or "category" not in result:
        logger.warning("Triage diagnosis missing required fields: %s", result)
        return DEFAULT_DIAGNOSIS

    return result
