"""
Agent-in-the-loop task hooks.

:class:`TaskAgentHooks` is instantiated once per task lifecycle and
exposes four hook methods that are called at key points in
:class:`~starfish.controller.tasks.abstract_task.AbstractTask`.

Every hook:
- Returns ``None`` when the agent is disabled or the LLM is unavailable.
- Never raises — all exceptions are caught and logged.
- Appends ``[Agent]`` prefixed messages to the task logger so they
  appear in ``starfish run logs`` and the web UI automatically.
"""

from __future__ import annotations

import logging

from starfish.controller.agent.engine import (
    query_llm,
    is_agent_enabled,
    SUMMARY_MODEL,
    DECISION_MODEL,
)
from starfish.controller.agent.prompts import (
    POST_TRAINING_SYSTEM,
    PRE_AGGREGATION_SYSTEM,
    POST_AGGREGATION_SYSTEM,
    ON_FAILURE_SYSTEM,
    build_post_training_message,
    build_pre_aggregation_message,
    build_post_aggregation_message,
    build_on_failure_message,
)

module_logger = logging.getLogger(__name__)


class TaskAgentHooks:
    """
    Optional LLM hooks called during the AbstractTask lifecycle.

    Parameters
    ----------
    task_config : dict
        The ``config`` dict from the current task definition.
        Agent settings live under ``config.agent``.
    """

    def __init__(self, task_config: dict | None = None):
        task_config = task_config or {}
        agent_cfg = task_config.get("agent", {})
        self.enabled = bool(agent_cfg.get("enabled", False))
        self.early_stopping = bool(agent_cfg.get("early_stopping", False))
        self.summaries = bool(agent_cfg.get("summaries", False))
        self.outlier_detection = bool(agent_cfg.get("outlier_detection", False))

    # ------------------------------------------------------------------
    # Hook 1: after training(), before uploading mid-artifacts
    # ------------------------------------------------------------------

    def post_training(
        self,
        task_type: str,
        round_num: int,
        total_round: int,
        mid_artifacts: dict,
        task_logger: logging.Logger | None = None,
    ) -> dict | None:
        """
        Analyse per-site training output and generate a round summary.

        Called after ``training()`` succeeds and before ``notify(5)``.

        Returns
        -------
        dict or None
            ``{"summary": "...", "flag": "..." | null}``
        """
        if not self.enabled or not self.summaries:
            return None
        try:
            msg = build_post_training_message(
                task_type, round_num, total_round, mid_artifacts,
            )
            result = query_llm(POST_TRAINING_SYSTEM, msg, model=SUMMARY_MODEL)
            if result and task_logger:
                if result.get("summary"):
                    task_logger.info(
                        "[Agent] Round %d summary: %s",
                        round_num, result["summary"],
                    )
                if result.get("flag"):
                    task_logger.warning(
                        "[Agent] Warning: %s", result["flag"],
                    )
            return result
        except Exception as e:
            module_logger.warning("post_training hook failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Hook 2: before do_aggregate() (coordinator only)
    # ------------------------------------------------------------------

    def pre_aggregation(
        self,
        task_type: str,
        round_num: int,
        total_round: int,
        all_mid_artifacts: list[dict],
        task_logger: logging.Logger | None = None,
    ) -> dict | None:
        """
        Compare mid-artifacts across sites and flag outliers.

        Called before ``do_aggregate()`` on the coordinator.

        Returns
        -------
        dict or None
            ``{"action": "...", "reason": "...", "flagged_sites": [...], "summary": "..."}``
        """
        if not self.enabled or not self.outlier_detection:
            return None
        try:
            msg = build_pre_aggregation_message(
                task_type, round_num, total_round, all_mid_artifacts,
            )
            result = query_llm(
                PRE_AGGREGATION_SYSTEM, msg, model=DECISION_MODEL,
            )
            if result and task_logger:
                if result.get("summary"):
                    task_logger.info(
                        "[Agent] Pre-aggregation: %s", result["summary"],
                    )
                if result.get("flagged_sites"):
                    task_logger.warning(
                        "[Agent] Flagged sites: %s", result["flagged_sites"],
                    )
            return result
        except Exception as e:
            module_logger.warning("pre_aggregation hook failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Hook 3: after do_aggregate(), before round decision
    # ------------------------------------------------------------------

    def post_aggregation(
        self,
        task_type: str,
        round_num: int,
        total_round: int,
        aggregated_result: dict,
        round_history: list[dict] | None = None,
        task_logger: logging.Logger | None = None,
    ) -> dict | None:
        """
        Evaluate convergence and decide whether to stop early.

        Called after ``do_aggregate()`` succeeds on the coordinator.

        Returns
        -------
        dict or None
            ``{"converged": bool, "reason": "...", "convergence_score": float, "summary": "..."}``
        """
        if not self.enabled or not self.early_stopping:
            return None
        try:
            msg = build_post_aggregation_message(
                task_type, round_num, total_round,
                aggregated_result, round_history,
            )
            result = query_llm(
                POST_AGGREGATION_SYSTEM, msg, model=DECISION_MODEL,
            )
            if result and task_logger:
                if result.get("summary"):
                    task_logger.info(
                        "[Agent] Aggregation round %d: %s",
                        round_num, result["summary"],
                    )
                if result.get("converged"):
                    task_logger.info(
                        "[Agent] Early stopping recommended: %s",
                        result.get("reason", "model converged"),
                    )
            return result
        except Exception as e:
            module_logger.warning("post_aggregation hook failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Hook 4: on failure (any site)
    # ------------------------------------------------------------------

    def on_failure(
        self,
        task_type: str,
        task_config: dict,
        round_num: int,
        role: str,
        error_msg: str,
        logs: list[str],
        task_logger: logging.Logger | None = None,
    ) -> dict | None:
        """
        Diagnose a task failure and suggest recovery.

        Called from ``pending_failed()`` before notifying the router.

        Returns
        -------
        dict or None
            ``{"root_cause": "...", "category": "...", "severity": "...", "suggestion": "..."}``
        """
        if not self.enabled:
            return None
        try:
            msg = build_on_failure_message(
                task_type, task_config, round_num, role, error_msg, logs,
            )
            result = query_llm(ON_FAILURE_SYSTEM, msg, model=SUMMARY_MODEL)
            if result and task_logger:
                task_logger.info(
                    "[Agent] Failure diagnosis: %s — %s",
                    result.get("root_cause", "unknown"),
                    result.get("suggestion", ""),
                )
            return result
        except Exception as e:
            module_logger.warning("on_failure hook failed: %s", e)
            return None
