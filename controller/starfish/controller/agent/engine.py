"""
LLM engine wrapper for the Controller's agent-in-the-loop hooks.

Thin wrapper around the Anthropic SDK with graceful degradation:
if the SDK or API key is unavailable, all calls return None and
the task lifecycle proceeds unchanged.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Use haiku for fast/cheap per-round summaries,
# sonnet for higher-stakes convergence decisions.
SUMMARY_MODEL = "claude-haiku-4-5-20251001"
DECISION_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024


def _get_client():
    """
    Create an Anthropic client if the SDK and API key are available.

    Returns None if unavailable, enabling graceful degradation so that
    tasks run identically to the non-agent path.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    except ImportError:
        return None


def query_llm(
    system_prompt: str,
    user_message: str,
    model: str = SUMMARY_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> dict | None:
    """
    Send a query to the LLM and parse the JSON response.

    Parameters
    ----------
    system_prompt : str
        System-level instructions for the LLM.
    user_message : str
        Data payload for analysis.
    model : str
        Which Claude model to use.
    max_tokens : int
        Maximum response tokens.

    Returns
    -------
    dict or None
        Parsed JSON from the LLM, or None on any failure.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Agent LLM returned non-JSON: %s", text[:200])
        return None
    except Exception as e:
        logger.warning("Agent LLM call failed: %s", e)
        return None


def is_agent_enabled(task_config: dict | None) -> bool:
    """
    Check whether agent hooks are enabled for this task.

    The agent block lives inside the task's ``config`` dict:

    .. code-block:: json

        {"agent": {"enabled": true, ...}}

    Parameters
    ----------
    task_config : dict or None
        The ``config`` dict from the current task definition.

    Returns
    -------
    bool
    """
    if not task_config:
        return False
    agent_cfg = task_config.get("agent")
    if not agent_cfg:
        return False
    return bool(agent_cfg.get("enabled", False))
