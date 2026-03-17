"""
LLM engine wrapper for the Router's embedded agent.

Provides a thin wrapper around the Anthropic SDK with:
- Graceful degradation (returns None on failure)
- Token budget enforcement
- Configurable model selection
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
AGGREGATION_MODEL = "claude-sonnet-4-6"
MAX_TOKENS_PER_CALL = 1024


def _get_client():
    """
    Create an Anthropic client if the SDK and API key are available.
    Returns None if unavailable (graceful degradation).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set, agent features disabled")
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    except ImportError:
        logger.debug("anthropic package not installed, agent features disabled")
        return None


def query_llm(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS_PER_CALL,
    agent_config: dict | None = None,
) -> dict | None:
    """
    Send a query to the LLM and return parsed JSON response.

    Parameters
    ----------
    system_prompt : str
        System prompt providing context and instructions.
    user_message : str
        The user message containing data for analysis.
    model : str
        Model to use for this query.
    max_tokens : int
        Maximum tokens for the response.
    agent_config : dict | None
        Project-level agent configuration. If None or not enabled, returns None.

    Returns
    -------
    dict | None
        Parsed JSON response from the LLM, or None if the call fails or is disabled.
    """
    if agent_config is None or not agent_config.get("enabled", False):
        return None

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

        # Try to parse as JSON
        text = text.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Agent LLM returned non-JSON response: %s", text[:200])
        return None
    except Exception as e:
        logger.warning("Agent LLM call failed: %s", str(e))
        return None


def is_agent_enabled(agent_config: dict | None, feature: str = None) -> bool:
    """
    Check if the agent is enabled, optionally for a specific feature.

    Parameters
    ----------
    agent_config : dict | None
        Project-level agent configuration.
    feature : str | None
        Specific feature to check (e.g., "aggregation", "triage", "scheduling").

    Returns
    -------
    bool
    """
    if agent_config is None:
        return False
    if not agent_config.get("enabled", False):
        return False
    if feature and not agent_config.get(feature, False):
        return False
    return True
