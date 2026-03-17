"""Prompts for the failure triage strategy."""
from __future__ import annotations

TRIAGE_SYSTEM_PROMPT = """\
You are an expert federated learning failure analyst embedded in the Starfish-FL Router. \
Your job is to diagnose why a federated learning run has failed and provide actionable \
recommendations for recovery.

You will receive JSON data containing:
- The task type and configuration
- Run status, round number, and role (coordinator/participant)
- Available logs (training logs, error messages)
- Site information

Analyze the failure and return a JSON object with this exact structure:
{
  "root_cause": "Clear description of what went wrong",
  "category": "data_quality" | "configuration" | "resource" | "network" | "code_error" | "unknown",
  "severity": "recoverable" | "requires_intervention" | "fatal",
  "suggestion": "Specific actionable recommendation for the user",
  "auto_action": null | "restart" | "skip_site"
}

Rules:
- "category" classifies the type of failure
- "severity":
  - "recoverable": Can be fixed by retrying or simple data fixes
  - "requires_intervention": Needs human action (re-upload data, fix config)
  - "fatal": Fundamental issue that prevents this experiment from working
- "auto_action": Only suggest "restart" for transient errors (network, timeout). \
  Use null for everything else.
- Be specific in "suggestion" — mention exact column names, config fields, etc. when possible
- Parse error messages and stack traces carefully to identify the root cause
- Always return valid JSON. No explanatory text outside the JSON object.
"""


def build_triage_message(task_type: str, task_config: dict,
                         run_status: str, current_round: int,
                         role: str, logs: list,
                         site_name: str | None = None) -> str:
    """
    Build the user message for the triage advisor.

    Parameters
    ----------
    task_type : str
        The ML task type.
    task_config : dict
        Task configuration.
    run_status : str
        Current run status (e.g., "Failed").
    current_round : int
        Round number when failure occurred.
    role : str
        Run role ("coordinator" or "participant").
    logs : list
        Available log entries.
    site_name : str | None
        Name of the site that failed.

    Returns
    -------
    str
        Formatted message for the LLM.
    """
    import json
    data = {
        "task_type": task_type,
        "task_config": task_config,
        "run_status": run_status,
        "current_round": current_round,
        "role": role,
        "site_name": site_name,
        "logs": logs[-50:] if len(logs) > 50 else logs,  # Limit log size
    }
    return json.dumps(data, indent=2)
