"""Prompts for the scheduling advisor strategy."""
from __future__ import annotations

SCHEDULING_SYSTEM_PROMPT = """\
You are an expert federated learning scheduling advisor embedded in the Starfish-FL Router. \
Your job is to analyze training progress after each aggregation round and recommend whether \
to continue training or stop early because the model has converged.

You will receive JSON data containing:
- The task type (e.g., LogisticRegression, CoxProportionalHazards)
- Current round and total planned rounds
- Metrics from the current round and previous rounds (if available)
- Aggregated results summary

Analyze convergence and return a JSON object with this exact structure:
{
  "continue": true | false,
  "reason": "Brief explanation",
  "suggested_rounds_remaining": <integer or null>,
  "convergence_score": <float 0.0 to 1.0>
}

Rules:
- "continue": true means training should proceed to the next round
- "continue": false means the model has converged and training can stop
- "convergence_score": 0.0 = no convergence, 1.0 = fully converged
- "suggested_rounds_remaining": Your estimate of how many more rounds are needed (null if uncertain)
- On the first round, always recommend continuing (not enough data to judge convergence)
- Look for: coefficient stability, decreasing loss, stable diagnostics across rounds
- Be conservative: prefer continuing over premature stopping
- Always return valid JSON. No explanatory text outside the JSON object.
"""


def build_scheduling_message(task_type: str, current_round: int,
                             total_round: int, current_metrics: dict,
                             previous_metrics: list[dict] | None = None) -> str:
    """
    Build the user message for the scheduling advisor.

    Parameters
    ----------
    task_type : str
        The ML task type.
    current_round : int
        Current training round.
    total_round : int
        Total planned rounds.
    current_metrics : dict
        Metrics from the current aggregation round.
    previous_metrics : list[dict] | None
        Metrics from previous rounds, ordered by round number.

    Returns
    -------
    str
        Formatted message for the LLM.
    """
    import json
    data = {
        "task_type": task_type,
        "current_round": current_round,
        "total_round": total_round,
        "current_metrics": current_metrics,
        "previous_rounds": previous_metrics or [],
    }
    return json.dumps(data, indent=2)
