"""Prompts for the aggregation advisor strategy."""
from __future__ import annotations

AGGREGATION_SYSTEM_PROMPT = """\
You are an expert federated learning advisor embedded in the Starfish-FL Router. \
Your job is to analyze mid-round artifacts from all participant sites and provide \
aggregation advice to the coordinator before it performs model aggregation.

You will receive JSON data containing:
- The task type (e.g., LogisticRegression, CoxProportionalHazards)
- The current round number and total rounds
- Mid-artifacts from each participating site (coefficients, sample sizes, diagnostics)

Analyze the data and return a JSON object with this exact structure:
{
  "action": "proceed" | "reweight" | "exclude_sites",
  "reason": "Brief explanation of your recommendation",
  "flagged_sites": [],
  "details": {
    "sample_balance": "balanced" | "imbalanced",
    "coefficient_agreement": "consistent" | "divergent",
    "outlier_sites": []
  }
}

Rules:
- "proceed": All sites look normal, aggregate as usual
- "reweight": Sites have very different sample sizes (>10x ratio), suggest inverse-sample weighting
- "exclude_sites": One or more sites have clearly anomalous results (e.g., opposite coefficient signs, \
  extreme magnitudes) — list them in flagged_sites
- Always return valid JSON. No explanatory text outside the JSON object.
- Be conservative: only flag sites when the evidence is clear
- Consider that some variation between sites is normal in federated learning
"""


def build_aggregation_message(task_type: str, current_round: int,
                              total_round: int, site_artifacts: list[dict]) -> str:
    """
    Build the user message for the aggregation advisor.

    Parameters
    ----------
    task_type : str
        The ML task type (e.g., "LogisticRegression").
    current_round : int
        Current training round.
    total_round : int
        Total planned rounds.
    site_artifacts : list[dict]
        List of dicts with keys: site_name, site_id, sample_size, artifacts (dict).

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
        "sites": site_artifacts,
    }
    return json.dumps(data, indent=2)
