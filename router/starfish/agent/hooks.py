"""
FSM transition hooks that connect the agent strategies to run state changes.

These functions are called from views.py when runs transition to specific states.
They are designed to be non-blocking and gracefully degrade if the agent is unavailable.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def on_aggregating(run, all_batch_runs):
    """
    Called when runs transition to AGGREGATING state.

    Inspects mid-artifacts from all sites and stores aggregation advice
    on the coordinator's run.

    Parameters
    ----------
    run : Run
        The coordinator's run instance.
    all_batch_runs : QuerySet
        All runs in the same batch.
    """
    from starfish.agent.strategies.aggregation import get_aggregation_advice

    project = run.project
    agent_config = getattr(project, 'agent_config', None) or {}

    if not agent_config.get("enabled"):
        return

    # Extract task info
    tasks = run.tasks
    if not tasks:
        return
    task = tasks[run.cur_seq - 1] if run.cur_seq <= len(tasks) else tasks[0]
    task_type = task.get("model", "Unknown")
    config = task.get("config", {})

    # Collect artifacts from all sites
    site_artifacts = []
    for batch_run in all_batch_runs:
        site_name = str(batch_run.site_uid)
        try:
            participant = batch_run.participant
            if participant and participant.site:
                site_name = participant.site.name
        except Exception:
            pass

        site_artifacts.append({
            "site_name": site_name,
            "site_id": batch_run.id,
            "artifacts": batch_run.middle_artifacts,
        })

    advice = get_aggregation_advice(
        agent_config=agent_config,
        task_type=task_type,
        current_round=config.get("current_round", 1),
        total_round=config.get("total_round", 1),
        site_artifacts=site_artifacts,
    )

    # Store advice on the coordinator's run
    run.agent_advice = advice
    run.save(update_fields=["agent_advice", "updated_at"])

    # Log the decision
    _append_agent_log(project, {
        "event": "aggregation_advice",
        "round": config.get("current_round", 1),
        "batch": run.batch,
        "advice": advice,
    })


def on_success(run):
    """
    Called when a coordinator's run transitions to SUCCESS after aggregation.

    Evaluates convergence and may recommend early stopping for future rounds.

    Parameters
    ----------
    run : Run
        The coordinator's run instance.
    """
    from starfish.agent.strategies.scheduling import get_scheduling_advice

    project = run.project
    agent_config = getattr(project, 'agent_config', None) or {}

    if not agent_config.get("enabled"):
        return

    tasks = run.tasks
    if not tasks:
        return
    task = tasks[run.cur_seq - 1] if run.cur_seq <= len(tasks) else tasks[0]
    task_type = task.get("model", "Unknown")
    config = task.get("config", {})

    advice = get_scheduling_advice(
        agent_config=agent_config,
        task_type=task_type,
        current_round=config.get("current_round", 1),
        total_round=config.get("total_round", 1),
        current_metrics=run.middle_artifacts[-1] if run.middle_artifacts else {},
        previous_metrics=run.middle_artifacts[:-1] if len(run.middle_artifacts) > 1 else None,
    )

    # Store scheduling advice on the run
    if hasattr(run, 'agent_advice') and isinstance(run.agent_advice, dict):
        run.agent_advice["scheduling"] = advice
    else:
        run.agent_advice = {"scheduling": advice}
    run.save(update_fields=["agent_advice", "updated_at"])

    _append_agent_log(project, {
        "event": "scheduling_advice",
        "round": config.get("current_round", 1),
        "batch": run.batch,
        "advice": advice,
    })


def on_failed(run):
    """
    Called when a run transitions to FAILED state.

    Diagnoses the failure and stores the diagnosis on the run.

    Parameters
    ----------
    run : Run
        The failed run instance.
    """
    from starfish.agent.strategies.triage import get_failure_diagnosis

    project = run.project
    agent_config = getattr(project, 'agent_config', None) or {}

    if not agent_config.get("enabled"):
        return

    tasks = run.tasks
    if not tasks:
        return
    task = tasks[run.cur_seq - 1] if run.cur_seq <= len(tasks) else tasks[0]
    task_type = task.get("model", "Unknown")
    config = task.get("config", {})

    site_name = str(run.site_uid)
    try:
        if run.participant and run.participant.site:
            site_name = run.participant.site.name
    except Exception:
        pass

    diagnosis = get_failure_diagnosis(
        agent_config=agent_config,
        task_type=task_type,
        task_config=config,
        run_status="Failed",
        current_round=config.get("current_round", 1),
        role="coordinator" if run.role == "CO" else "participant",
        logs=run.logs,
        site_name=site_name,
    )

    run.agent_diagnosis = diagnosis
    run.save(update_fields=["agent_diagnosis", "updated_at"])

    _append_agent_log(project, {
        "event": "failure_triage",
        "round": config.get("current_round", 1),
        "batch": run.batch,
        "run_id": run.id,
        "diagnosis": diagnosis,
    })


def _append_agent_log(project, entry):
    """Append an entry to the project's agent_log."""
    from django.utils import timezone

    entry["timestamp"] = timezone.now().isoformat()
    if not hasattr(project, 'agent_log') or project.agent_log is None:
        project.agent_log = []
    project.agent_log.append(entry)
    try:
        project.save(update_fields=["agent_log", "updated_at"])
    except Exception as e:
        logger.warning("Failed to save agent log: %s", str(e))
