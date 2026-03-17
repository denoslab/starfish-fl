"""Tests for the agent FSM transition hooks."""

from unittest.mock import patch, MagicMock, PropertyMock
from django.test import TestCase

from starfish.agent.hooks import on_aggregating, on_success, on_failed


def _make_run(
    run_id=1,
    batch=1,
    cur_seq=1,
    role="CO",
    tasks=None,
    middle_artifacts=None,
    logs=None,
    site_uid="test-uid",
    agent_config=None,
    agent_advice=None,
    agent_diagnosis=None,
):
    """Create a mock Run object."""
    run = MagicMock()
    run.id = run_id
    run.batch = batch
    run.cur_seq = cur_seq
    run.role = role
    run.tasks = tasks or [{"model": "LogisticRegression", "config": {"current_round": 2, "total_round": 5}}]
    run.middle_artifacts = middle_artifacts or []
    run.logs = logs or []
    run.site_uid = site_uid
    run.agent_advice = agent_advice or {}
    run.agent_diagnosis = agent_diagnosis or {}

    # Mock project
    project = MagicMock()
    project.agent_config = agent_config or {}
    project.agent_log = []
    run.project = project

    # Mock participant
    participant = MagicMock()
    participant.site.name = "Test Site"
    run.participant = participant

    return run


def _make_batch_runs(count=3):
    """Create mock batch runs."""
    runs = []
    for i in range(count):
        r = MagicMock()
        r.id = i + 1
        r.site_uid = f"site-uid-{i}"
        r.middle_artifacts = [{"coef": [0.5 + i * 0.1]}]
        participant = MagicMock()
        participant.site.name = f"Site {i + 1}"
        r.participant = participant
        runs.append(r)
    return runs


class TestOnAggregating(TestCase):
    """Test the aggregation hook."""

    def test_noop_when_agent_disabled(self):
        run = _make_run(agent_config={})
        batch_runs = _make_batch_runs()
        on_aggregating(run, batch_runs)
        # Should not modify the run
        run.save.assert_not_called()

    @patch("starfish.agent.hooks.get_aggregation_advice")
    def test_stores_advice_on_run(self, mock_advice):
        mock_advice.return_value = {
            "action": "proceed",
            "reason": "all good",
            "flagged_sites": [],
        }

        run = _make_run(agent_config={"enabled": True, "aggregation": True})
        batch_runs = _make_batch_runs()
        on_aggregating(run, batch_runs)

        self.assertEqual(run.agent_advice, mock_advice.return_value)
        run.save.assert_called()

    @patch("starfish.agent.hooks.get_aggregation_advice")
    def test_passes_site_artifacts(self, mock_advice):
        mock_advice.return_value = {"action": "proceed", "reason": "ok", "flagged_sites": []}

        run = _make_run(agent_config={"enabled": True, "aggregation": True})
        batch_runs = _make_batch_runs(2)
        on_aggregating(run, batch_runs)

        call_kwargs = mock_advice.call_args.kwargs
        self.assertEqual(len(call_kwargs["site_artifacts"]), 2)
        self.assertEqual(call_kwargs["task_type"], "LogisticRegression")

    @patch("starfish.agent.hooks.get_aggregation_advice")
    def test_logs_decision_to_project(self, mock_advice):
        mock_advice.return_value = {"action": "proceed", "reason": "ok", "flagged_sites": []}

        run = _make_run(agent_config={"enabled": True, "aggregation": True})
        on_aggregating(run, _make_batch_runs())

        project = run.project
        self.assertEqual(len(project.agent_log), 1)
        self.assertEqual(project.agent_log[0]["event"], "aggregation_advice")

    def test_handles_empty_tasks(self):
        run = _make_run(agent_config={"enabled": True, "aggregation": True}, tasks=[])
        on_aggregating(run, _make_batch_runs())
        run.save.assert_not_called()


class TestOnSuccess(TestCase):
    """Test the scheduling hook on success."""

    def test_noop_when_agent_disabled(self):
        run = _make_run(agent_config={})
        on_success(run)
        run.save.assert_not_called()

    @patch("starfish.agent.hooks.get_scheduling_advice")
    def test_stores_scheduling_advice(self, mock_advice):
        mock_advice.return_value = {
            "continue": True,
            "reason": "still converging",
            "suggested_rounds_remaining": 3,
            "convergence_score": 0.5,
        }

        run = _make_run(agent_config={"enabled": True, "scheduling": True})
        on_success(run)

        self.assertIn("scheduling", run.agent_advice)
        self.assertTrue(run.agent_advice["scheduling"]["continue"])
        run.save.assert_called()

    @patch("starfish.agent.hooks.get_scheduling_advice")
    def test_logs_decision_to_project(self, mock_advice):
        mock_advice.return_value = {"continue": False, "reason": "converged"}

        run = _make_run(agent_config={"enabled": True, "scheduling": True})
        on_success(run)

        project = run.project
        self.assertEqual(len(project.agent_log), 1)
        self.assertEqual(project.agent_log[0]["event"], "scheduling_advice")

    def test_handles_empty_tasks(self):
        run = _make_run(agent_config={"enabled": True, "scheduling": True}, tasks=[])
        on_success(run)
        run.save.assert_not_called()


class TestOnFailed(TestCase):
    """Test the failure triage hook."""

    def test_noop_when_agent_disabled(self):
        run = _make_run(agent_config={})
        on_failed(run)
        run.save.assert_not_called()

    @patch("starfish.agent.hooks.get_failure_diagnosis")
    def test_stores_diagnosis_on_run(self, mock_diagnosis):
        mock_diagnosis.return_value = {
            "root_cause": "Missing column",
            "category": "data_quality",
            "severity": "recoverable",
            "suggestion": "Re-upload dataset",
            "auto_action": None,
        }

        run = _make_run(
            agent_config={"enabled": True, "triage": True},
            logs=["KeyError: 'outcome'"],
        )
        on_failed(run)

        self.assertEqual(run.agent_diagnosis["category"], "data_quality")
        run.save.assert_called()

    @patch("starfish.agent.hooks.get_failure_diagnosis")
    def test_passes_correct_params(self, mock_diagnosis):
        mock_diagnosis.return_value = {
            "root_cause": "error",
            "category": "unknown",
            "severity": "fatal",
            "suggestion": "check",
            "auto_action": None,
        }

        run = _make_run(
            agent_config={"enabled": True, "triage": True},
            role="PA",
            logs=["error line 1", "error line 2"],
        )
        on_failed(run)

        call_kwargs = mock_diagnosis.call_args.kwargs
        self.assertEqual(call_kwargs["task_type"], "LogisticRegression")
        self.assertEqual(call_kwargs["role"], "participant")
        self.assertEqual(len(call_kwargs["logs"]), 2)

    @patch("starfish.agent.hooks.get_failure_diagnosis")
    def test_logs_decision_to_project(self, mock_diagnosis):
        mock_diagnosis.return_value = {
            "root_cause": "error",
            "category": "unknown",
            "severity": "fatal",
            "suggestion": "check",
            "auto_action": None,
        }

        run = _make_run(agent_config={"enabled": True, "triage": True})
        on_failed(run)

        project = run.project
        self.assertEqual(len(project.agent_log), 1)
        self.assertEqual(project.agent_log[0]["event"], "failure_triage")

    def test_handles_empty_tasks(self):
        run = _make_run(agent_config={"enabled": True, "triage": True}, tasks=[])
        on_failed(run)
        run.save.assert_not_called()
