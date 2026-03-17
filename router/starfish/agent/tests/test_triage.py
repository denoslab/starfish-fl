"""Tests for the failure triage strategy."""

from unittest.mock import patch
from django.test import TestCase

from starfish.agent.strategies.triage import (
    get_failure_diagnosis,
    DEFAULT_DIAGNOSIS,
)


class TestGetFailureDiagnosis(TestCase):
    """Test failure triage advisor responses."""

    def test_returns_default_when_disabled(self):
        result = get_failure_diagnosis(
            agent_config={},
            task_type="LogisticRegression",
            task_config={"total_round": 5},
            run_status="Failed",
            current_round=2,
            role="participant",
            logs=["Error occurred"],
        )
        self.assertEqual(result, DEFAULT_DIAGNOSIS)

    def test_returns_default_when_feature_disabled(self):
        result = get_failure_diagnosis(
            agent_config={"enabled": True, "triage": False},
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=[],
        )
        self.assertEqual(result, DEFAULT_DIAGNOSIS)

    @patch("starfish.agent.strategies.triage.query_llm")
    def test_data_quality_diagnosis(self, mock_query):
        mock_query.return_value = {
            "root_cause": "Site B has mismatched column names",
            "category": "data_quality",
            "severity": "recoverable",
            "suggestion": "Re-upload dataset with columns: [age, treatment, outcome]",
            "auto_action": None,
        }

        result = get_failure_diagnosis(
            agent_config={"enabled": True, "triage": True},
            task_type="LogisticRegression",
            task_config={"total_round": 5, "target_column": "outcome"},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=["KeyError: 'outcome'", "Traceback..."],
            site_name="Hospital B",
        )

        self.assertEqual(result["category"], "data_quality")
        self.assertEqual(result["severity"], "recoverable")
        self.assertIn("column", result["suggestion"].lower())

    @patch("starfish.agent.strategies.triage.query_llm")
    def test_network_error_diagnosis(self, mock_query):
        mock_query.return_value = {
            "root_cause": "Connection timeout to router",
            "category": "network",
            "severity": "recoverable",
            "suggestion": "Check network connectivity and retry",
            "auto_action": "restart",
        }

        result = get_failure_diagnosis(
            agent_config={"enabled": True, "triage": True},
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=2,
            role="participant",
            logs=["ConnectionError: timed out"],
        )

        self.assertEqual(result["category"], "network")
        self.assertEqual(result["auto_action"], "restart")

    @patch("starfish.agent.strategies.triage.query_llm")
    def test_resource_error_diagnosis(self, mock_query):
        mock_query.return_value = {
            "root_cause": "Out of memory during training",
            "category": "resource",
            "severity": "recoverable",
            "suggestion": "Reduce batch size or increase memory allocation",
            "auto_action": None,
        }

        result = get_failure_diagnosis(
            agent_config={"enabled": True, "triage": True},
            task_type="FederatedUNet",
            task_config={"batch_size": 64},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=["MemoryError: unable to allocate array"],
        )

        self.assertEqual(result["category"], "resource")

    @patch("starfish.agent.strategies.triage.query_llm")
    def test_returns_default_on_llm_failure(self, mock_query):
        mock_query.return_value = None

        result = get_failure_diagnosis(
            agent_config={"enabled": True, "triage": True},
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=["some error"],
        )

        self.assertEqual(result, DEFAULT_DIAGNOSIS)

    @patch("starfish.agent.strategies.triage.query_llm")
    def test_returns_default_on_missing_fields(self, mock_query):
        mock_query.return_value = {"some_field": "but no root_cause or category"}

        result = get_failure_diagnosis(
            agent_config={"enabled": True, "triage": True},
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=[],
        )

        self.assertEqual(result, DEFAULT_DIAGNOSIS)

    @patch("starfish.agent.strategies.triage.query_llm")
    def test_log_truncation(self, mock_query):
        """Verify very long logs are truncated before sending to LLM."""
        mock_query.return_value = {
            "root_cause": "error",
            "category": "unknown",
            "severity": "fatal",
            "suggestion": "check logs",
            "auto_action": None,
        }

        long_logs = [f"log line {i}" for i in range(100)]
        get_failure_diagnosis(
            agent_config={"enabled": True, "triage": True},
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=long_logs,
        )

        # Verify the message passed to query_llm contains truncated logs
        call_args = mock_query.call_args
        import json
        message_data = json.loads(call_args.kwargs["user_message"])
        self.assertLessEqual(len(message_data["logs"]), 50)
