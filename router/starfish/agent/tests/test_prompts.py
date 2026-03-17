"""Tests for prompt construction functions."""

import json
from django.test import TestCase

from starfish.agent.prompts.aggregation import (
    AGGREGATION_SYSTEM_PROMPT,
    build_aggregation_message,
)
from starfish.agent.prompts.scheduling import (
    SCHEDULING_SYSTEM_PROMPT,
    build_scheduling_message,
)
from starfish.agent.prompts.triage import (
    TRIAGE_SYSTEM_PROMPT,
    build_triage_message,
)


class TestAggregationPrompts(TestCase):
    """Test aggregation prompt construction."""

    def test_system_prompt_content(self):
        self.assertIn("federated learning", AGGREGATION_SYSTEM_PROMPT.lower())
        self.assertIn("proceed", AGGREGATION_SYSTEM_PROMPT)
        self.assertIn("reweight", AGGREGATION_SYSTEM_PROMPT)
        self.assertIn("exclude_sites", AGGREGATION_SYSTEM_PROMPT)
        self.assertIn("JSON", AGGREGATION_SYSTEM_PROMPT)

    def test_build_message_returns_valid_json(self):
        msg = build_aggregation_message(
            task_type="LogisticRegression",
            current_round=2,
            total_round=5,
            site_artifacts=[
                {"site_name": "A", "site_id": 1, "artifacts": {"coef": [0.5]}},
            ],
        )
        data = json.loads(msg)
        self.assertEqual(data["task_type"], "LogisticRegression")
        self.assertEqual(data["current_round"], 2)
        self.assertEqual(data["total_round"], 5)
        self.assertEqual(len(data["sites"]), 1)

    def test_build_message_empty_artifacts(self):
        msg = build_aggregation_message(
            task_type="CoxProportionalHazards",
            current_round=1,
            total_round=3,
            site_artifacts=[],
        )
        data = json.loads(msg)
        self.assertEqual(data["sites"], [])


class TestSchedulingPrompts(TestCase):
    """Test scheduling prompt construction."""

    def test_system_prompt_content(self):
        self.assertIn("convergence", SCHEDULING_SYSTEM_PROMPT.lower())
        self.assertIn("continue", SCHEDULING_SYSTEM_PROMPT)
        self.assertIn("early", SCHEDULING_SYSTEM_PROMPT.lower())
        self.assertIn("JSON", SCHEDULING_SYSTEM_PROMPT)

    def test_build_message_returns_valid_json(self):
        msg = build_scheduling_message(
            task_type="LogisticRegression",
            current_round=3,
            total_round=10,
            current_metrics={"coef": [0.5], "loss": 0.1},
            previous_metrics=[
                {"coef": [0.8], "loss": 0.5},
                {"coef": [0.6], "loss": 0.2},
            ],
        )
        data = json.loads(msg)
        self.assertEqual(data["task_type"], "LogisticRegression")
        self.assertEqual(data["current_round"], 3)
        self.assertEqual(len(data["previous_rounds"]), 2)

    def test_build_message_no_previous_metrics(self):
        msg = build_scheduling_message(
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            current_metrics={},
        )
        data = json.loads(msg)
        self.assertEqual(data["previous_rounds"], [])


class TestTriagePrompts(TestCase):
    """Test triage prompt construction."""

    def test_system_prompt_content(self):
        self.assertIn("failure", TRIAGE_SYSTEM_PROMPT.lower())
        self.assertIn("root_cause", TRIAGE_SYSTEM_PROMPT)
        self.assertIn("category", TRIAGE_SYSTEM_PROMPT)
        self.assertIn("severity", TRIAGE_SYSTEM_PROMPT)
        self.assertIn("data_quality", TRIAGE_SYSTEM_PROMPT)
        self.assertIn("JSON", TRIAGE_SYSTEM_PROMPT)

    def test_build_message_returns_valid_json(self):
        msg = build_triage_message(
            task_type="LogisticRegression",
            task_config={"total_round": 5, "target_column": "y"},
            run_status="Failed",
            current_round=2,
            role="participant",
            logs=["Error: KeyError 'y'", "Traceback..."],
            site_name="Hospital B",
        )
        data = json.loads(msg)
        self.assertEqual(data["task_type"], "LogisticRegression")
        self.assertEqual(data["run_status"], "Failed")
        self.assertEqual(data["role"], "participant")
        self.assertEqual(data["site_name"], "Hospital B")
        self.assertEqual(len(data["logs"]), 2)

    def test_build_message_truncates_long_logs(self):
        long_logs = [f"line {i}" for i in range(100)]
        msg = build_triage_message(
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=1,
            role="coordinator",
            logs=long_logs,
        )
        data = json.loads(msg)
        self.assertLessEqual(len(data["logs"]), 50)

    def test_build_message_none_site_name(self):
        msg = build_triage_message(
            task_type="LogisticRegression",
            task_config={},
            run_status="Failed",
            current_round=1,
            role="participant",
            logs=[],
            site_name=None,
        )
        data = json.loads(msg)
        self.assertIsNone(data["site_name"])
