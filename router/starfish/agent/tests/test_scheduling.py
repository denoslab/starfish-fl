"""Tests for the scheduling advisor strategy."""

from unittest.mock import patch
from django.test import TestCase

from starfish.agent.strategies.scheduling import (
    get_scheduling_advice,
    DEFAULT_SCHEDULING,
)


class TestGetSchedulingAdvice(TestCase):
    """Test scheduling advisor responses."""

    def test_returns_default_when_disabled(self):
        result = get_scheduling_advice(
            agent_config={},
            task_type="LogisticRegression",
            current_round=2,
            total_round=5,
            current_metrics={},
        )
        self.assertEqual(result, DEFAULT_SCHEDULING)

    def test_returns_default_when_feature_disabled(self):
        result = get_scheduling_advice(
            agent_config={"enabled": True, "scheduling": False},
            task_type="LogisticRegression",
            current_round=2,
            total_round=5,
            current_metrics={},
        )
        self.assertEqual(result, DEFAULT_SCHEDULING)

    def test_first_round_always_continues(self):
        result = get_scheduling_advice(
            agent_config={"enabled": True, "scheduling": True},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            current_metrics={"coef": [0.5]},
        )

        self.assertTrue(result["continue"])
        self.assertIn("First round", result["reason"])
        self.assertEqual(result["suggested_rounds_remaining"], 4)
        self.assertEqual(result["convergence_score"], 0.0)

    @patch("starfish.agent.strategies.scheduling.query_llm")
    def test_converging_model_stops_early(self, mock_query):
        mock_query.return_value = {
            "continue": False,
            "reason": "Coefficients stable for 2 rounds",
            "suggested_rounds_remaining": 0,
            "convergence_score": 0.95,
        }

        result = get_scheduling_advice(
            agent_config={"enabled": True, "scheduling": True},
            task_type="LogisticRegression",
            current_round=3,
            total_round=10,
            current_metrics={"coef": [0.501]},
            previous_metrics=[
                {"coef": [0.5]},
                {"coef": [0.500]},
            ],
        )

        self.assertFalse(result["continue"])
        self.assertGreater(result["convergence_score"], 0.9)

    @patch("starfish.agent.strategies.scheduling.query_llm")
    def test_diverging_model_continues(self, mock_query):
        mock_query.return_value = {
            "continue": True,
            "reason": "Coefficients still oscillating",
            "suggested_rounds_remaining": 5,
            "convergence_score": 0.2,
        }

        result = get_scheduling_advice(
            agent_config={"enabled": True, "scheduling": True},
            task_type="LogisticRegression",
            current_round=3,
            total_round=10,
            current_metrics={"coef": [1.5]},
            previous_metrics=[{"coef": [0.5]}, {"coef": [-0.3]}],
        )

        self.assertTrue(result["continue"])

    @patch("starfish.agent.strategies.scheduling.query_llm")
    def test_returns_default_on_llm_failure(self, mock_query):
        mock_query.return_value = None

        result = get_scheduling_advice(
            agent_config={"enabled": True, "scheduling": True},
            task_type="LogisticRegression",
            current_round=3,
            total_round=5,
            current_metrics={},
        )

        self.assertEqual(result, DEFAULT_SCHEDULING)

    @patch("starfish.agent.strategies.scheduling.query_llm")
    def test_returns_default_on_missing_continue_field(self, mock_query):
        mock_query.return_value = {"reason": "no continue field"}

        result = get_scheduling_advice(
            agent_config={"enabled": True, "scheduling": True},
            task_type="LogisticRegression",
            current_round=3,
            total_round=5,
            current_metrics={},
        )

        self.assertEqual(result, DEFAULT_SCHEDULING)
