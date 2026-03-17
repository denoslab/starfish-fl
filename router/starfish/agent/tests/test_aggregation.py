"""Tests for the aggregation advisor strategy."""

from unittest.mock import patch, MagicMock
from django.test import TestCase

from starfish.agent.strategies.aggregation import (
    get_aggregation_advice,
    DEFAULT_ADVICE,
)


class TestGetAggregationAdvice(TestCase):
    """Test aggregation advisor responses."""

    def test_returns_default_when_disabled(self):
        result = get_aggregation_advice(
            agent_config={},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            site_artifacts=[],
        )
        self.assertEqual(result, DEFAULT_ADVICE)

    def test_returns_default_when_feature_disabled(self):
        result = get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": False},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            site_artifacts=[],
        )
        self.assertEqual(result, DEFAULT_ADVICE)

    @patch("starfish.agent.strategies.aggregation.query_llm")
    def test_returns_llm_advice_on_proceed(self, mock_query):
        mock_query.return_value = {
            "action": "proceed",
            "reason": "All sites have consistent coefficients",
            "flagged_sites": [],
            "details": {
                "sample_balance": "balanced",
                "coefficient_agreement": "consistent",
                "outlier_sites": [],
            },
        }

        result = get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": True},
            task_type="LogisticRegression",
            current_round=2,
            total_round=5,
            site_artifacts=[
                {"site_name": "Site A", "site_id": 1, "artifacts": {"coef": [0.5, -0.3]}},
                {"site_name": "Site B", "site_id": 2, "artifacts": {"coef": [0.4, -0.2]}},
            ],
        )

        self.assertEqual(result["action"], "proceed")
        mock_query.assert_called_once()

    @patch("starfish.agent.strategies.aggregation.query_llm")
    def test_returns_advice_with_outlier(self, mock_query):
        mock_query.return_value = {
            "action": "exclude_sites",
            "reason": "Site C has coefficient 10x larger than others",
            "flagged_sites": ["Site C"],
            "details": {
                "sample_balance": "balanced",
                "coefficient_agreement": "divergent",
                "outlier_sites": ["Site C"],
            },
        }

        result = get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": True},
            task_type="LogisticRegression",
            current_round=2,
            total_round=5,
            site_artifacts=[
                {"site_name": "Site A", "site_id": 1, "artifacts": {"coef": [0.5]}},
                {"site_name": "Site B", "site_id": 2, "artifacts": {"coef": [0.4]}},
                {"site_name": "Site C", "site_id": 3, "artifacts": {"coef": [5.0]}},
            ],
        )

        self.assertEqual(result["action"], "exclude_sites")
        self.assertIn("Site C", result["flagged_sites"])

    @patch("starfish.agent.strategies.aggregation.query_llm")
    def test_returns_reweight_advice(self, mock_query):
        mock_query.return_value = {
            "action": "reweight",
            "reason": "Site A has 10x more data than Site B",
            "flagged_sites": [],
            "details": {
                "sample_balance": "imbalanced",
                "coefficient_agreement": "consistent",
                "outlier_sites": [],
            },
        }

        result = get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": True},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            site_artifacts=[
                {"site_name": "Site A", "site_id": 1, "artifacts": {"n": 10000}},
                {"site_name": "Site B", "site_id": 2, "artifacts": {"n": 100}},
            ],
        )

        self.assertEqual(result["action"], "reweight")

    @patch("starfish.agent.strategies.aggregation.query_llm")
    def test_returns_default_on_llm_failure(self, mock_query):
        mock_query.return_value = None

        result = get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": True},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            site_artifacts=[],
        )

        self.assertEqual(result, DEFAULT_ADVICE)

    @patch("starfish.agent.strategies.aggregation.query_llm")
    def test_returns_default_on_missing_fields(self, mock_query):
        mock_query.return_value = {"incomplete": True}

        result = get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": True},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            site_artifacts=[],
        )

        self.assertEqual(result, DEFAULT_ADVICE)

    @patch("starfish.agent.strategies.aggregation.query_llm")
    def test_uses_aggregation_model(self, mock_query):
        mock_query.return_value = {"action": "proceed", "reason": "ok"}

        get_aggregation_advice(
            agent_config={"enabled": True, "aggregation": True},
            task_type="LogisticRegression",
            current_round=1,
            total_round=5,
            site_artifacts=[],
        )

        call_kwargs = mock_query.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "claude-sonnet-4-6")
