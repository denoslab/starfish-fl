"""Tests for TaskAgentHooks class."""

import logging
from unittest.mock import patch, MagicMock
from django.test import TestCase

from starfish.controller.agent.hooks import TaskAgentHooks

# Patch targets — the strategy functions called inside hooks
_ENGINE = "starfish.controller.agent.engine.query_llm"


class TestTaskAgentHooksInit(TestCase):
    """Test hook initialisation from task config."""

    def test_defaults_all_disabled(self):
        hooks = TaskAgentHooks()
        self.assertFalse(hooks.enabled)
        self.assertFalse(hooks.early_stopping)
        self.assertFalse(hooks.summaries)
        self.assertFalse(hooks.outlier_detection)

    def test_empty_config(self):
        hooks = TaskAgentHooks({})
        self.assertFalse(hooks.enabled)

    def test_enabled_with_features(self):
        cfg = {"agent": {
            "enabled": True,
            "early_stopping": True,
            "summaries": True,
            "outlier_detection": True,
        }}
        hooks = TaskAgentHooks(cfg)
        self.assertTrue(hooks.enabled)
        self.assertTrue(hooks.early_stopping)
        self.assertTrue(hooks.summaries)
        self.assertTrue(hooks.outlier_detection)

    def test_partial_features(self):
        cfg = {"agent": {"enabled": True, "summaries": True}}
        hooks = TaskAgentHooks(cfg)
        self.assertTrue(hooks.enabled)
        self.assertTrue(hooks.summaries)
        self.assertFalse(hooks.early_stopping)
        self.assertFalse(hooks.outlier_detection)


class TestPostTraining(TestCase):
    """Test post_training hook."""

    def test_returns_none_when_disabled(self):
        hooks = TaskAgentHooks()
        result = hooks.post_training("LR", 1, 5, {})
        self.assertIsNone(result)

    def test_returns_none_when_summaries_off(self):
        hooks = TaskAgentHooks({"agent": {"enabled": True, "summaries": False}})
        result = hooks.post_training("LR", 1, 5, {})
        self.assertIsNone(result)

    @patch(_ENGINE)
    def test_returns_llm_result(self, mock_query):
        mock_query.return_value = {"summary": "Good results", "flag": None}
        hooks = TaskAgentHooks({"agent": {"enabled": True, "summaries": True}})
        result = hooks.post_training("LR", 2, 5, {"coef": [0.5]})
        self.assertEqual(result["summary"], "Good results")
        self.assertIsNone(result["flag"])

    @patch(_ENGINE)
    def test_logs_summary(self, mock_query):
        mock_query.return_value = {"summary": "Round went well", "flag": None}
        hooks = TaskAgentHooks({"agent": {"enabled": True, "summaries": True}})
        mock_logger = MagicMock()
        hooks.post_training("LR", 2, 5, {}, task_logger=mock_logger)
        mock_logger.info.assert_called()
        log_msg = mock_logger.info.call_args[0][0]
        self.assertIn("[Agent]", log_msg)

    @patch(_ENGINE)
    def test_logs_flag_as_warning(self, mock_query):
        mock_query.return_value = {"summary": "ok", "flag": "High VIF detected"}
        hooks = TaskAgentHooks({"agent": {"enabled": True, "summaries": True}})
        mock_logger = MagicMock()
        hooks.post_training("LR", 2, 5, {}, task_logger=mock_logger)
        mock_logger.warning.assert_called()

    @patch(_ENGINE)
    def test_returns_none_on_llm_failure(self, mock_query):
        mock_query.return_value = None
        hooks = TaskAgentHooks({"agent": {"enabled": True, "summaries": True}})
        result = hooks.post_training("LR", 1, 5, {})
        self.assertIsNone(result)

    @patch(_ENGINE, side_effect=RuntimeError("boom"))
    def test_catches_exceptions(self, mock_query):
        hooks = TaskAgentHooks({"agent": {"enabled": True, "summaries": True}})
        result = hooks.post_training("LR", 1, 5, {})
        self.assertIsNone(result)


class TestPreAggregation(TestCase):
    """Test pre_aggregation hook."""

    def test_returns_none_when_disabled(self):
        hooks = TaskAgentHooks()
        self.assertIsNone(hooks.pre_aggregation("LR", 1, 5, []))

    def test_returns_none_when_outlier_off(self):
        hooks = TaskAgentHooks({"agent": {"enabled": True, "outlier_detection": False}})
        self.assertIsNone(hooks.pre_aggregation("LR", 1, 5, []))

    @patch(_ENGINE)
    def test_returns_advice(self, mock_query):
        mock_query.return_value = {
            "action": "proceed", "reason": "ok",
            "flagged_sites": [], "summary": "All consistent",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True, "outlier_detection": True}})
        result = hooks.pre_aggregation("LR", 2, 5, [{"coef": [0.5]}])
        self.assertEqual(result["action"], "proceed")

    @patch(_ENGINE)
    def test_logs_flagged_sites(self, mock_query):
        mock_query.return_value = {
            "action": "exclude_sites", "reason": "outlier",
            "flagged_sites": [2], "summary": "Site 2 divergent",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True, "outlier_detection": True}})
        mock_logger = MagicMock()
        hooks.pre_aggregation("LR", 2, 5, [], task_logger=mock_logger)
        mock_logger.warning.assert_called()


class TestPostAggregation(TestCase):
    """Test post_aggregation hook."""

    def test_returns_none_when_disabled(self):
        hooks = TaskAgentHooks()
        self.assertIsNone(hooks.post_aggregation("LR", 1, 5, {}))

    def test_returns_none_when_early_stopping_off(self):
        hooks = TaskAgentHooks({"agent": {"enabled": True, "early_stopping": False}})
        self.assertIsNone(hooks.post_aggregation("LR", 1, 5, {}))

    @patch(_ENGINE)
    def test_converged(self, mock_query):
        mock_query.return_value = {
            "converged": True, "reason": "stable",
            "convergence_score": 0.95, "summary": "Model converged",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True, "early_stopping": True}})
        result = hooks.post_aggregation("LR", 5, 10, {})
        self.assertTrue(result["converged"])

    @patch(_ENGINE)
    def test_not_converged(self, mock_query):
        mock_query.return_value = {
            "converged": False, "reason": "still changing",
            "convergence_score": 0.3, "summary": "Continue",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True, "early_stopping": True}})
        result = hooks.post_aggregation("LR", 2, 10, {})
        self.assertFalse(result["converged"])

    @patch(_ENGINE)
    def test_logs_early_stop(self, mock_query):
        mock_query.return_value = {
            "converged": True, "reason": "stable",
            "convergence_score": 0.99, "summary": "Done",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True, "early_stopping": True}})
        mock_logger = MagicMock()
        hooks.post_aggregation("LR", 5, 10, {}, task_logger=mock_logger)
        # Should log both summary and early stopping
        self.assertGreaterEqual(mock_logger.info.call_count, 2)


class TestOnFailure(TestCase):
    """Test on_failure hook."""

    def test_returns_none_when_disabled(self):
        hooks = TaskAgentHooks()
        self.assertIsNone(hooks.on_failure("LR", {}, 1, "participant", "err", []))

    @patch(_ENGINE)
    def test_returns_diagnosis(self, mock_query):
        mock_query.return_value = {
            "root_cause": "Missing column",
            "category": "data_quality",
            "severity": "recoverable",
            "suggestion": "Re-upload",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True}})
        result = hooks.on_failure("LR", {}, 1, "participant", "KeyError", ["traceback"])
        self.assertEqual(result["category"], "data_quality")

    @patch(_ENGINE)
    def test_logs_diagnosis(self, mock_query):
        mock_query.return_value = {
            "root_cause": "OOM", "category": "resource",
            "severity": "recoverable", "suggestion": "reduce batch",
        }
        hooks = TaskAgentHooks({"agent": {"enabled": True}})
        mock_logger = MagicMock()
        hooks.on_failure("LR", {}, 1, "co", "MemoryError", [], task_logger=mock_logger)
        mock_logger.info.assert_called()
        log_msg = mock_logger.info.call_args[0][0]
        self.assertIn("[Agent]", log_msg)

    @patch(_ENGINE, side_effect=RuntimeError("boom"))
    def test_catches_exceptions(self, mock_query):
        hooks = TaskAgentHooks({"agent": {"enabled": True}})
        result = hooks.on_failure("LR", {}, 1, "co", "err", [])
        self.assertIsNone(result)
