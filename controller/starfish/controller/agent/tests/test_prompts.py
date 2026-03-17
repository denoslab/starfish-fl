"""Tests for prompt templates and message builders."""

import json
from django.test import TestCase

from starfish.controller.agent.prompts import (
    POST_TRAINING_SYSTEM,
    PRE_AGGREGATION_SYSTEM,
    POST_AGGREGATION_SYSTEM,
    ON_FAILURE_SYSTEM,
    build_post_training_message,
    build_pre_aggregation_message,
    build_post_aggregation_message,
    build_on_failure_message,
)


class TestSystemPrompts(TestCase):
    """Verify system prompts contain essential instructions."""

    def test_post_training_prompt(self):
        self.assertIn("summary", POST_TRAINING_SYSTEM)
        self.assertIn("flag", POST_TRAINING_SYSTEM)
        self.assertIn("JSON", POST_TRAINING_SYSTEM)

    def test_pre_aggregation_prompt(self):
        self.assertIn("proceed", PRE_AGGREGATION_SYSTEM)
        self.assertIn("reweight", PRE_AGGREGATION_SYSTEM)
        self.assertIn("exclude_sites", PRE_AGGREGATION_SYSTEM)
        self.assertIn("JSON", PRE_AGGREGATION_SYSTEM)

    def test_post_aggregation_prompt(self):
        self.assertIn("converged", POST_AGGREGATION_SYSTEM)
        self.assertIn("convergence_score", POST_AGGREGATION_SYSTEM)
        self.assertIn("JSON", POST_AGGREGATION_SYSTEM)

    def test_on_failure_prompt(self):
        self.assertIn("root_cause", ON_FAILURE_SYSTEM)
        self.assertIn("category", ON_FAILURE_SYSTEM)
        self.assertIn("severity", ON_FAILURE_SYSTEM)
        self.assertIn("JSON", ON_FAILURE_SYSTEM)


class TestMessageBuilders(TestCase):
    """Verify message builders produce valid JSON with expected fields."""

    def test_post_training_message(self):
        msg = build_post_training_message(
            "LogisticRegression", 2, 5, {"coef": [0.5], "sample_size": 100})
        data = json.loads(msg)
        self.assertEqual(data["task_type"], "LogisticRegression")
        self.assertEqual(data["round"], 2)
        self.assertEqual(data["total_round"], 5)
        self.assertIn("mid_artifacts", data)

    def test_pre_aggregation_message(self):
        artifacts = [{"site": "A", "coef": [0.5]}, {"site": "B", "coef": [0.4]}]
        msg = build_pre_aggregation_message("LogisticRegression", 1, 5, artifacts)
        data = json.loads(msg)
        self.assertEqual(len(data["site_artifacts"]), 2)

    def test_post_aggregation_message(self):
        msg = build_post_aggregation_message(
            "CoxPH", 3, 10, {"coef": [0.5]},
            [{"coef": [0.8]}, {"coef": [0.6]}])
        data = json.loads(msg)
        self.assertEqual(data["round"], 3)
        self.assertEqual(len(data["previous_rounds"]), 2)

    def test_post_aggregation_no_history(self):
        msg = build_post_aggregation_message("LR", 1, 5, {})
        data = json.loads(msg)
        self.assertEqual(data["previous_rounds"], [])

    def test_on_failure_message(self):
        msg = build_on_failure_message(
            "LogisticRegression", {"total_round": 5}, 2,
            "participant", "KeyError: 'y'", ["line1", "line2"])
        data = json.loads(msg)
        self.assertEqual(data["error"], "KeyError: 'y'")
        self.assertEqual(data["role"], "participant")

    def test_on_failure_truncates_logs(self):
        long_logs = [f"line {i}" for i in range(100)]
        msg = build_on_failure_message("LR", {}, 1, "co", "err", long_logs)
        data = json.loads(msg)
        self.assertLessEqual(len(data["logs"]), 50)
