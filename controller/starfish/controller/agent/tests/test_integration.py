"""
Integration tests: verify hooks are wired into AbstractTask lifecycle.

Uses a minimal concrete task subclass with mocked I/O and LLM calls.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch, MagicMock, PropertyMock
from django.test import TestCase

from starfish.controller.agent.hooks import TaskAgentHooks


def _make_run(agent_config=None):
    """Build a minimal run dict for AbstractTask.__init__."""
    config = {"total_round": 5, "current_round": 2}
    if agent_config:
        config["agent"] = agent_config
    return {
        "id": 1,
        "project": 10,
        "batch": 1,
        "role": "coordinator",
        "status": "standby",
        "cur_seq": 1,
        "tasks": [{"seq": 1, "model": "LogisticRegression", "config": config}],
    }


class TestAbstractTaskAgentInit(TestCase):
    """Verify _agent_hooks is initialised during post_init."""

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_hooks_disabled_by_default(self, mock_create, mock_logs):
        """Without agent config, hooks should be initialised but disabled."""
        from starfish.controller.tasks.abstract_task import AbstractTask

        # Create a minimal concrete subclass
        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        task = DummyTask(_make_run())
        self.assertIsNotNone(task._agent_hooks)
        self.assertFalse(task._agent_hooks.enabled)

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_hooks_enabled_with_config(self, mock_create, mock_logs):
        """With agent config, hooks should be enabled."""
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        run = _make_run(agent_config={
            "enabled": True, "summaries": True, "early_stopping": True,
        })
        task = DummyTask(run)
        self.assertTrue(task._agent_hooks.enabled)
        self.assertTrue(task._agent_hooks.summaries)
        self.assertTrue(task._agent_hooks.early_stopping)


class TestRunningHookIntegration(TestCase):
    """Verify the post_training hook fires during running()."""

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    @patch("starfish.controller.tasks.abstract_task.gen_mid_artifacts_url", return_value="/tmp/fake_mid")
    def test_post_training_hook_called_on_success(self, mock_mid_url, mock_create, mock_logs):
        """When training succeeds, post_training hook should fire."""
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        run = _make_run(agent_config={"enabled": True, "summaries": True})
        task = DummyTask(run)
        task.notify = MagicMock()
        task._agent_hooks.post_training = MagicMock(return_value={"summary": "ok"})

        task.running(run)

        task._agent_hooks.post_training.assert_called_once()
        # Should still notify 5 (pending_success)
        task.notify.assert_called_with(5)


class TestAggregatingHookIntegration(TestCase):
    """Verify pre/post aggregation hooks fire during aggregating()."""

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_early_stopping_triggers(self, mock_create, mock_logs):
        """When post_aggregation says converged, should notify SUCCESS (8)."""
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        run = _make_run(agent_config={"enabled": True, "early_stopping": True})
        task = DummyTask(run)
        task.notify = MagicMock()
        task.runs_in_fails = MagicMock(return_value=False)
        task._agent_hooks.post_aggregation = MagicMock(return_value={
            "converged": True, "reason": "stable", "summary": "done",
        })
        # is_last_round returns False (round 2 of 5), but agent says converged
        task.aggregating(run)

        # Should call notify(8) for SUCCESS (early stop)
        task.notify.assert_called_with(8, param={'update_all': True})

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_no_early_stop_when_not_converged(self, mock_create, mock_logs):
        """When post_aggregation says not converged, continue normally."""
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        run = _make_run(agent_config={"enabled": True, "early_stopping": True})
        task = DummyTask(run)
        task.notify = MagicMock()
        task.runs_in_fails = MagicMock(return_value=False)
        task._agent_hooks.post_aggregation = MagicMock(return_value={
            "converged": False, "reason": "still changing",
        })

        task.aggregating(run)

        # Should call notify(2) with increase_round (not 8)
        task.notify.assert_called_with(
            2, param={'increase_round': True, 'update_all': True})

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_pre_aggregation_hook_called(self, mock_create, mock_logs):
        """Pre-aggregation hook should fire before do_aggregate."""
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        run = _make_run(agent_config={"enabled": True, "outlier_detection": True})
        task = DummyTask(run)
        task.notify = MagicMock()
        task.runs_in_fails = MagicMock(return_value=False)
        task._agent_hooks.pre_aggregation = MagicMock(return_value=None)

        task.aggregating(run)

        task._agent_hooks.pre_aggregation.assert_called_once()


class TestPendingFailedHookIntegration(TestCase):
    """Verify on_failure hook fires during pending_failed()."""

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_on_failure_hook_called(self, mock_create, mock_logs):
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        run = _make_run(agent_config={"enabled": True})
        task = DummyTask(run)
        task.notify = MagicMock()
        task.upload = MagicMock(return_value=True)
        task._agent_hooks.on_failure = MagicMock(return_value=None)

        task.pending_failed(run)

        task._agent_hooks.on_failure.assert_called_once()
        task.notify.assert_called_with(0)


class TestBackwardCompatibility(TestCase):
    """Ensure tasks without agent config work exactly as before."""

    @patch("starfish.controller.tasks.abstract_task.gen_logs_url", return_value="/tmp/test_log")
    @patch("starfish.controller.tasks.abstract_task.create_if_not_exist")
    def test_no_agent_config_runs_normally(self, mock_create, mock_logs):
        from starfish.controller.tasks.abstract_task import AbstractTask

        class DummyTask(AbstractTask):
            def validate(self): return True
            def prepare_data(self): return True
            def training(self): return True
            def do_aggregate(self): return True

        task = DummyTask(_make_run())
        task.notify = MagicMock()
        task.runs_in_fails = MagicMock(return_value=False)

        # aggregating should work without agent
        task.aggregating(_make_run())
        # Should still reach notify (either 8 or 2)
        task.notify.assert_called()
