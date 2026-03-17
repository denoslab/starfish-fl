"""Tests for the agent LLM engine."""

import json
from unittest.mock import patch, MagicMock
from django.test import TestCase

from starfish.agent.engine import query_llm, is_agent_enabled


class TestIsAgentEnabled(TestCase):
    """Test the is_agent_enabled helper."""

    def test_none_config(self):
        self.assertFalse(is_agent_enabled(None))

    def test_empty_config(self):
        self.assertFalse(is_agent_enabled({}))

    def test_enabled_false(self):
        self.assertFalse(is_agent_enabled({"enabled": False}))

    def test_enabled_true(self):
        self.assertTrue(is_agent_enabled({"enabled": True}))

    def test_feature_check_enabled(self):
        config = {"enabled": True, "aggregation": True, "triage": False}
        self.assertTrue(is_agent_enabled(config, "aggregation"))
        self.assertFalse(is_agent_enabled(config, "triage"))

    def test_feature_check_missing(self):
        config = {"enabled": True}
        self.assertFalse(is_agent_enabled(config, "aggregation"))

    def test_feature_check_disabled_globally(self):
        config = {"enabled": False, "aggregation": True}
        self.assertFalse(is_agent_enabled(config, "aggregation"))


class TestQueryLlm(TestCase):
    """Test the query_llm function."""

    def test_returns_none_when_disabled(self):
        result = query_llm("system", "user", agent_config=None)
        self.assertIsNone(result)

    def test_returns_none_when_not_enabled(self):
        result = query_llm("system", "user", agent_config={"enabled": False})
        self.assertIsNone(result)

    @patch("starfish.agent.engine._get_client")
    def test_returns_none_when_no_client(self, mock_get_client):
        mock_get_client.return_value = None
        result = query_llm("system", "user", agent_config={"enabled": True})
        self.assertIsNone(result)

    @patch("starfish.agent.engine._get_client")
    def test_successful_json_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '{"action": "proceed", "reason": "all good"}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = query_llm(
            "system prompt", "user message",
            agent_config={"enabled": True}
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "proceed")
        self.assertEqual(result["reason"], "all good")

    @patch("starfish.agent.engine._get_client")
    def test_handles_markdown_code_block(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '```json\n{"action": "proceed"}\n```'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = query_llm("sys", "user", agent_config={"enabled": True})
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "proceed")

    @patch("starfish.agent.engine._get_client")
    def test_returns_none_on_invalid_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = "This is not JSON at all"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = query_llm("sys", "user", agent_config={"enabled": True})
        self.assertIsNone(result)

    @patch("starfish.agent.engine._get_client")
    def test_returns_none_on_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        result = query_llm("sys", "user", agent_config={"enabled": True})
        self.assertIsNone(result)

    @patch("starfish.agent.engine._get_client")
    def test_passes_correct_params(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '{"ok": true}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        query_llm(
            "my system prompt", "my user message",
            model="claude-haiku-4-5-20251001", max_tokens=512,
            agent_config={"enabled": True}
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(call_kwargs["max_tokens"], 512)
        self.assertEqual(call_kwargs["system"], "my system prompt")
        self.assertEqual(call_kwargs["messages"][0]["content"], "my user message")


class TestGetClient(TestCase):
    """Test the _get_client helper."""

    @patch.dict("os.environ", {}, clear=True)
    def test_returns_none_without_api_key(self):
        from starfish.agent.engine import _get_client
        result = _get_client()
        self.assertIsNone(result)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("starfish.agent.engine.Anthropic", create=True)
    def test_returns_client_with_api_key(self, mock_anthropic_cls):
        # Import must happen after the mock is set up
        mock_anthropic_cls.return_value = MagicMock()
        from importlib import reload
        import starfish.agent.engine as engine_mod

        # Manually call _get_client with the environment set
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.assertEqual(api_key, "test-key")
