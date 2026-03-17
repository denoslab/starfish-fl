"""Tests for the controller agent LLM engine."""

from unittest.mock import patch, MagicMock
from django.test import TestCase

from starfish.controller.agent.engine import query_llm, is_agent_enabled


class TestIsAgentEnabled(TestCase):
    """Test the is_agent_enabled helper."""

    def test_none_config(self):
        self.assertFalse(is_agent_enabled(None))

    def test_empty_config(self):
        self.assertFalse(is_agent_enabled({}))

    def test_no_agent_key(self):
        self.assertFalse(is_agent_enabled({"total_round": 5}))

    def test_agent_disabled(self):
        self.assertFalse(is_agent_enabled({"agent": {"enabled": False}}))

    def test_agent_enabled(self):
        self.assertTrue(is_agent_enabled({"agent": {"enabled": True}}))

    def test_agent_empty_block(self):
        self.assertFalse(is_agent_enabled({"agent": {}}))


class TestQueryLlm(TestCase):
    """Test the query_llm function."""

    @patch("starfish.controller.agent.engine._get_client")
    def test_returns_none_when_no_client(self, mock_client):
        mock_client.return_value = None
        result = query_llm("system", "user")
        self.assertIsNone(result)

    @patch("starfish.controller.agent.engine._get_client")
    def test_parses_valid_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '{"action": "proceed"}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = query_llm("sys", "user")
        self.assertEqual(result["action"], "proceed")

    @patch("starfish.controller.agent.engine._get_client")
    def test_strips_markdown_fences(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '```json\n{"ok": true}\n```'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        result = query_llm("sys", "user")
        self.assertTrue(result["ok"])

    @patch("starfish.controller.agent.engine._get_client")
    def test_returns_none_on_bad_json(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = "not json"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        self.assertIsNone(query_llm("sys", "user"))

    @patch("starfish.controller.agent.engine._get_client")
    def test_returns_none_on_api_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("boom")

        self.assertIsNone(query_llm("sys", "user"))

    @patch("starfish.controller.agent.engine._get_client")
    def test_passes_model_param(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_block = MagicMock()
        mock_block.text = '{"ok": true}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client.messages.create.return_value = mock_response

        query_llm("sys", "user", model="claude-sonnet-4-6")
        kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")


class TestGetClient(TestCase):
    """Test _get_client graceful degradation."""

    @patch.dict("os.environ", {}, clear=True)
    def test_no_api_key_returns_none(self):
        from starfish.controller.agent.engine import _get_client
        self.assertIsNone(_get_client())
