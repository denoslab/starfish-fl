"""
Tests for the agent loop with mocked LLM responses.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from starfish_cli.agent.agent import (
    run_agent_loop,
    extract_final_response,
    create_client,
    DEFAULT_MODEL,
    MAX_TURNS,
)


def _make_text_block(text):
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id, name, input_data):
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_response(content_blocks, stop_reason="end_turn"):
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    return response


class TestAgentLoop:
    """Test the main agent loop."""

    @patch("starfish_cli.agent.agent.execute_tool")
    @patch("starfish_cli.agent.agent.Anthropic")
    def test_simple_text_response(self, mock_anthropic_cls, mock_execute):
        """Agent responds with text only (no tool calls)."""
        client = MagicMock()
        response = _make_response([_make_text_block("Sites are registered.")])
        client.messages.create.return_value = response

        messages = run_agent_loop("What sites are registered?", client=client)

        assert len(messages) == 2  # user + assistant
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        mock_execute.assert_not_called()

    @patch("starfish_cli.agent.agent.execute_tool")
    def test_single_tool_call_then_text(self, mock_execute):
        """Agent calls one tool, then responds with text."""
        mock_execute.return_value = '{"success": true, "data": {"name": "Hospital A"}}'

        client = MagicMock()
        # First call: tool use
        tool_block = _make_tool_use_block("call_1", "starfish_site_info", {})
        response1 = _make_response([tool_block])
        # Second call: text response
        response2 = _make_response([_make_text_block("Site Hospital A is registered.")])

        client.messages.create.side_effect = [response1, response2]

        messages = run_agent_loop("Check site info", client=client)

        assert len(messages) == 4  # user, assistant(tool_use), user(tool_result), assistant(text)
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        mock_execute.assert_called_once_with("starfish_site_info", {})

    @patch("starfish_cli.agent.agent.execute_tool")
    def test_multi_step_workflow(self, mock_execute):
        """Agent performs a multi-step workflow: register site then create project."""
        mock_execute.side_effect = [
            '{"success": true, "msg": "Site registered"}',
            '{"success": true, "msg": "Project created"}',
        ]

        client = MagicMock()

        # Turn 1: register site
        tool1 = _make_tool_use_block("call_1", "starfish_site_register", {"name": "Site1"})
        resp1 = _make_response([tool1])

        # Turn 2: create project
        tool2 = _make_tool_use_block("call_2", "starfish_project_new", {
            "name": "Study", "tasks": '[{"seq":1}]'
        })
        resp2 = _make_response([tool2])

        # Turn 3: final text
        resp3 = _make_response([_make_text_block("Done! Site registered and project created.")])

        client.messages.create.side_effect = [resp1, resp2, resp3]

        messages = run_agent_loop("Register site and create project", client=client)

        assert len(messages) == 6
        assert mock_execute.call_count == 2

    @patch("starfish_cli.agent.agent.execute_tool")
    def test_parallel_tool_calls(self, mock_execute):
        """Agent calls multiple tools in a single turn."""
        mock_execute.side_effect = [
            '{"success": true, "data": {"name": "Site1"}}',
            '{"success": true, "data": {"name": "Site2"}}',
        ]

        client = MagicMock()

        # Turn 1: two tool calls in one response
        tool1 = _make_tool_use_block("call_1", "starfish_site_info", {})
        tool2 = _make_tool_use_block("call_2", "starfish_site_info", {"env_file": ".env.site2"})
        resp1 = _make_response([tool1, tool2])

        # Turn 2: text
        resp2 = _make_response([_make_text_block("Both sites checked.")])

        client.messages.create.side_effect = [resp1, resp2]

        messages = run_agent_loop("Check both sites", client=client)

        assert mock_execute.call_count == 2
        # Tool results should be in the same message
        tool_result_msg = messages[2]
        assert len(tool_result_msg["content"]) == 2

    @patch("starfish_cli.agent.agent.execute_tool")
    def test_error_response_from_tool(self, mock_execute):
        """Agent receives an error from a tool call."""
        mock_execute.return_value = '{"success": false, "msg": "Site not found"}'

        client = MagicMock()
        tool_block = _make_tool_use_block("call_1", "starfish_site_info", {})
        resp1 = _make_response([tool_block])
        resp2 = _make_response([_make_text_block("The site is not registered.")])

        client.messages.create.side_effect = [resp1, resp2]

        messages = run_agent_loop("Check site info", client=client)

        # Agent should still complete even with error tool results
        final = extract_final_response(messages)
        assert "not registered" in final

    @patch("starfish_cli.agent.agent.execute_tool")
    def test_max_turns_limit(self, mock_execute):
        """Agent stops after max_turns iterations."""
        mock_execute.return_value = '{"success": true}'

        client = MagicMock()
        # Always return a tool call (never finishes)
        tool_block = _make_tool_use_block("call_1", "starfish_run_status", {"project_id": 1})
        response = _make_response([tool_block])
        client.messages.create.return_value = response

        messages = run_agent_loop("Monitor forever", client=client, max_turns=3)

        # Should have 3 iterations * 2 messages each + 1 initial = 7
        assert client.messages.create.call_count == 3

    @patch("starfish_cli.agent.agent.execute_tool")
    def test_verbose_mode(self, mock_execute, capsys):
        """Verbose mode prints tool calls and results."""
        mock_execute.return_value = '{"success": true, "data": {"name": "Site1"}}'

        client = MagicMock()
        tool_block = _make_tool_use_block("call_1", "starfish_site_info", {})
        resp1 = _make_response([tool_block])
        resp2 = _make_response([_make_text_block("Done.")])
        client.messages.create.side_effect = [resp1, resp2]

        run_agent_loop("Check site", client=client, verbose=True)

        captured = capsys.readouterr()
        assert "[Tool Call] starfish_site_info" in captured.out
        assert "Result:" in captured.out

    def test_system_prompt_is_included(self):
        """Verify system prompt is passed to the API."""
        client = MagicMock()
        response = _make_response([_make_text_block("Hello")])
        client.messages.create.return_value = response

        run_agent_loop("Hello", client=client)

        call_kwargs = client.messages.create.call_args
        assert "system" in call_kwargs.kwargs
        assert "Starfish-FL" in call_kwargs.kwargs["system"]

    def test_tools_are_passed_to_api(self):
        """Verify tool schemas are passed to the API."""
        client = MagicMock()
        response = _make_response([_make_text_block("Hello")])
        client.messages.create.return_value = response

        run_agent_loop("Hello", client=client)

        call_kwargs = client.messages.create.call_args
        assert "tools" in call_kwargs.kwargs
        tools = call_kwargs.kwargs["tools"]
        assert len(tools) == 15

    def test_default_model(self):
        """Verify default model is used."""
        client = MagicMock()
        response = _make_response([_make_text_block("Hello")])
        client.messages.create.return_value = response

        run_agent_loop("Hello", client=client)

        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == DEFAULT_MODEL

    def test_custom_model(self):
        """Verify custom model can be specified."""
        client = MagicMock()
        response = _make_response([_make_text_block("Hello")])
        client.messages.create.return_value = response

        run_agent_loop("Hello", client=client, model="claude-haiku-4-5-20251001")

        call_kwargs = client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"


class TestExtractFinalResponse:
    """Test extracting the final text from conversation history."""

    def test_extract_text_from_last_assistant(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [_make_text_block("Response")]},
        ]
        assert extract_final_response(messages) == "Response"

    def test_extract_multiple_text_blocks(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                _make_text_block("Part 1"),
                _make_text_block("Part 2"),
            ]},
        ]
        result = extract_final_response(messages)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_skip_tool_use_blocks(self):
        tool_block = _make_tool_use_block("call_1", "starfish_site_info", {})
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [tool_block]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "{}"}]},
            {"role": "assistant", "content": [_make_text_block("Final answer")]},
        ]
        assert extract_final_response(messages) == "Final answer"

    def test_empty_messages(self):
        assert extract_final_response([]) == ""

    def test_no_assistant_messages(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert extract_final_response(messages) == ""

    def test_assistant_with_only_tool_use(self):
        """If last assistant message is only tool use, return empty."""
        tool_block = _make_tool_use_block("call_1", "starfish_site_info", {})
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [tool_block]},
        ]
        assert extract_final_response(messages) == ""


class TestCreateClient:
    """Test client creation."""

    @patch("starfish_cli.agent.agent.Anthropic")
    def test_create_with_api_key(self, mock_cls):
        create_client(api_key="test-key")
        mock_cls.assert_called_once_with(api_key="test-key")

    @patch("starfish_cli.agent.agent.Anthropic")
    def test_create_without_api_key(self, mock_cls):
        create_client()
        mock_cls.assert_called_once_with()
