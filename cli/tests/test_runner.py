"""
Tests for the agent CLI runner (starfish agent run/tools commands).
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from starfish_cli.agent.runner import app

runner = CliRunner()

# Mock paths must target the source module since runner.py imports locally
_AGENT_MOD = "starfish_cli.agent.agent"


class TestToolsCommand:
    """Test `starfish agent tools` command."""

    def test_tools_json_output(self):
        result = runner.invoke(app, ["tools", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 15
        names = {t["name"] for t in data}
        assert "starfish_site_info" in names
        assert "starfish_run_start" in names

    def test_tools_table_output(self):
        result = runner.invoke(app, ["tools"])
        assert result.exit_code == 0
        assert "Starfish Agent Tools" in result.output
        assert "starfish_site_info" in result.output


class TestRunCommand:
    """Test `starfish agent run` command."""

    def test_run_without_api_key_fails(self):
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["run", "test goal"])
            assert result.exit_code == 1

    @patch(f"{_AGENT_MOD}.run_agent_loop")
    @patch(f"{_AGENT_MOD}.extract_final_response")
    def test_run_with_api_key(self, mock_extract, mock_loop):
        mock_loop.return_value = []
        mock_extract.return_value = "Agent completed the task."

        result = runner.invoke(app, ["run", "test goal", "--api-key", "test-key-123"])
        assert result.exit_code == 0
        assert "Agent completed the task." in result.output
        mock_loop.assert_called_once()

    @patch(f"{_AGENT_MOD}.run_agent_loop")
    @patch(f"{_AGENT_MOD}.extract_final_response")
    def test_run_verbose(self, mock_extract, mock_loop):
        mock_loop.return_value = []
        mock_extract.return_value = "Done."

        result = runner.invoke(app, [
            "run", "test goal", "--api-key", "test-key", "--verbose"
        ])
        assert result.exit_code == 0
        assert "Agent goal: test goal" in result.output

    @patch(f"{_AGENT_MOD}.run_agent_loop")
    def test_run_json_output(self, mock_loop):
        mock_loop.return_value = [
            {"role": "user", "content": "test"},
        ]

        result = runner.invoke(app, [
            "run", "test goal", "--api-key", "test-key", "--json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)

    @patch(f"{_AGENT_MOD}.run_agent_loop")
    @patch(f"{_AGENT_MOD}.extract_final_response")
    def test_run_custom_model(self, mock_extract, mock_loop):
        mock_loop.return_value = []
        mock_extract.return_value = "Done."

        runner.invoke(app, [
            "run", "test", "--api-key", "key", "--model", "claude-haiku-4-5-20251001"
        ])

        call_kwargs = mock_loop.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"

    @patch(f"{_AGENT_MOD}.run_agent_loop")
    @patch(f"{_AGENT_MOD}.extract_final_response")
    def test_run_custom_max_turns(self, mock_extract, mock_loop):
        mock_loop.return_value = []
        mock_extract.return_value = "Done."

        runner.invoke(app, [
            "run", "test", "--api-key", "key", "--max-turns", "10"
        ])

        call_kwargs = mock_loop.call_args
        assert call_kwargs.kwargs["max_turns"] == 10

    @patch(f"{_AGENT_MOD}.run_agent_loop")
    @patch(f"{_AGENT_MOD}.extract_final_response")
    def test_run_empty_response(self, mock_extract, mock_loop):
        mock_loop.return_value = []
        mock_extract.return_value = ""

        result = runner.invoke(app, ["run", "test", "--api-key", "key"])
        assert result.exit_code == 0
        assert "completed without a final text response" in result.output
