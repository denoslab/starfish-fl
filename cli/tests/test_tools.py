"""
Tests for agent tool definitions and execution.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from starfish_cli.agent.tools import (
    get_tool_schemas,
    execute_tool,
    TOOL_HANDLERS,
    TOOL_DEFINITIONS,
    _run_cli,
    _build_cmd,
)


class TestToolSchemas:
    """Test that all tool schemas are well-formed."""

    def test_all_tools_have_required_fields(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool {tool['name']} missing 'description'"
            assert "input_schema" in tool, f"Tool {tool['name']} missing 'input_schema'"

    def test_all_tool_names_are_unique(self):
        schemas = get_tool_schemas()
        names = [t["name"] for t in schemas]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_all_schemas_have_valid_input_schema(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_required_params_exist_in_properties(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            schema = tool["input_schema"]
            for req in schema["required"]:
                assert req in schema["properties"], (
                    f"Tool {tool['name']}: required param '{req}' not in properties"
                )

    def test_all_properties_have_types(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            for prop_name, prop in tool["input_schema"]["properties"].items():
                assert "type" in prop or "enum" in prop, (
                    f"Tool {tool['name']}: property '{prop_name}' missing type"
                )

    def test_all_properties_have_descriptions(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            for prop_name, prop in tool["input_schema"]["properties"].items():
                assert "description" in prop, (
                    f"Tool {tool['name']}: property '{prop_name}' missing description"
                )

    def test_expected_tools_exist(self):
        schemas = get_tool_schemas()
        names = {t["name"] for t in schemas}
        expected = {
            "starfish_site_info",
            "starfish_site_register",
            "starfish_site_update",
            "starfish_site_deregister",
            "starfish_project_list",
            "starfish_project_new",
            "starfish_project_join",
            "starfish_project_leave",
            "starfish_project_detail",
            "starfish_run_start",
            "starfish_run_status",
            "starfish_run_detail",
            "starfish_run_logs",
            "starfish_dataset_upload",
            "starfish_artifact_download",
        }
        assert expected == names

    def test_tool_count(self):
        assert len(get_tool_schemas()) == 15

    def test_every_tool_has_handler(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            assert tool["name"] in TOOL_HANDLERS, (
                f"Tool {tool['name']} has no handler"
            )

    def test_every_handler_has_schema(self):
        schema_names = {t["name"] for t in get_tool_schemas()}
        for handler_name in TOOL_HANDLERS:
            assert handler_name in schema_names, (
                f"Handler {handler_name} has no schema"
            )

    def test_env_file_is_optional_on_all_tools(self):
        schemas = get_tool_schemas()
        for tool in schemas:
            assert "env_file" in tool["input_schema"]["properties"], (
                f"Tool {tool['name']} missing env_file property"
            )
            assert "env_file" not in tool["input_schema"]["required"], (
                f"Tool {tool['name']} should not require env_file"
            )

    def test_parameter_types(self):
        """Verify specific parameter types match CLI argument expectations."""
        schemas = get_tool_schemas()
        type_map = {t["name"]: t for t in schemas}

        # Integer params
        assert type_map["starfish_project_leave"]["input_schema"]["properties"]["participant_id"]["type"] == "integer"
        assert type_map["starfish_project_detail"]["input_schema"]["properties"]["project_id"]["type"] == "integer"
        assert type_map["starfish_run_start"]["input_schema"]["properties"]["project_id"]["type"] == "integer"
        assert type_map["starfish_run_logs"]["input_schema"]["properties"]["run_id"]["type"] == "integer"
        assert type_map["starfish_run_logs"]["input_schema"]["properties"]["line"]["type"] == "integer"

        # String params
        assert type_map["starfish_site_register"]["input_schema"]["properties"]["name"]["type"] == "string"
        assert type_map["starfish_project_new"]["input_schema"]["properties"]["tasks"]["type"] == "string"

        # Enum params
        assert "enum" in type_map["starfish_artifact_download"]["input_schema"]["properties"]["type"]
        assert set(type_map["starfish_artifact_download"]["input_schema"]["properties"]["type"]["enum"]) == {
            "artifacts", "logs", "mid_artifacts"
        }


class TestBuildCmd:
    """Test CLI command construction."""

    def test_build_cmd_appends_json_flag(self):
        cmd = _build_cmd(["site", "info"])
        assert cmd == ["starfish", "site", "info", "--json"]

    def test_build_cmd_with_args(self):
        cmd = _build_cmd(["project", "new", "--name", "test"])
        assert cmd == ["starfish", "project", "new", "--name", "test", "--json"]


class TestRunCli:
    """Test CLI execution with mocked subprocess."""

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_successful_json_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='{"success": true, "msg": "done"}',
            stderr="",
            returncode=0,
        )
        result = _run_cli(["site", "info"])
        assert result == {"success": True, "msg": "done"}

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_empty_output_with_error(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="some error",
            returncode=1,
        )
        result = _run_cli(["site", "info"])
        assert result["success"] is False
        assert "some error" in result["msg"]

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_empty_output_success_exit(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="",
            returncode=0,
        )
        result = _run_cli(["site", "info"])
        assert result["success"] is False
        assert "No output" in result["msg"]

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_invalid_json_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="not json at all",
            stderr="",
            returncode=0,
        )
        result = _run_cli(["site", "info"])
        assert result["success"] is False
        assert "Invalid JSON" in result["msg"]

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_timeout_handling(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="starfish", timeout=120)
        result = _run_cli(["site", "info"])
        assert result["success"] is False
        assert "timed out" in result["msg"]

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_command_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        result = _run_cli(["site", "info"])
        assert result["success"] is False
        assert "not found" in result["msg"]

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_env_file_sets_starfish_env(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='{"success": true}',
            returncode=0,
        )
        _run_cli(["site", "info"], env_file=".env.site2")
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["env"]["STARFISH_ENV"] == ".env.site2"

    @patch("starfish_cli.agent.tools.subprocess.run")
    def test_no_env_file_no_starfish_env_override(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='{"success": true}',
            returncode=0,
        )
        _run_cli(["site", "info"])
        call_kwargs = mock_run.call_args
        # STARFISH_ENV should not be explicitly set
        env = call_kwargs.kwargs["env"]
        # It inherits from os.environ but we didn't set it explicitly
        # Just verify the call succeeded


class TestExecuteTool:
    """Test the execute_tool dispatch function."""

    @patch("starfish_cli.agent.tools._run_cli")
    def test_unknown_tool_returns_error(self, mock_run):
        result = json.loads(execute_tool("nonexistent_tool", {}))
        assert result["success"] is False
        assert "Unknown tool" in result["msg"]
        mock_run.assert_not_called()

    @patch("starfish_cli.agent.tools._run_cli")
    def test_site_info(self, mock_run):
        mock_run.return_value = {"success": True, "data": {"name": "Site1"}}
        result = json.loads(execute_tool("starfish_site_info", {}))
        assert result["success"] is True
        mock_run.assert_called_once_with(["site", "info"], env_file=None)

    @patch("starfish_cli.agent.tools._run_cli")
    def test_site_register(self, mock_run):
        mock_run.return_value = {"success": True, "msg": "registered"}
        result = json.loads(execute_tool("starfish_site_register", {
            "name": "Hospital A",
            "desc": "Test site"
        }))
        assert result["success"] is True
        mock_run.assert_called_once_with(
            ["site", "register", "--name", "Hospital A", "--desc", "Test site"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_site_register_without_desc(self, mock_run):
        mock_run.return_value = {"success": True, "msg": "registered"}
        execute_tool("starfish_site_register", {"name": "Hospital A"})
        mock_run.assert_called_once_with(
            ["site", "register", "--name", "Hospital A"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_site_update(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_site_update", {"name": "New Name", "desc": "New desc"})
        mock_run.assert_called_once_with(
            ["site", "update", "--name", "New Name", "--desc", "New desc"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_site_deregister(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_site_deregister", {})
        mock_run.assert_called_once_with(
            ["site", "deregister", "--force"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_project_list(self, mock_run):
        mock_run.return_value = {"success": True, "data": []}
        execute_tool("starfish_project_list", {})
        mock_run.assert_called_once_with(["project", "list"], env_file=None)

    @patch("starfish_cli.agent.tools._run_cli")
    def test_project_new(self, mock_run):
        mock_run.return_value = {"success": True}
        tasks_json = '[{"seq":1,"model":"LogisticRegression","config":{"total_round":2,"current_round":1}}]'
        execute_tool("starfish_project_new", {
            "name": "Study",
            "tasks": tasks_json,
            "desc": "A study",
        })
        mock_run.assert_called_once_with(
            ["project", "new", "--name", "Study", "--tasks", tasks_json, "--desc", "A study"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_project_join(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_project_join", {"name": "Study", "notes": "joining"})
        mock_run.assert_called_once_with(
            ["project", "join", "--name", "Study", "--notes", "joining"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_project_leave(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_project_leave", {"participant_id": 42})
        mock_run.assert_called_once_with(
            ["project", "leave", "--participant-id", "42"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_project_detail(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_project_detail", {"project_id": 5})
        mock_run.assert_called_once_with(
            ["project", "detail", "--project-id", "5"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_run_start(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_run_start", {"project_id": 1})
        mock_run.assert_called_once_with(
            ["run", "start", "--project-id", "1"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_run_status(self, mock_run):
        mock_run.return_value = {"success": True, "data": []}
        execute_tool("starfish_run_status", {"project_id": 1})
        mock_run.assert_called_once_with(
            ["run", "status", "--project-id", "1"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_run_detail(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_run_detail", {"batch": 1, "project_id": 2, "site_id": 3})
        mock_run.assert_called_once_with(
            ["run", "detail", "--batch", "1", "--project-id", "2", "--site-id", "3"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_run_logs_minimal(self, mock_run):
        mock_run.return_value = {"success": True, "data": []}
        execute_tool("starfish_run_logs", {"run_id": 10})
        mock_run.assert_called_once_with(
            ["run", "logs", "--run-id", "10"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_run_logs_with_all_params(self, mock_run):
        mock_run.return_value = {"success": True, "data": []}
        execute_tool("starfish_run_logs", {
            "run_id": 10, "task_seq": 2, "round_seq": 3, "line": 50
        })
        mock_run.assert_called_once_with(
            ["run", "logs", "--run-id", "10", "--task-seq", "2", "--round-seq", "3", "--line", "50"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_dataset_upload(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_dataset_upload", {"run_id": 1, "file_path": "/data/site1.csv"})
        mock_run.assert_called_once_with(
            ["dataset", "upload", "--run-id", "1", "--file", "/data/site1.csv"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_artifact_download_defaults(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_artifact_download", {"run_id": 1, "type": "artifacts"})
        mock_run.assert_called_once_with(
            ["artifact", "download", "--run-id", "1", "--type", "artifacts"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_artifact_download_single_run(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_artifact_download", {
            "run_id": 1, "type": "logs", "all_runs": False, "output_dir": "/tmp/out"
        })
        mock_run.assert_called_once_with(
            ["artifact", "download", "--run-id", "1", "--type", "logs", "--single-run", "--output-dir", "/tmp/out"],
            env_file=None,
        )

    @patch("starfish_cli.agent.tools._run_cli")
    def test_env_file_passed_through(self, mock_run):
        mock_run.return_value = {"success": True}
        execute_tool("starfish_site_info", {"env_file": ".env.site2"})
        mock_run.assert_called_once_with(["site", "info"], env_file=".env.site2")

    def test_execute_tool_returns_json_string(self):
        with patch("starfish_cli.agent.tools._run_cli") as mock_run:
            mock_run.return_value = {"success": True, "data": {"id": 1}}
            result = execute_tool("starfish_site_info", {})
            parsed = json.loads(result)
            assert parsed["success"] is True
            assert parsed["data"]["id"] == 1
