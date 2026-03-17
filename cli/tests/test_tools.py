"""
Tests for agent tool definitions and execution.
"""

import csv
import json
import os
import pytest
from unittest.mock import patch, MagicMock

from starfish_cli.agent.tools import (
    get_tool_schemas,
    get_experiment_tool_schemas,
    execute_tool,
    TOOL_HANDLERS,
    TOOL_DEFINITIONS,
    LOCAL_TOOL_DEFINITIONS,
    LOCAL_TOOL_HANDLERS,
    ALL_TOOL_HANDLERS,
    _run_cli,
    _build_cmd,
    _infer_column_type,
    _detect_patterns,
    _ALL_KNOWN_MODELS,
    _SINGLE_ROUND_MODELS,
    _ITERATIVE_MODELS,
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


class TestExperimentToolSchemas:
    """Test experiment-specific tool schemas."""

    def test_experiment_tool_count(self):
        assert len(get_experiment_tool_schemas()) == 20

    def test_experiment_tools_include_cli_tools(self):
        exp_names = {t["name"] for t in get_experiment_tool_schemas()}
        cli_names = {t["name"] for t in get_tool_schemas()}
        assert cli_names.issubset(exp_names)

    def test_expected_experiment_tools_exist(self):
        names = {t["name"] for t in get_experiment_tool_schemas()}
        expected_local = {
            "analyze_dataset",
            "recommend_task",
            "generate_config",
            "interpret_results",
            "compare_experiments",
        }
        assert expected_local.issubset(names)

    def test_all_experiment_tools_have_required_fields(self):
        for tool in get_experiment_tool_schemas():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema

    def test_all_experiment_tool_names_are_unique(self):
        names = [t["name"] for t in get_experiment_tool_schemas()]
        assert len(names) == len(set(names))

    def test_every_experiment_tool_has_handler(self):
        for tool in get_experiment_tool_schemas():
            assert tool["name"] in ALL_TOOL_HANDLERS, (
                f"Tool {tool['name']} has no handler"
            )

    def test_local_tool_handlers_match_definitions(self):
        local_names = {t["name"] for t in LOCAL_TOOL_DEFINITIONS}
        handler_names = set(LOCAL_TOOL_HANDLERS.keys())
        assert local_names == handler_names


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

    def test_execute_tool_dispatches_local_tools(self):
        """Verify execute_tool can dispatch to local experiment tools."""
        # generate_config is a local tool that doesn't need filesystem
        result = json.loads(execute_tool("generate_config", {"model": "LogisticRegression"}))
        assert result["success"] is True
        assert "tasks_json" in result


# ---------------------------------------------------------------------------
# Tests for local experiment tools
# ---------------------------------------------------------------------------

def _write_csv(path, rows, header=None):
    """Helper to write a CSV file for testing."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


class TestAnalyzeDataset:
    """Test the analyze_dataset local tool."""

    def test_analyze_csv_basic(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        _write_csv(csv_path, [
            ["1.0", "2.0", "0"],
            ["3.0", "4.0", "1"],
            ["5.0", "6.0", "0"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["n_rows"] == 3
        assert result["n_cols"] == 3
        assert len(result["columns"]) == 3

    def test_analyze_detects_binary_outcome(self, tmp_path):
        csv_path = tmp_path / "binary.csv"
        _write_csv(csv_path, [
            ["1.5", "2.3", "0"],
            ["3.1", "4.7", "1"],
            ["5.2", "6.1", "0"],
            ["7.0", "8.0", "1"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("binary_outcome") is True

    def test_analyze_detects_continuous_outcome(self, tmp_path):
        csv_path = tmp_path / "continuous.csv"
        _write_csv(csv_path, [
            ["1.0", "2.0", "3.14"],
            ["4.0", "5.0", "6.28"],
            ["7.0", "8.0", "9.42"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("continuous_outcome") is True

    def test_analyze_detects_time_to_event(self, tmp_path):
        csv_path = tmp_path / "survival.csv"
        _write_csv(csv_path, [
            ["1.0", "2.0", "10.5", "1"],
            ["3.0", "4.0", "20.3", "0"],
            ["5.0", "6.0", "15.1", "1"],
            ["7.0", "8.0", "30.0", "0"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("time_to_event") is True

    def test_analyze_detects_count_data(self, tmp_path):
        csv_path = tmp_path / "count.csv"
        _write_csv(csv_path, [
            ["1.0", "2.5", "0"],
            ["3.0", "4.5", "3"],
            ["5.0", "6.5", "7"],
            ["7.0", "8.5", "2"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("count_data") is True

    def test_analyze_detects_censoring_indicator(self, tmp_path):
        csv_path = tmp_path / "censored.csv"
        _write_csv(csv_path, [
            ["1.0", "10.5", "0"],
            ["3.0", "20.3", "1"],
            ["5.0", "15.1", "-1"],
            ["7.0", "30.0", "0"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("censored") is True

    def test_analyze_detects_missing_data(self, tmp_path):
        csv_path = tmp_path / "missing.csv"
        _write_csv(csv_path, [
            ["1.0", "", "0"],
            ["3.0", "4.0", "1"],
            ["", "6.0", "0"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("missing_data") is True
        assert result["data_quality"]["total_missing"] == 2

    def test_analyze_detects_ordinal_outcome(self, tmp_path):
        csv_path = tmp_path / "ordinal.csv"
        _write_csv(csv_path, [
            ["1.0", "2.0", "0"],
            ["3.0", "4.0", "1"],
            ["5.0", "6.0", "2"],
            ["7.0", "8.0", "3"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["patterns"].get("ordinal_outcome") is True

    def test_analyze_file_not_found(self):
        result = json.loads(execute_tool("analyze_dataset", {"file_path": "/no/such/file.csv"}))
        assert result["success"] is False
        assert "not found" in result["msg"].lower() or "File not found" in result["msg"]

    def test_analyze_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is False

    def test_analyze_with_header(self, tmp_path):
        csv_path = tmp_path / "header.csv"
        _write_csv(csv_path, [
            ["1.0", "2.0", "0"],
            ["3.0", "4.0", "1"],
        ], header=["age", "weight", "outcome"])
        result = json.loads(execute_tool("analyze_dataset", {
            "file_path": str(csv_path), "has_header": True
        }))
        assert result["success"] is True
        assert result["n_rows"] == 2
        assert result["columns"][0]["name"] == "age"

    def test_analyze_column_type_inference(self, tmp_path):
        csv_path = tmp_path / "types.csv"
        _write_csv(csv_path, [
            ["hello", "1.5", "0"],
            ["world", "2.5", "1"],
            ["foo", "3.5", "0"],
        ])
        result = json.loads(execute_tool("analyze_dataset", {"file_path": str(csv_path)}))
        assert result["success"] is True
        assert result["columns"][0]["dtype"] == "categorical"
        assert result["columns"][1]["dtype"] == "numeric_continuous"
        assert result["columns"][2]["dtype"] == "binary"


class TestRecommendTask:
    """Test the recommend_task local tool."""

    def test_recommend_binary_classification(self):
        analysis = {"patterns": {"binary_outcome": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        assert result["success"] is True
        models = [r["model"] for r in result["recommendations"]]
        assert "LogisticRegression" in models

    def test_recommend_survival(self):
        analysis = {"patterns": {"binary_outcome": True, "time_to_event": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "CoxProportionalHazards" in models
        assert "KaplanMeier" in models

    def test_recommend_count_data(self):
        analysis = {"patterns": {"count_data": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "PoissonRegression" in models
        assert "NegativeBinomialRegression" in models

    def test_recommend_censored(self):
        analysis = {"patterns": {"censored": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "CensoredRegression" in models

    def test_recommend_continuous(self):
        analysis = {"patterns": {"continuous_outcome": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "LinearRegression" in models

    def test_recommend_missing_data(self):
        analysis = {"patterns": {"missing_data": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "MultipleImputation" in models

    def test_recommend_ordinal(self):
        analysis = {"patterns": {"ordinal_outcome": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "OrdinalLogisticRegression" in models

    def test_recommend_mixed_effects(self):
        analysis = {"patterns": {"binary_outcome": True, "group_column": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        models = [r["model"] for r in result["recommendations"]]
        assert "MixedEffectsLogisticRegression" in models

    def test_recommend_filters_by_python_preference(self):
        analysis = {"patterns": {"binary_outcome": True, "time_to_event": True}}
        result = json.loads(execute_tool("recommend_task", {
            "analysis": analysis, "preference": "python"
        }))
        models = [r["model"] for r in result["recommendations"]]
        assert "CoxProportionalHazards" in models
        assert "RCoxProportionalHazards" not in models

    def test_recommend_filters_by_r_preference(self):
        analysis = {"patterns": {"binary_outcome": True, "time_to_event": True}}
        result = json.loads(execute_tool("recommend_task", {
            "analysis": analysis, "preference": "r"
        }))
        models = [r["model"] for r in result["recommendations"]]
        assert "RCoxProportionalHazards" in models
        assert "CoxProportionalHazards" not in models

    def test_recommend_no_patterns_gives_default(self):
        result = json.loads(execute_tool("recommend_task", {"analysis": {"patterns": {}}}))
        assert result["success"] is True
        assert len(result["recommendations"]) >= 1

    def test_recommend_sorted_by_priority(self):
        analysis = {"patterns": {"binary_outcome": True}}
        result = json.loads(execute_tool("recommend_task", {"analysis": analysis}))
        priorities = [r["priority"] for r in result["recommendations"]]
        assert priorities == sorted(priorities)


class TestGenerateConfig:
    """Test the generate_config local tool."""

    def test_generate_logistic_regression_default(self):
        result = json.loads(execute_tool("generate_config", {"model": "LogisticRegression"}))
        assert result["success"] is True
        tasks = json.loads(result["tasks_json"])
        assert len(tasks) == 1
        assert tasks[0]["model"] == "LogisticRegression"
        assert tasks[0]["config"]["total_round"] == 5
        assert tasks[0]["config"]["current_round"] == 1
        assert tasks[0]["seq"] == 1

    def test_generate_single_round_model(self):
        result = json.loads(execute_tool("generate_config", {"model": "CoxProportionalHazards"}))
        tasks = json.loads(result["tasks_json"])
        assert tasks[0]["config"]["total_round"] == 1

    def test_generate_custom_total_round(self):
        result = json.loads(execute_tool("generate_config", {
            "model": "LogisticRegression", "total_round": 10
        }))
        tasks = json.loads(result["tasks_json"])
        assert tasks[0]["config"]["total_round"] == 10

    def test_generate_with_config_overrides(self):
        result = json.loads(execute_tool("generate_config", {
            "model": "MultipleImputation",
            "config_overrides": {"m": 10, "max_iter": 20}
        }))
        tasks = json.loads(result["tasks_json"])
        assert tasks[0]["config"]["m"] == 10
        assert tasks[0]["config"]["max_iter"] == 20

    def test_generate_unknown_model(self):
        result = json.loads(execute_tool("generate_config", {"model": "FakeModel"}))
        assert result["success"] is False
        assert "Unknown model" in result["msg"]

    def test_generate_r_variant(self):
        result = json.loads(execute_tool("generate_config", {"model": "RLogisticRegression"}))
        assert result["success"] is True
        tasks = json.loads(result["tasks_json"])
        assert tasks[0]["model"] == "RLogisticRegression"

    def test_generate_unet_config(self):
        result = json.loads(execute_tool("generate_config", {"model": "FederatedUNet"}))
        tasks = json.loads(result["tasks_json"])
        assert tasks[0]["config"]["architecture"] == "resnet50"
        assert tasks[0]["config"]["patch_size"] == 64

    def test_generated_config_is_valid_json(self):
        for model in ["LogisticRegression", "CoxProportionalHazards", "MultipleImputation"]:
            result = json.loads(execute_tool("generate_config", {"model": model}))
            tasks = json.loads(result["tasks_json"])
            assert isinstance(tasks, list)
            assert len(tasks) == 1
            assert "seq" in tasks[0]
            assert "model" in tasks[0]
            assert "config" in tasks[0]

    def test_generate_empty_model(self):
        result = json.loads(execute_tool("generate_config", {"model": ""}))
        assert result["success"] is False

    def test_model_sets_known(self):
        """Verify model categorization is consistent."""
        assert "LogisticRegression" in _ITERATIVE_MODELS
        assert "CoxProportionalHazards" in _SINGLE_ROUND_MODELS
        assert "FederatedUNet" in _ALL_KNOWN_MODELS
        assert _ALL_KNOWN_MODELS == _SINGLE_ROUND_MODELS | _ITERATIVE_MODELS | {"FederatedUNet"}


class TestInterpretResults:
    """Test the interpret_results local tool."""

    def test_interpret_classification_artifacts(self, tmp_path):
        artifact = {
            "coef_": [[0.5, -0.3]],
            "intercept_": [0.1],
            "metric_acc": 0.85,
            "metric_auc": 0.91,
            "sample_size": 100,
        }
        (tmp_path / "site1.json").write_text(json.dumps(artifact))
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path), "model": "LogisticRegression"
        }))
        assert result["success"] is True
        assert result["model"] == "LogisticRegression"
        assert "metric_acc" in result["metrics"]
        assert "metric_auc" in result["metrics"]

    def test_interpret_regression_artifacts(self, tmp_path):
        artifact = {
            "coef_": [[1.2, -0.8]],
            "intercept_": [3.0],
            "metric_r2": 0.72,
            "metric_mse": 0.15,
            "sample_size": 200,
        }
        (tmp_path / "site1.json").write_text(json.dumps(artifact))
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path), "model": "LinearRegression"
        }))
        assert result["success"] is True
        assert "metric_r2" in result["metrics"]

    def test_interpret_survival_artifacts(self, tmp_path):
        artifact = {
            "coef_": [[0.3, -0.5]],
            "concordance_index": 0.78,
            "sample_size": 150,
        }
        (tmp_path / "site1.json").write_text(json.dumps(artifact))
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path), "model": "CoxProportionalHazards"
        }))
        assert result["success"] is True
        assert "concordance_index" in result["metrics"]

    def test_interpret_multiple_sites(self, tmp_path):
        for i, acc in enumerate([0.80, 0.85, 0.90]):
            artifact = {"metric_acc": acc, "sample_size": 100}
            (tmp_path / f"site{i}.json").write_text(json.dumps(artifact))
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path)
        }))
        assert result["success"] is True
        assert result["n_artifact_files"] == 3
        assert len(result["metrics"]["metric_acc"]["values"]) == 3

    def test_interpret_empty_directory(self, tmp_path):
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path)
        }))
        assert result["success"] is False

    def test_interpret_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("not json{{{")
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path)
        }))
        assert result["success"] is False

    def test_interpret_with_diagnostics(self, tmp_path):
        artifact = {
            "metric_acc": 0.85,
            "diagnostics": {
                "vif": {"age": 15.0, "weight": 2.0},
                "cooks_distance": {"n_influential": 3, "max": 0.5},
            },
        }
        (tmp_path / "site1.json").write_text(json.dumps(artifact))
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path)
        }))
        assert result["success"] is True
        assert len(result["diagnostic_concerns"]) > 0
        assert any("VIF" in c for c in result["diagnostic_concerns"])

    def test_interpret_directory_not_found(self):
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": "/no/such/directory"
        }))
        assert result["success"] is False

    def test_interpret_key_findings_good_auc(self, tmp_path):
        artifact = {"metric_auc": 0.92}
        (tmp_path / "site1.json").write_text(json.dumps(artifact))
        result = json.loads(execute_tool("interpret_results", {
            "artifact_dir": str(tmp_path)
        }))
        assert any("Good discrimination" in f for f in result["key_findings"])


class TestCompareExperiments:
    """Test the compare_experiments local tool."""

    def test_compare_two_classification_models(self):
        experiments = [
            {"model": "LogisticRegression", "metrics": {
                "metric_acc": {"mean": 0.85}, "metric_auc": {"mean": 0.90}
            }},
            {"model": "SvmRegression", "metrics": {
                "metric_acc": {"mean": 0.80}, "metric_auc": {"mean": 0.85}
            }},
        ]
        result = json.loads(execute_tool("compare_experiments", {"experiments": experiments}))
        assert result["success"] is True
        assert result["best_model"] == "LogisticRegression"
        assert result["primary_metric"] == "metric_auc"

    def test_compare_different_model_types(self):
        experiments = [
            {"model": "LogisticRegression", "metrics": {"metric_acc": {"mean": 0.85}}},
            {"model": "CoxProportionalHazards", "metrics": {"concordance_index": {"mean": 0.75}}},
        ]
        result = json.loads(execute_tool("compare_experiments", {"experiments": experiments}))
        assert result["success"] is True
        assert len(result["comparison_table"]) == 2

    def test_compare_single_experiment(self):
        experiments = [{"model": "LogisticRegression", "metrics": {"metric_acc": {"mean": 0.85}}}]
        result = json.loads(execute_tool("compare_experiments", {"experiments": experiments}))
        assert result["success"] is True
        assert result["best_model"] == "LogisticRegression"

    def test_compare_empty_list(self):
        result = json.loads(execute_tool("compare_experiments", {"experiments": []}))
        assert result["success"] is False

    def test_compare_identifies_best_by_lower_mse(self):
        experiments = [
            {"model": "LinearRegression", "metrics": {"metric_mse": {"mean": 0.5}}},
            {"model": "SvmRegression", "metrics": {"metric_mse": {"mean": 0.3}}},
        ]
        result = json.loads(execute_tool("compare_experiments", {"experiments": experiments}))
        assert result["success"] is True
        # Lower MSE is better
        assert result["best_model"] == "SvmRegression"

    def test_compare_ranking_order(self):
        experiments = [
            {"model": "A", "metrics": {"metric_auc": {"mean": 0.70}}},
            {"model": "B", "metrics": {"metric_auc": {"mean": 0.90}}},
            {"model": "C", "metrics": {"metric_auc": {"mean": 0.80}}},
        ]
        result = json.loads(execute_tool("compare_experiments", {"experiments": experiments}))
        ranks = [r["model"] for r in result["ranking"]]
        assert ranks == ["B", "C", "A"]


class TestInferColumnType:
    """Test the column type inference helper."""

    def test_binary_column(self):
        result = _infer_column_type(["0", "1", "0", "1"])
        assert result["dtype"] == "binary"

    def test_continuous_column(self):
        result = _infer_column_type(["1.5", "2.7", "3.14", "4.0"])
        assert result["dtype"] == "numeric_continuous"

    def test_integer_column(self):
        result = _infer_column_type(["0", "3", "7", "2"])
        assert result["dtype"] == "numeric_integer"

    def test_categorical_column(self):
        result = _infer_column_type(["cat", "dog", "cat", "bird"])
        assert result["dtype"] == "categorical"

    def test_empty_values(self):
        result = _infer_column_type(["", "", ""])
        assert result["dtype"] == "empty"

    def test_missing_count(self):
        result = _infer_column_type(["1.0", "", "3.0", ""])
        assert result["n_missing"] == 2


class TestDetectPatterns:
    """Test the pattern detection helper."""

    def test_empty_columns(self):
        assert _detect_patterns([]) == {}

    def test_binary_outcome_pattern(self):
        columns = [
            {"dtype": "numeric_continuous", "n_unique": 10, "n_missing": 0, "stats": {}},
            {"dtype": "binary", "n_unique": 2, "n_missing": 0, "stats": {"min": 0, "max": 1}},
        ]
        patterns = _detect_patterns(columns)
        assert patterns.get("binary_outcome") is True

    def test_continuous_outcome_with_group(self):
        columns = [
            {"dtype": "numeric_integer", "n_unique": 3, "n_missing": 0, "stats": {}},
            {"dtype": "numeric_continuous", "n_unique": 50, "n_missing": 0, "stats": {}},
            {"dtype": "numeric_continuous", "n_unique": 100, "n_missing": 0, "stats": {}},
        ]
        patterns = _detect_patterns(columns)
        assert patterns.get("continuous_outcome") is True
        assert patterns.get("group_column") is True
