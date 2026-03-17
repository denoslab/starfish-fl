"""
Tool definitions that map starfish CLI commands to LLM agent tools.

Each tool has a schema (for the LLM) and an execute function (runs the CLI).
"""

import json
import os
import subprocess


def _build_cmd(args: list[str], env_file: str | None = None) -> list[str]:
    """Build a starfish CLI command list."""
    return ["starfish"] + args + ["--json"]


def _run_cli(args: list[str], env_file: str | None = None) -> dict:
    """
    Execute a starfish CLI command and return parsed JSON output.

    Parameters
    ----------
    args : list[str]
        CLI arguments (e.g. ["site", "info"])
    env_file : str | None
        Optional path to .env file for multi-site support

    Returns
    -------
    dict
        Parsed JSON response from the CLI, or an error dict.
    """
    cmd = _build_cmd(args, env_file)
    env = os.environ.copy()
    if env_file:
        env["STARFISH_ENV"] = env_file

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        stdout = result.stdout.strip()
        if not stdout:
            if result.returncode != 0:
                return {"success": False, "msg": result.stderr.strip() or f"Command failed with exit code {result.returncode}"}
            return {"success": False, "msg": "No output from command"}
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"success": False, "msg": f"Invalid JSON output: {result.stdout[:500]}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "msg": "Command timed out after 120 seconds"}
    except FileNotFoundError:
        return {"success": False, "msg": "starfish CLI not found. Is it installed?"}


# ---------------------------------------------------------------------------
# Tool schemas for the LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    # ---- Site tools ----
    {
        "name": "starfish_site_info",
        "description": "Get information about the current site's registration status with the router.",
        "input_schema": {
            "type": "object",
            "properties": {
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support (e.g. '.env.site2')"
                }
            },
            "required": []
        }
    },
    {
        "name": "starfish_site_register",
        "description": "Register the current site with the Starfish router. Must be done before creating or joining projects.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for this site"
                },
                "desc": {
                    "type": "string",
                    "description": "Optional description for this site"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "starfish_site_update",
        "description": "Update the current site's name and description.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "New name for this site"
                },
                "desc": {
                    "type": "string",
                    "description": "New description for this site"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "starfish_site_deregister",
        "description": "Deregister the current site from the router. This is destructive and removes the site entirely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": []
        }
    },
    # ---- Project tools ----
    {
        "name": "starfish_project_list",
        "description": "List all federated learning projects this site is involved in.",
        "input_schema": {
            "type": "object",
            "properties": {
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": []
        }
    },
    {
        "name": "starfish_project_new",
        "description": "Create a new federated learning project. This site becomes the coordinator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Project name"
                },
                "desc": {
                    "type": "string",
                    "description": "Project description"
                },
                "tasks": {
                    "type": "string",
                    "description": "Tasks as JSON string, e.g. '[{\"seq\":1,\"model\":\"LogisticRegression\",\"config\":{\"total_round\":5,\"current_round\":1}}]'"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["name", "tasks"]
        }
    },
    {
        "name": "starfish_project_join",
        "description": "Join an existing project as a participant site.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the project to join"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes when joining"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "starfish_project_leave",
        "description": "Leave a project using your participant ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "participant_id": {
                    "type": "integer",
                    "description": "Your participant ID in the project"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["participant_id"]
        }
    },
    {
        "name": "starfish_project_detail",
        "description": "Show detailed information about a project including its participants.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "integer",
                    "description": "Project ID"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["project_id"]
        }
    },
    # ---- Run tools ----
    {
        "name": "starfish_run_start",
        "description": "Start a new federated learning run batch for a project. Only the coordinator site can do this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "integer",
                    "description": "Project ID to start a run for"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "starfish_run_status",
        "description": "Show all runs and their statuses for a project. Use this to monitor run progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "integer",
                    "description": "Project ID"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["project_id"]
        }
    },
    {
        "name": "starfish_run_detail",
        "description": "Show detailed information for a specific run batch including per-site run info.",
        "input_schema": {
            "type": "object",
            "properties": {
                "batch": {
                    "type": "integer",
                    "description": "Batch number"
                },
                "project_id": {
                    "type": "integer",
                    "description": "Project ID"
                },
                "site_id": {
                    "type": "integer",
                    "description": "Site ID"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["batch", "project_id", "site_id"]
        }
    },
    {
        "name": "starfish_run_logs",
        "description": "Fetch logs for a specific run. Useful for diagnosing failures.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Run ID"
                },
                "task_seq": {
                    "type": "integer",
                    "description": "Task sequence number (default: 1)"
                },
                "round_seq": {
                    "type": "integer",
                    "description": "Round sequence number (default: 1)"
                },
                "line": {
                    "type": "integer",
                    "description": "Start from line number (default: 0)"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["run_id"]
        }
    },
    # ---- Data tools ----
    {
        "name": "starfish_dataset_upload",
        "description": "Upload a dataset CSV file for a run. This changes the run status from Standby to Preparing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Run ID to upload dataset for"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the dataset CSV file"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["run_id", "file_path"]
        }
    },
    {
        "name": "starfish_artifact_download",
        "description": "Download artifacts or logs for a run as a ZIP file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Run ID"
                },
                "type": {
                    "type": "string",
                    "enum": ["artifacts", "logs", "mid_artifacts"],
                    "description": "Type of download"
                },
                "all_runs": {
                    "type": "boolean",
                    "description": "Download for all runs in batch (default: true)"
                },
                "output_dir": {
                    "type": "string",
                    "description": "Directory to save the file (default: '.')"
                },
                "env_file": {
                    "type": "string",
                    "description": "Optional path to .env file for multi-site support"
                }
            },
            "required": ["run_id", "type"]
        }
    },
]


def get_tool_schemas() -> list[dict]:
    """Return tool definitions formatted for the Anthropic API."""
    return TOOL_DEFINITIONS


# ---------------------------------------------------------------------------
# Tool execution dispatch
# ---------------------------------------------------------------------------

def _execute_site_info(params: dict) -> dict:
    return _run_cli(["site", "info"], env_file=params.get("env_file"))


def _execute_site_register(params: dict) -> dict:
    args = ["site", "register", "--name", params["name"]]
    if params.get("desc"):
        args += ["--desc", params["desc"]]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_site_update(params: dict) -> dict:
    args = ["site", "update", "--name", params["name"]]
    if params.get("desc"):
        args += ["--desc", params["desc"]]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_site_deregister(params: dict) -> dict:
    return _run_cli(["site", "deregister", "--force"], env_file=params.get("env_file"))


def _execute_project_list(params: dict) -> dict:
    return _run_cli(["project", "list"], env_file=params.get("env_file"))


def _execute_project_new(params: dict) -> dict:
    args = ["project", "new", "--name", params["name"], "--tasks", params["tasks"]]
    if params.get("desc"):
        args += ["--desc", params["desc"]]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_project_join(params: dict) -> dict:
    args = ["project", "join", "--name", params["name"]]
    if params.get("notes"):
        args += ["--notes", params["notes"]]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_project_leave(params: dict) -> dict:
    args = ["project", "leave", "--participant-id", str(params["participant_id"])]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_project_detail(params: dict) -> dict:
    args = ["project", "detail", "--project-id", str(params["project_id"])]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_run_start(params: dict) -> dict:
    args = ["run", "start", "--project-id", str(params["project_id"])]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_run_status(params: dict) -> dict:
    args = ["run", "status", "--project-id", str(params["project_id"])]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_run_detail(params: dict) -> dict:
    args = [
        "run", "detail",
        "--batch", str(params["batch"]),
        "--project-id", str(params["project_id"]),
        "--site-id", str(params["site_id"]),
    ]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_run_logs(params: dict) -> dict:
    args = ["run", "logs", "--run-id", str(params["run_id"])]
    if params.get("task_seq") is not None:
        args += ["--task-seq", str(params["task_seq"])]
    if params.get("round_seq") is not None:
        args += ["--round-seq", str(params["round_seq"])]
    if params.get("line") is not None:
        args += ["--line", str(params["line"])]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_dataset_upload(params: dict) -> dict:
    args = ["dataset", "upload", "--run-id", str(params["run_id"]), "--file", params["file_path"]]
    return _run_cli(args, env_file=params.get("env_file"))


def _execute_artifact_download(params: dict) -> dict:
    args = ["artifact", "download", "--run-id", str(params["run_id"]), "--type", params["type"]]
    if params.get("all_runs") is False:
        args += ["--single-run"]
    if params.get("output_dir"):
        args += ["--output-dir", params["output_dir"]]
    return _run_cli(args, env_file=params.get("env_file"))


TOOL_HANDLERS = {
    "starfish_site_info": _execute_site_info,
    "starfish_site_register": _execute_site_register,
    "starfish_site_update": _execute_site_update,
    "starfish_site_deregister": _execute_site_deregister,
    "starfish_project_list": _execute_project_list,
    "starfish_project_new": _execute_project_new,
    "starfish_project_join": _execute_project_join,
    "starfish_project_leave": _execute_project_leave,
    "starfish_project_detail": _execute_project_detail,
    "starfish_run_start": _execute_run_start,
    "starfish_run_status": _execute_run_status,
    "starfish_run_detail": _execute_run_detail,
    "starfish_run_logs": _execute_run_logs,
    "starfish_dataset_upload": _execute_dataset_upload,
    "starfish_artifact_download": _execute_artifact_download,
}


def execute_tool(tool_name: str, params: dict) -> str:
    """
    Execute a tool by name with the given parameters.

    Returns JSON string of the result.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return json.dumps({"success": False, "msg": f"Unknown tool: {tool_name}"})
    result = handler(params)
    return json.dumps(result, indent=2)
