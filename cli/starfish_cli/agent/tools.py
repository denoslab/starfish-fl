"""
Tool definitions that map starfish CLI commands to LLM agent tools.

Each tool has a schema (for the LLM) and an execute function (runs the CLI).
Experiment-specific local tools (analyze_dataset, recommend_task, etc.) perform
direct Python operations without invoking the CLI.
"""

import csv
import json
import os
import statistics
import subprocess
from collections import Counter
from pathlib import Path


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


# ---------------------------------------------------------------------------
# Experiment-specific local tools (no CLI subprocess)
# ---------------------------------------------------------------------------

LOCAL_TOOL_DEFINITIONS = [
    {
        "name": "analyze_dataset",
        "description": (
            "Analyze a local CSV file to understand its structure, column types, "
            "and statistical properties. Returns column metadata, basic statistics, "
            "and detected patterns (binary outcome, time-to-event, count data, etc.) "
            "to help choose the right FL task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file to analyze"
                },
                "has_header": {
                    "type": "boolean",
                    "description": "Whether the CSV has a header row (default: false, matching Starfish convention)"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "recommend_task",
        "description": (
            "Recommend suitable Starfish FL task types based on dataset analysis. "
            "Takes the output from analyze_dataset and returns ranked task "
            "recommendations with rationale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "description": "Dataset analysis result from analyze_dataset tool"
                },
                "preference": {
                    "type": "string",
                    "enum": ["python", "r", "any"],
                    "description": "Language preference for task implementation (default: 'any')"
                }
            },
            "required": ["analysis"]
        }
    },
    {
        "name": "generate_config",
        "description": (
            "Generate a Starfish task configuration JSON array for a given model "
            "type. Returns the complete tasks JSON string ready for project creation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model class name (e.g., 'LogisticRegression', 'CoxProportionalHazards')"
                },
                "total_round": {
                    "type": "integer",
                    "description": "Number of federated rounds (default depends on model type)"
                },
                "config_overrides": {
                    "type": "object",
                    "description": "Additional config parameters (e.g., {'m': 10} for MultipleImputation)"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "interpret_results",
        "description": (
            "Parse Starfish artifact files from a directory and produce a "
            "structured interpretation of the FL experiment results. Reads JSON "
            "artifact files and extracts model coefficients, metrics, and diagnostics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_dir": {
                    "type": "string",
                    "description": "Directory containing downloaded artifact files"
                },
                "model": {
                    "type": "string",
                    "description": "Model type to help interpret results (e.g., 'LogisticRegression')"
                }
            },
            "required": ["artifact_dir"]
        }
    },
    {
        "name": "compare_experiments",
        "description": (
            "Compare results from multiple FL experiments to identify the best "
            "model. Takes a list of experiment result summaries and returns a "
            "comparative analysis with rankings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of experiment result objects (from interpret_results)"
                }
            },
            "required": ["experiments"]
        }
    },
]


def _infer_column_type(values: list[str]) -> dict:
    """Infer the type and compute stats for a column of string values."""
    non_empty = [v for v in values if v.strip() != ""]
    n_missing = len(values) - len(non_empty)

    if not non_empty:
        return {"dtype": "empty", "n_unique": 0, "n_missing": n_missing, "stats": {}}

    # Try to parse as numeric
    numeric_vals = []
    for v in non_empty:
        try:
            numeric_vals.append(float(v))
        except (ValueError, TypeError):
            break

    if len(numeric_vals) == len(non_empty):
        # All values are numeric
        unique = set(numeric_vals)
        n_unique = len(unique)
        is_integer = all(v == int(v) for v in numeric_vals)

        if is_integer and unique == {0, 1}:
            dtype = "binary"
        elif is_integer and n_unique <= 20:
            dtype = "numeric_integer"
        else:
            dtype = "numeric_continuous"

        stats = {
            "min": min(numeric_vals),
            "max": max(numeric_vals),
            "mean": round(statistics.mean(numeric_vals), 4),
        }
        if len(numeric_vals) >= 2:
            stats["std"] = round(statistics.stdev(numeric_vals), 4)

        return {
            "dtype": dtype,
            "n_unique": n_unique,
            "n_missing": n_missing,
            "stats": stats,
        }

    # Non-numeric: categorical
    counter = Counter(non_empty)
    top_3 = counter.most_common(3)
    return {
        "dtype": "categorical",
        "n_unique": len(counter),
        "n_missing": n_missing,
        "stats": {"top_values": [{"value": v, "count": c} for v, c in top_3]},
    }


def _detect_patterns(columns: list[dict]) -> dict:
    """Detect data patterns from column analysis to guide task selection."""
    if not columns:
        return {}

    patterns = {}
    last = columns[-1]
    second_last = columns[-2] if len(columns) >= 2 else None

    # Binary outcome: last column is binary (0/1)
    if last["dtype"] == "binary":
        patterns["binary_outcome"] = True

        # Time-to-event: second-to-last is positive continuous, last is binary
        if second_last and second_last["dtype"] == "numeric_continuous":
            if second_last["stats"].get("min", -1) >= 0:
                patterns["time_to_event"] = True

        # Group/cluster column: first column has few unique values
        first = columns[0]
        if first["dtype"] in ("numeric_integer", "categorical") and first["n_unique"] <= 20:
            patterns["group_column"] = True

    # Count data: last column is non-negative integer
    elif last["dtype"] == "numeric_integer":
        if last["stats"].get("min", -1) >= 0:
            patterns["count_data"] = True
        # Ordinal outcome: 3+ ordered integer levels
        if last["n_unique"] >= 3:
            patterns["ordinal_outcome"] = True

    # Continuous outcome
    elif last["dtype"] == "numeric_continuous":
        patterns["continuous_outcome"] = True

        # Censoring indicator: second-to-last is continuous, last has {-1, 0, 1}
        if second_last and second_last["dtype"] == "numeric_continuous":
            pass  # Just continuous regression

        # Group column for ANCOVA
        first = columns[0]
        if first["dtype"] in ("numeric_integer", "categorical") and first["n_unique"] <= 20:
            patterns["group_column"] = True

    # Censoring: last column contains only {-1, 0, 1}
    if last["dtype"] == "numeric_integer" and last["n_unique"] <= 3:
        vals = set()
        if last["stats"].get("min") is not None:
            vals.add(int(last["stats"]["min"]))
        if last["stats"].get("max") is not None:
            vals.add(int(last["stats"]["max"]))
        if vals <= {-1, 0, 1} and len(vals) >= 2:
            # Check if second-to-last is continuous (outcome variable)
            if second_last and second_last["dtype"] == "numeric_continuous":
                patterns["censored"] = True

    # Missing data: any column has missing values
    if any(c["n_missing"] > 0 for c in columns):
        patterns["missing_data"] = True

    return patterns


def _execute_analyze_dataset(params: dict) -> dict:
    """Analyze a CSV file and return column metadata and detected patterns."""
    file_path = params.get("file_path", "")
    has_header = params.get("has_header", False)

    if not file_path or not os.path.exists(file_path):
        return {"success": False, "msg": f"File not found: {file_path}"}

    try:
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        return {"success": False, "msg": f"Error reading CSV: {str(e)}"}

    if not rows:
        return {"success": False, "msg": "CSV file is empty"}

    if has_header:
        headers = rows[0]
        data_rows = rows[1:]
    else:
        data_rows = rows
        headers = [f"col_{i}" for i in range(len(rows[0]))]

    if not data_rows:
        return {"success": False, "msg": "CSV file has no data rows"}

    n_cols = len(headers)
    n_rows = len(data_rows)

    # Transpose to get columns
    columns = []
    for col_idx in range(n_cols):
        values = [row[col_idx] if col_idx < len(row) else "" for row in data_rows]
        col_info = _infer_column_type(values)
        col_info["name"] = headers[col_idx]
        col_info["index"] = col_idx
        columns.append(col_info)

    patterns = _detect_patterns(columns)

    # Data quality summary
    total_cells = n_rows * n_cols
    total_missing = sum(c["n_missing"] for c in columns)

    return {
        "success": True,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "patterns": patterns,
        "data_quality": {
            "total_missing": total_missing,
            "pct_missing": round(100 * total_missing / total_cells, 2) if total_cells > 0 else 0,
        },
    }


# Model defaults for config generation
_SINGLE_ROUND_MODELS = {
    "CoxProportionalHazards", "RCoxProportionalHazards",
    "KaplanMeier", "RKaplanMeier",
    "PoissonRegression", "RPoissonRegression",
    "NegativeBinomialRegression", "RNegativeBinomialRegression",
    "CensoredRegression", "RCensoredRegression",
    "MultipleImputation", "RMultipleImputation",
    "Ancova",
    "LogisticRegressionStats",
    "OrdinalLogisticRegression",
    "MixedEffectsLogisticRegression",
}

_ITERATIVE_MODELS = {
    "LogisticRegression", "RLogisticRegression",
    "LinearRegression",
    "SvmRegression",
}

_ALL_KNOWN_MODELS = _SINGLE_ROUND_MODELS | _ITERATIVE_MODELS | {"FederatedUNet"}

_MODEL_EXTRA_DEFAULTS = {
    "MultipleImputation": {"m": 5, "max_iter": 10},
    "RMultipleImputation": {"m": 5, "max_iter": 10},
    "MixedEffectsLogisticRegression": {"vcp_p": 1.0, "fe_p": 2.0},
    "FederatedUNet": {
        "local_epochs": 1,
        "architecture": "resnet50",
        "type_Unet": "unet",
        "patch_size": 64,
        "batch_size": 1,
        "learning_rate": 0.0001,
    },
}


def _execute_recommend_task(params: dict) -> dict:
    """Recommend FL task types based on dataset analysis patterns."""
    analysis = params.get("analysis", {})
    preference = params.get("preference", "any")
    patterns = analysis.get("patterns", {})

    if not patterns:
        return {
            "success": True,
            "recommendations": [{
                "model": "LogisticRegression",
                "rationale": "Default recommendation — no clear patterns detected",
                "priority": 5,
            }],
        }

    recommendations = []

    def _add(model, rationale, priority, r_variant=None):
        if preference == "r" and r_variant:
            recommendations.append({"model": r_variant, "rationale": rationale, "priority": priority})
        elif preference == "python":
            recommendations.append({"model": model, "rationale": rationale, "priority": priority})
        else:
            recommendations.append({"model": model, "rationale": rationale, "priority": priority})
            if r_variant:
                recommendations.append({"model": r_variant, "rationale": rationale + " (R variant)", "priority": priority + 1})

    if patterns.get("time_to_event"):
        _add("CoxProportionalHazards",
             "Time-to-event data detected (continuous time + binary event indicator)",
             1, "RCoxProportionalHazards")
        _add("KaplanMeier",
             "Non-parametric survival estimation for time-to-event data",
             2, "RKaplanMeier")

    if patterns.get("missing_data"):
        _add("MultipleImputation",
             "Missing data detected — MICE handles incomplete observations",
             1, "RMultipleImputation")

    if patterns.get("censored"):
        _add("CensoredRegression",
             "Censoring indicator detected (values in {-1, 0, 1})",
             1, "RCensoredRegression")

    if patterns.get("binary_outcome") and not patterns.get("time_to_event"):
        if patterns.get("group_column"):
            _add("MixedEffectsLogisticRegression",
                 "Binary outcome with group/cluster column — multilevel model appropriate",
                 1)
        _add("LogisticRegression",
             "Binary outcome (0/1) detected — standard classification",
             2, "RLogisticRegression")
        _add("LogisticRegressionStats",
             "Binary outcome with statistical inference (p-values, odds ratios, CI)",
             3)

    if patterns.get("ordinal_outcome") and not patterns.get("count_data"):
        _add("OrdinalLogisticRegression",
             "Ordinal integer outcome with 3+ levels detected",
             2)

    if patterns.get("count_data"):
        _add("PoissonRegression",
             "Non-negative integer outcome detected — count data model",
             1, "RPoissonRegression")
        _add("NegativeBinomialRegression",
             "Alternative for overdispersed count data",
             2, "RNegativeBinomialRegression")

    if patterns.get("continuous_outcome"):
        if patterns.get("group_column"):
            _add("Ancova",
                 "Continuous outcome with group column — ANCOVA for group comparisons",
                 1)
        _add("LinearRegression",
             "Continuous outcome detected — standard regression",
             2)

    if not recommendations:
        recommendations.append({
            "model": "LogisticRegression",
            "rationale": "Default recommendation — no specific patterns matched",
            "priority": 5,
        })

    recommendations.sort(key=lambda r: r["priority"])
    return {"success": True, "recommendations": recommendations}


def _execute_generate_config(params: dict) -> dict:
    """Generate task configuration JSON for a given model type."""
    model = params.get("model", "")
    if not model:
        return {"success": False, "msg": "Model name is required"}
    if model not in _ALL_KNOWN_MODELS:
        return {"success": False, "msg": f"Unknown model: {model}. Known models: {sorted(_ALL_KNOWN_MODELS)}"}

    # Determine default total_round
    if model in _SINGLE_ROUND_MODELS:
        default_rounds = 1
    elif model == "FederatedUNet":
        default_rounds = 5
    else:
        default_rounds = 5

    total_round = params.get("total_round", default_rounds)
    config = {"total_round": total_round, "current_round": 1}

    # Apply model-specific defaults
    if model in _MODEL_EXTRA_DEFAULTS:
        config.update(_MODEL_EXTRA_DEFAULTS[model])

    # Apply user overrides
    overrides = params.get("config_overrides")
    if overrides and isinstance(overrides, dict):
        config.update(overrides)

    tasks_json = json.dumps([{"seq": 1, "model": model, "config": config}])
    return {"success": True, "tasks_json": tasks_json, "config": config}


def _execute_interpret_results(params: dict) -> dict:
    """Parse artifact files from a directory and interpret experiment results."""
    artifact_dir = params.get("artifact_dir", "")
    model = params.get("model", "")

    if not artifact_dir or not os.path.isdir(artifact_dir):
        return {"success": False, "msg": f"Directory not found: {artifact_dir}"}

    # Walk directory for JSON files
    results = []
    dir_path = Path(artifact_dir)
    json_files = sorted(dir_path.rglob("*.json")) + sorted(dir_path.rglob("*artifacts*"))

    # Deduplicate
    seen = set()
    unique_files = []
    for f in json_files:
        if f.resolve() not in seen and f.is_file():
            seen.add(f.resolve())
            unique_files.append(f)

    if not unique_files:
        # Try reading files without extension (artifacts are often extensionless)
        for f in sorted(dir_path.rglob("*")):
            if f.is_file() and f.resolve() not in seen:
                seen.add(f.resolve())
                unique_files.append(f)

    parsed_files = []
    for fpath in unique_files:
        try:
            text = fpath.read_text().strip()
            if not text:
                continue
            data = json.loads(text)
            parsed_files.append({"file": str(fpath.relative_to(dir_path)), "data": data})
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    if not parsed_files:
        return {"success": False, "msg": "No parseable artifact files found in directory"}

    # Extract metrics and coefficients
    all_metrics = {}
    all_coefficients = {}
    all_diagnostics = {}

    for pf in parsed_files:
        data = pf["data"]
        file_name = pf["file"]

        # Collect known metric keys
        for key, val in data.items():
            if key.startswith("metric_") and isinstance(val, (int, float)):
                all_metrics.setdefault(key, []).append({"file": file_name, "value": val})
            elif key == "coef_":
                all_coefficients[file_name] = val
            elif key == "intercept_":
                all_coefficients.setdefault(file_name + "_intercept", val)
            elif key == "diagnostics" and isinstance(val, dict):
                all_diagnostics[file_name] = val
            elif key == "concordance_index" and isinstance(val, (int, float)):
                all_metrics.setdefault("concordance_index", []).append({"file": file_name, "value": val})
            elif key == "sample_size" and isinstance(val, (int, float)):
                all_metrics.setdefault("sample_size", []).append({"file": file_name, "value": val})

    # Compute metric summaries
    metric_summary = {}
    for metric_name, values in all_metrics.items():
        vals = [v["value"] for v in values]
        metric_summary[metric_name] = {
            "mean": round(statistics.mean(vals), 4) if vals else None,
            "values": values,
        }

    # Build key findings
    key_findings = []
    for metric_name, summary in metric_summary.items():
        mean_val = summary["mean"]
        if mean_val is None:
            continue
        if "acc" in metric_name and mean_val > 0.8:
            key_findings.append(f"Good accuracy: {metric_name} = {mean_val}")
        elif "acc" in metric_name:
            key_findings.append(f"Moderate accuracy: {metric_name} = {mean_val}")
        if "auc" in metric_name and mean_val > 0.8:
            key_findings.append(f"Good discrimination: {metric_name} = {mean_val}")
        elif "auc" in metric_name:
            key_findings.append(f"Moderate discrimination: {metric_name} = {mean_val}")
        if "r2" in metric_name and mean_val > 0.6:
            key_findings.append(f"Reasonable fit: {metric_name} = {mean_val}")
        elif "r2" in metric_name:
            key_findings.append(f"Weak fit: {metric_name} = {mean_val}")
        if metric_name == "concordance_index" and mean_val > 0.7:
            key_findings.append(f"Good concordance: {mean_val}")
        elif metric_name == "concordance_index":
            key_findings.append(f"Moderate concordance: {mean_val}")

    # Check diagnostics for concerns
    diagnostic_concerns = []
    for file_name, diag in all_diagnostics.items():
        vif = diag.get("vif", {})
        if isinstance(vif, dict):
            high_vif = {k: v for k, v in vif.items() if isinstance(v, (int, float)) and v > 10}
            if high_vif:
                diagnostic_concerns.append(f"High VIF (multicollinearity) in {file_name}: {high_vif}")
        cooks = diag.get("cooks_distance", {})
        if isinstance(cooks, dict) and cooks.get("n_influential", 0) > 0:
            diagnostic_concerns.append(
                f"Influential outliers detected in {file_name}: {cooks['n_influential']} points"
            )

    return {
        "success": True,
        "model": model or "unknown",
        "n_artifact_files": len(parsed_files),
        "metrics": metric_summary,
        "coefficients": all_coefficients,
        "diagnostics": all_diagnostics,
        "key_findings": key_findings,
        "diagnostic_concerns": diagnostic_concerns,
    }


def _execute_compare_experiments(params: dict) -> dict:
    """Compare results from multiple FL experiments."""
    experiments = params.get("experiments", [])
    if not experiments:
        return {"success": False, "msg": "No experiments to compare"}
    if len(experiments) == 1:
        return {
            "success": True,
            "comparison": "Only one experiment provided",
            "best_model": experiments[0].get("model", "unknown"),
            "ranking": [{"rank": 1, "model": experiments[0].get("model", "unknown")}],
        }

    # Collect all metric names across experiments
    all_metric_names = set()
    for exp in experiments:
        metrics = exp.get("metrics", {})
        all_metric_names.update(metrics.keys())

    # Build comparison table
    comparison_table = []
    for exp in experiments:
        model_name = exp.get("model", "unknown")
        metrics = exp.get("metrics", {})
        row = {"model": model_name}
        for metric_name in sorted(all_metric_names):
            metric_data = metrics.get(metric_name, {})
            row[metric_name] = metric_data.get("mean") if isinstance(metric_data, dict) else None
        comparison_table.append(row)

    # Determine primary metric for ranking
    primary_metric = None
    metric_priority = [
        "metric_auc", "metric_acc", "concordance_index",
        "metric_r2", "metric_mse", "metric_mae",
    ]
    for m in metric_priority:
        if m in all_metric_names:
            primary_metric = m
            break

    # Rank experiments
    ranking = []
    if primary_metric:
        # Higher is better for auc, acc, r2, concordance; lower for mse, mae
        reverse = primary_metric not in ("metric_mse", "metric_mae")
        scored = []
        for row in comparison_table:
            val = row.get(primary_metric)
            scored.append((row["model"], val if val is not None else float("-inf") if reverse else float("inf")))
        scored.sort(key=lambda x: x[1], reverse=reverse)
        ranking = [{"rank": i + 1, "model": m, primary_metric: v} for i, (m, v) in enumerate(scored)]
        best_model = ranking[0]["model"] if ranking else "unknown"
    else:
        ranking = [{"rank": i + 1, "model": row["model"]} for i, row in enumerate(comparison_table)]
        best_model = comparison_table[0]["model"] if comparison_table else "unknown"

    return {
        "success": True,
        "comparison_table": comparison_table,
        "primary_metric": primary_metric,
        "best_model": best_model,
        "ranking": ranking,
    }


LOCAL_TOOL_HANDLERS = {
    "analyze_dataset": _execute_analyze_dataset,
    "recommend_task": _execute_recommend_task,
    "generate_config": _execute_generate_config,
    "interpret_results": _execute_interpret_results,
    "compare_experiments": _execute_compare_experiments,
}

ALL_TOOL_HANDLERS = {**TOOL_HANDLERS, **LOCAL_TOOL_HANDLERS}


def get_experiment_tool_schemas() -> list[dict]:
    """Return all tool definitions including experiment-specific local tools."""
    return TOOL_DEFINITIONS + LOCAL_TOOL_DEFINITIONS


def execute_tool(tool_name: str, params: dict) -> str:
    """
    Execute a tool by name with the given parameters.

    Returns JSON string of the result.
    """
    handler = ALL_TOOL_HANDLERS.get(tool_name)
    if not handler:
        return json.dumps({"success": False, "msg": f"Unknown tool: {tool_name}"})
    result = handler(params)
    return json.dumps(result, indent=2)
