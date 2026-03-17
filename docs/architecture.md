# Architecture

## System Overview

Starfish-FL uses a hub-and-spoke architecture where a central **Router** coordinates communication between distributed **Controller** sites.

```
Site A (Controller)  ──┐
                       ├──  Router (Coordination Server)
Site B (Controller)  ──┘
```

- **Sites** run a Controller and can act as a **Coordinator** or **Participant**
- The **Router** maintains global state and forwards messages -- it never sees raw data
- Raw data stays local; only model parameters and summary statistics are exchanged

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Site** | A participant node with a unique UUID, connects via heartbeat |
| **Project** | Defines FL tasks (stored as JSON), owned by a coordinator site |
| **Run** | Execution instance of a project for one batch |
| **Task** | Individual ML operation within a project (e.g., `CoxProportionalHazards`, `FederatedUNet`) |

## Data Model (Router)

- **Site** -- participant node with unique `uid`
- **Project** -- FL task definitions, owned by a site (coordinator)
- **ProjectParticipant** -- links a Site to a Project with role (`CO`=coordinator, `PA`=participant)
- **Run** -- execution instance; uses `django-fsm` for state machine transitions

## Run State Machine

```
STANDBY --> PREPARING --> RUNNING --> PENDING_SUCCESS --> PENDING_AGGREGATING --> AGGREGATING --> SUCCESS
    |           |           |              |                    |                   |
    +-----------+-----------+--------------+--------------------+-------------------+--> FAILED
```

## FL Task Execution Flow

Each ML task inherits from `AbstractTask` (Python) or `AbstractRTask` (R). The lifecycle methods map to Run states:

1. **`standby()`** -- Validate previous round, notify router
2. **`preparing()`** -- Load and validate data via `prepare_data()`
3. **`running()`** -- Execute training via `training()`
4. **`pending_success()`** -- Upload mid-artifacts and logs to router
5. **`pending_aggregating()`** -- Coordinator downloads all participant artifacts
6. **`aggregating()`** -- Coordinator calls `do_aggregate()`, uploads result, loops or finishes

### Agent-in-the-Loop Hooks (Controller)

The Controller's `AbstractTask` includes optional LLM hook points that fire at key lifecycle stages. All hooks are no-ops when the agent is disabled or the API key is absent.

| Hook | When | What it does |
|------|------|-------------|
| `post_training` | After `training()`, before uploading | Generates per-site round summaries, flags anomalies |
| `pre_aggregation` | Before `do_aggregate()` (coordinator) | Compares cross-site artifacts, detects outliers |
| `post_aggregation` | After `do_aggregate()` (coordinator) | Evaluates convergence, recommends early stopping |
| `on_failure` | In `pending_failed()` | Diagnoses failure root cause, suggests recovery |

Enable via task config:

```json
{"agent": {"enabled": true, "summaries": true, "early_stopping": true, "outlier_detection": true}}
```

The Controller uses two Celery queues:

- `starfish.run` -- polling and heartbeat
- `starfish.processor` -- task execution

## Adding a New ML Task

### Python Tasks

1. Create a directory: `controller/starfish/controller/tasks/<task_name>/`
2. Subclass `AbstractTask` and implement: `validate()`, `prepare_data()`, `training()`, `do_aggregate()`
3. Add diagnostics via `from starfish.controller.tasks.diagnostics import ...`
4. Document the task config schema in `controller/TASK_GUIDE.md`

### R Tasks

1. Create a directory: `controller/starfish/controller/tasks/r_<task_name>/` with a `scripts/` subdirectory
2. Implement `prepare_data.R`, `training.R`, `aggregate.R` in `scripts/`
3. Source shared diagnostics: `source(file.path(dirname(this_script), "..", "..", "r_diagnostics_utils.R"))`
4. Create `task.py` extending `AbstractRTask` (sets `r_script_dir`)

!!! note "Dynamic Task Discovery"
    Tasks are discovered automatically via dynamic import. The model name in CamelCase is converted to snake_case to find the module: `CensoredRegression` -> `censored_regression/task.py`. No explicit registration is required.

## Embedded Agent (Router)

The Router includes an optional LLM-powered agent layer that hooks into FSM state transitions to make intelligent orchestration decisions. All features are opt-in per project via the `agent_config` JSON field.

### Agent Features

| Feature | Hook Point | Description |
|---------|-----------|-------------|
| **Aggregation Advisor** | `AGGREGATING` transition | Analyzes mid-artifacts from all sites, detects outliers, and advises on aggregation strategy |
| **Scheduling Advisor** | `SUCCESS` (post-aggregation) | Evaluates convergence and recommends early stopping |
| **Failure Triage** | `FAILED` transition | Diagnoses failures and provides actionable recovery suggestions |

### Enabling the Agent

Set `agent_config` on a project:

```json
{
  "enabled": true,
  "aggregation": true,
  "scheduling": true,
  "triage": true
}
```

Requires `ANTHROPIC_API_KEY` environment variable. Without it, all agent features gracefully degrade to default (no-op) behavior.

### Agent Data Flow

```
State Transition → Agent Hook → LLM Query → Store Result on Run/Project
                                    ↓
                              (on failure: use default, no-op)
```

Agent results are stored in:
- `Run.agent_advice` — aggregation and scheduling recommendations
- `Run.agent_diagnosis` — failure triage results
- `Project.agent_log` — history of all agent decisions

## Router API

Base URL: `http://localhost:8000/starfish/api/v1/`

Auth: HTTP Basic Auth

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/users/` | GET/POST | User management |
| `/groups/` | GET/POST | Group management |
| `/sites/` | GET/POST | Site registration |
| `/sites/{id}/` | GET/PUT/PATCH/DELETE | Site detail |
| `/sites/lookup/` | GET | Lookup site by `uid` |
| `/sites/heartbeat/` | POST | Site liveness signal |
| `/projects/` | GET/POST | Project management |
| `/projects/{id}/` | GET/PUT/PATCH/DELETE | Project detail |
| `/projects/lookup/` | GET | Lookup projects by `site_id` or `name` |
| `/project-participants/` | GET/POST | Manage project participants |
| `/project-participants/lookup/` | GET | Get participants by `project` |
| `/runs/` | GET/POST | List runs / bulk-create runs for a project batch |
| `/runs/{id}/` | GET/PUT/PATCH | Run detail |
| `/runs/{id}/status/` | PUT | State transitions |
| `/runs/lookup/` | GET | Lookup runs by `project`, `batch_id`, `site_uid` |
| `/runs/active/` | GET | Get all active runs |
| `/runs/detail/` | GET | Get run details by `batch`, `project`, `site` |
| `/runs-action/upload/` | POST | Upload artifacts/logs |
| `/runs-action/download/` | GET | Download artifacts (zipped) |
| `/runs-action/update/` | PUT | Update run status by action |
