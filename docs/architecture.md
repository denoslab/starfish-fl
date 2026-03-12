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
| **Task** | Individual ML operation within a project (e.g., `CoxProportionalHazards`) |

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

## Router API

Base URL: `http://localhost:8000/starfish/api/v1/`

Auth: HTTP Basic Auth

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sites/` | GET/POST | Site registration |
| `/sites/heartbeat/` | POST | Site liveness signal |
| `/projects/` | GET/POST | Project management |
| `/runs/` | POST | Create runs for a project batch |
| `/runs/{id}/status/` | PUT | State transitions |
| `/runs-action/upload/` | POST | Upload artifacts/logs |
| `/runs-action/download/` | GET | Download artifacts (zipped) |
