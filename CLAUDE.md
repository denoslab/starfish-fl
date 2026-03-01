# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Starfish-FL is an agentic federated learning (FL) framework organized as a mono repo with three components:

- **`controller/`** — Django app installed on every site; handles local ML training, Celery task queuing, and a web UI (port 8001)
- **`router/`** — Django REST Framework app acting as central coordination server; manages global state, message forwarding, and artifact storage (port 8000)
- **`cli/`** — Typer-based CLI (`starfish` command) that replicates web portal functionality for human and AI agent use
- **`workbench/`** — Docker Compose orchestration for running all components together in development

## Commands

### Development Setup (Full Stack)

```bash
cd workbench
make build          # Build all Docker images
make up             # Start all services (detached)
make stop           # Stop services
make down           # Stop and remove containers
```

First-time initialization after `make up`:
```bash
./init_db.sh        # Create PostgreSQL database
docker-compose exec -it router bash
# Inside container:
poetry run python3 manage.py migrate
poetry run python3 manage.py createsuperuser
```

### Running Tests

```bash
# Router (inside container or local venv)
cd router && python3 manage.py test

# Controller
cd controller && python3 manage.py test

# Via Docker
docker exec -it starfish-router poetry run python3 manage.py test
docker exec -it starfish-controller poetry run python3 manage.py test
```

### Local Development Without Docker

**Router** (requires PostgreSQL):
```bash
cd router
poetry install
python3 manage.py migrate
python3 manage.py createsuperuser
python3 manage.py runserver
```

**Controller** (requires Redis):
```bash
cd controller
poetry install
python3 manage.py migrate
# Start Celery workers in separate terminals:
celery -A starfish beat -l DEBUG
celery -A starfish worker -l DEBUG -Q starfish.run
celery -A starfish worker -l DEBUG --concurrency=1 -Q starfish.processor
python3 manage.py runserver
```

**CLI**:
```bash
cd cli
cp .env.example .env   # fill in SITE_UID, ROUTER_URL, ROUTER_USERNAME, ROUTER_PASSWORD, CONTROLLER_URL
poetry install
poetry run starfish --help
```

### Code Formatting

```bash
autopep8 --exclude='*/migrations/*' --in-place --recursive ./starfish/
```

### Router Background Jobs

```bash
python3 manage.py runjob check_site_status
```

## Architecture

### Data Model (Router)

The Router owns the authoritative data model:
- **Site** — a participant node with a unique UUID (`uid`), connects via heartbeat
- **Project** — defines FL tasks (stored as JSON), owned by a site (coordinator)
- **ProjectParticipant** — links a Site to a Project with a role (`CO`=coordinator, `PA`=participant)
- **Run** — execution instance of a Project for one batch; uses `django-fsm` for state machine transitions

Run state machine: `STANDBY → PREPARING → RUNNING → PENDING_SUCCESS → PENDING_AGGREGATING → AGGREGATING → SUCCESS` (or `→ FAILED` at any point)

### FL Task Execution Flow (Controller)

Each ML task inherits from `AbstractTask` (`controller/starfish/controller/tasks/abstract_task.py`). Task lifecycle methods map to Run states:
1. `standby()` → validate previous round, notify router
2. `preparing()` → load/validate data (`prepare_data()`)
3. `running()` → execute training (`training()`)
4. `pending_success()` → upload mid-artifacts and logs to router
5. `pending_aggregating()` → coordinator downloads all participant artifacts
6. `aggregating()` → coordinator calls `do_aggregate()`, uploads result, loops or finishes

The Controller uses two Celery queues: `starfish.run` (polling/heartbeat) and `starfish.processor` (task execution).

### Adding a New ML Task

1. Create a new file in `controller/starfish/controller/tasks/` (or `tasks/stats_models/` for statistical models)
2. Subclass `AbstractTask` and implement: `validate()`, `prepare_data()`, `training()`, `do_aggregate()`
3. Register the task class in `controller/starfish/controller/tasks/__init__.py`
4. Document the task config schema in `controller/TASK_GUIDE.md`

### Router API

Base URL: `http://localhost:8000/starfish/api/v1/`
Auth: HTTP Basic Auth with superuser credentials

Key endpoints:
- `GET/POST /sites/` — site registration
- `POST /sites/heartbeat/` — site liveness signal
- `GET/POST /projects/` — project management
- `POST /runs/` — create runs for a project batch
- `PUT /runs/{id}/status/` — state transitions (called by Controller tasks)
- `POST /runs-action/upload/` — upload artifacts/logs
- `GET /runs-action/download/` — download artifacts (zipped)

### Configuration

Each component uses a `.env` file (copy from `.env.example`):

- Controller: `SITE_UID`, `ROUTER_URL`, `ROUTER_USERNAME`, `ROUTER_PASSWORD`, `CELERY_BROKER_URL`, `REDIS_HOST`
- Router: `DATABASE_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `SECRET_KEY`, `DEBUG`
- CLI: same as Controller plus `CONTROLLER_URL`; supports `STARFISH_ENV` env var to point to an alternate `.env` file

### Tech Stack

- Python 3.10.10, Django 4.2, Poetry for dependency management
- Controller DB: SQLite; Router DB: PostgreSQL
- Task queue: Celery + Redis
- ML: scikit-learn, numpy, pandas, statsmodels, scipy
- Router state machine: `django-fsm`
