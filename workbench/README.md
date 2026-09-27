# Starfish Workbench

> **Note**: This is a component of the Starfish federated learning platform. For the complete system overview, see the [main README](../README.md).

Local development and test environment for Starfish based on docker-compose.

This workbench provides a unified environment to run and test all Starfish components together.

## Overview

The workbench orchestrates the following components:
- **Router**: Routing server for coordinating federated learning
- **Controller**: Site management and FL task execution (includes R runtime for R-based tasks)
- **PostgreSQL**: Database for the router
- **Redis**: Cache and message broker for the controller
- **OpenClaw**: LLM-powered agent framework for autonomous orchestration (optional but included)

All components are configured to work together out of the box.

> **Note:** The controller Docker image includes R and compiled R packages (`jsonlite`, `survival`, `mice`) for R-based FL tasks. The first build may take longer due to R package compilation.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Make](https://www.gnu.org/software/make/) utility (usually pre-installed on Linux/macOS)

## Quick Start

### Compile

We are using [make](https://www.gnu.org/software/make/manual/make.html) utility for easier maintenance. 

Run `make build` to compile all services:
```bash
make build
```

Or specify `make router` or `make controller` to only compile and build docker image for specific service.

### Run and Stop

Start all services and dependencies:
```bash
make up
```

#### First Time Setup

If it is a brand new environment or a clean database, you need to create the database and a superuser:

1. **Create the database:**
   ```bash
   ./init_db.sh
   ```

2. **Restart services** (if needed):
   ```bash
   docker-compose restart router
   ```

3. **Run migrations and create superuser:**
   ```bash
   docker-compose exec -it router bash
   ```

   Then inside the container:
   ```bash
   poetry run python3 manage.py makemigrations
   poetry run python3 manage.py migrate
   poetry run python3 manage.py createsuperuser
   ```

4. **Configure credentials:**
   
   Make sure the username and password you created match what's configured in `config/controller.env`.

#### Stop Services

To stop all services:
```bash
make stop
```

To stop and remove containers:
```bash
make down
```

## BabelBrain FL Profile

A separate stack for the BabelBrain federated learning work: a router, a coordinator site and two participant sites, each reading its own synthetic BabelBrain sample store. It is defined in `compose/babelbrain.yaml` and uses its own Compose project, `starfish-babelbrain`, and its own volumes.

- Controllers use the image `starfish-controller-babelbrain`, built with `INSTALL_TORCH=cpu`, so CPU-only PyTorch and no CUDA libraries.
- No agent code runs. `STARFISH_DISABLE_AGENTS=1` is set on the router and every controller, no `ANTHROPIC_API_KEY` is passed, and the `BabelBrainFno` task refuses agent hooks on its own.
- A one-shot `store-init` service writes one synthetic store per site: 20 train and 4 val samples at 250 kHz, with a different seed per site. Each site mounts only its own store, read-only, at `/babelbrain-store`, and finds it through `BABELBRAIN_FL_STORE`.
- The coordinator, site a, also mounts a held-out eval store of 6 samples at `/babelbrain-eval-store`, `BABELBRAIN_FL_EVAL_STORE`, which the SF-05 evaluation gate scores every candidate model on.

| Site | Role | Port | Redis DB | Store volume |
| --- | --- | --- | --- | --- |
| a | Coordinator | 8001 | 1 | `bb_store_a` |
| b | Participant | 8002 | 2 | `bb_store_b` |
| c | Participant | 8003 | 3 | `bb_store_c` |

The router is on port 8000, with user `admin` and password `1234`. Stop the default and e2e stacks first; they use the same ports.

```bash
make babelbrain-up      # build if needed, start, wait until healthy
make babelbrain-check   # per site: no anthropic, no agent module, own store readable and read-only
make babelbrain-e2e     # a BabelBrainFno run through the router API, no dataset upload
make babelbrain-logs
make babelbrain-down    # stop, keep volumes
make babelbrain-clean   # stop and delete volumes, including the stores
```

### Site tokens, SF-09

Sites can authenticate with their own token instead of the router's superuser account. The admin issues a single-use enrolment code, the site enrols with it and gets a token once, and the controller then sends `ROUTER_TOKEN`. A site on a token can read and write only its own runs; a coordinator's run can also read its batch. Superuser Basic auth keeps working.

```bash
# as the admin: a code valid for 72 hours, optionally for one project
curl -u admin:1234 -X POST http://localhost:8000/starfish/api/v1/enrolment-codes/ \
     -H 'Content-Type: application/json' -d '{"project": 1, "note": "NeuroFUS site B"}'
# as the site: enrol once and keep the token
curl -X POST http://localhost:8000/starfish/api/v1/sites/enrol/ -H 'Content-Type: application/json' \
     -d '{"code": "<code>", "uid": "<site uuid>", "name": "site-b", "description": "..."}'
# as the admin: list and revoke tokens
curl -u admin:1234 http://localhost:8000/starfish/api/v1/site-tokens/
curl -u admin:1234 -X POST http://localhost:8000/starfish/api/v1/site-tokens/<id>/revoke/
```

`make babelbrain-tokens` enrols all three workbench sites, restarts their controllers on tokens through `BB_TOKEN_A`, `BB_TOKEN_B` and `BB_TOKEN_C`, runs 2 rounds, and switches back. Outside the workbench, run the router with `DEBUG` off or `STARFISH_REQUIRE_TLS=1`, which turns on HTTPS redirects, secure cookies and HSTS, behind a proxy that sets `X-Forwarded-Proto`.

`make babelbrain-e2e` needs `pip install -r ../e2e/requirements.txt`. It runs 3 BabelBrainFno rounds on the stand-in model and checks that every run ends `Success`, that each site read its own store and trained, that the router holds one global model per round, and that no log holds a store path. `make babelbrain-transfer` sends a 2 GB file to the router and back and checks peak memory on both sides.

## Configuration

Environment variables are managed in the `config/` directory:
- `config/router.env` - Router configuration
- `config/controller.env` - Controller configuration

Update these files with your specific settings before running the services.

## Accessing Services

Once running, you can access:
- **Router API**: http://localhost:8000/starfish/api/v1/
- **Controller Web UI**: http://localhost:8001/
- **OpenClaw UI**: http://localhost:18789/ (see [OpenClaw documentation](docs/openclaw.md) for setup details)

## Troubleshooting

### Port Conflicts

If you encounter port conflicts (e.g., Redis port 6379 already in use):
```bash
sudo systemctl stop redis
```

### Database Issues

If the database connection fails, ensure PostgreSQL is running:
```bash
docker-compose ps postgres
```

### Viewing Logs

View logs for all services:
```bash
docker-compose logs -f
```

View logs for a specific service:
```bash
docker-compose logs -f router
docker-compose logs -f controller
```

## Development Workflow

1. Make code changes in `../controller` or `../router` directories
2. Rebuild the specific service:
   ```bash
   make controller  # or make router
   ```
3. Restart the service:
   ```bash
   docker-compose restart controller  # or router
   ```

## Additional Information

For more details about each component, see:
- [Controller Documentation](../controller/README.md)
- [Router Documentation](../router/README.md)
- [Main Starfish Documentation](../README.md) 
- [OpenClaw Documentation](docs/openclaw.md)

