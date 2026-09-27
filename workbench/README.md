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

`make babelbrain-e2e` needs `pip install -r ../e2e/requirements.txt`. Until SF-04 adds training, the run it starts ends `Failed` at the training step on purpose; the test checks that each site read its own store and that no log holds a store path.

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

