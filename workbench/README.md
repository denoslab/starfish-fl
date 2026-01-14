# Starfish Workbench

> **Note**: This is a component of the Starfish federated learning platform. For the complete system overview, see the [main README](../README.md).

Local development and test environment for Starfish based on docker-compose.

This workbench provides a unified environment to run and test all Starfish components together.

## Overview

The workbench orchestrates the following components:
- **Router**: Routing server for coordinating federated learning
- **Controller**: Site management and FL task execution  
- **PostgreSQL**: Database for the router
- **Redis**: Cache and message broker for the controller

All components are configured to work together out of the box.

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

## Configuration

Environment variables are managed in the `config/` directory:
- `config/router.env` - Router configuration
- `config/controller.env` - Controller configuration

Update these files with your specific settings before running the services.

## Accessing Services

Once running, you can access:
- **Router API**: http://localhost:8000/starfish/api/v1/
- **Controller Web UI**: http://localhost:8001/

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

