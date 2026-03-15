# Getting Started

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start (All Components)

1. **Clone the repository**
    ```bash
    git clone https://github.com/denoslab/starfish-fl.git
    cd starfish-fl
    ```

2. **Start all services using Workbench**
    ```bash
    cd workbench
    make build
    make up
    ```

3. **Initialize the database** (first time only)
    ```bash
    ./init_db.sh
    ```

4. **Create superuser for router** (first time only)
    ```bash
    docker exec -it starfish-router poetry run python3 manage.py makemigrations
    docker exec -it starfish-router poetry run python3 manage.py migrate
    docker exec -it starfish-router poetry run python3 manage.py createsuperuser
    ```
    Make sure the username and password match what's configured in `workbench/config/controller.env`.

5. **Access the applications**
    - Router API: [http://localhost:8000/starfish/api/v1/](http://localhost:8000/starfish/api/v1/)
    - Controller Web Interface: [http://localhost:8001/](http://localhost:8001/)

6. **Stop the services**
    ```bash
    make stop    # Stop services
    make down    # Stop and remove containers
    ```

## Local Development Without Docker

### Router (requires PostgreSQL)

```bash
cd router
poetry install
python3 manage.py migrate
python3 manage.py createsuperuser
python3 manage.py runserver
```

### Controller (requires Redis)

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

### CLI

```bash
cd cli
cp .env.example .env   # fill in SITE_UID, ROUTER_URL, ROUTER_USERNAME, ROUTER_PASSWORD, CONTROLLER_URL
poetry install
poetry run starfish --help
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10.10, Django 4.2 |
| Task Queue | Celery + Redis |
| Databases | PostgreSQL (Router), SQLite (Controller) |
| Python ML | scikit-learn, NumPy, Pandas, statsmodels, scipy, lifelines |
| Image Segmentation (optional) | TensorFlow 2.15, Keras, segmentation-models, Pillow |
| R Runtime | R 4.x with `jsonlite`, `survival`, `mice`, `MASS` |
| Containers | Docker, Docker Compose |
| Dependency Management | Poetry |

## Configuration

Each component uses a `.env` file (copy from `.env.example`):

### Controller

| Variable | Description |
|----------|-------------|
| `SITE_UID` | Unique UUID for this site |
| `ROUTER_URL` | URL of the routing server |
| `ROUTER_USERNAME` | Router authentication username |
| `ROUTER_PASSWORD` | Router authentication password |
| `CELERY_BROKER_URL` | Redis connection for Celery |
| `REDIS_HOST` | Redis host for caching |

### Router

| Variable | Description |
|----------|-------------|
| `DATABASE_HOST` | PostgreSQL host |
| `POSTGRES_DB` | Database name |
| `POSTGRES_USER` | Database username |
| `POSTGRES_PASSWORD` | Database password |
| `SECRET_KEY` | Django secret key |
| `DEBUG` | Debug mode (False in production) |

## Running Tests

```bash
# Router
docker exec -it starfish-router poetry run python3 manage.py test

# Controller
docker exec -it starfish-controller poetry run python3 manage.py test
```
